"""Grammar → wire format: stats, layer specs, HTML document."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import copy

from plot3.encode import encode_codes, encode_norm, pack_u16, pack_u32, pack_u8
from plot3.mathtext import split_math
from plot3.geoms import (
    _Geom,
    aes,
    coord_3d,
    coord_equal,
    coord_polar,
    geom_col,
    scale_colour_continuous,
)
from plot3.scales import Scale, col_values, fmt_num, resolution

from plot3.stats3d import isosurface_levels, regular_grid_mesh
from plot3.table import (
    category_labels,
    count_by,
    filter_equal,
    get_columns,
    group_pieces,
    has_column,
    materialize_columns,
    n_rows,
    numeric_array,
    require_columns,
    subsample_rows,
    unique_levels,
)
from plot3.themes import _CONT_PALETTES, _THEMES as THEMES


def copy_geom_with_density_n(geom: _Geom, n: int) -> _Geom:
    out = copy.copy(geom)
    out._density_n = int(n)
    return out


def _boxplot_stats(values: np.ndarray, coef: float = 1.5):
    """Tukey five-number box + outliers (ggplot2 / geom_boxplot default)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    if coef <= 0:
        return (
            float(values.min()),
            float(q1),
            float(med),
            float(q3),
            float(values.max()),
            np.asarray([], dtype=np.float64),
        )
    iqr = q3 - q1
    lo_fence = q1 - coef * iqr
    hi_fence = q3 + coef * iqr
    inside = values[(values >= lo_fence) & (values <= hi_fence)]
    ymin = float(inside.min()) if inside.size else float(q1)
    ymax = float(inside.max()) if inside.size else float(q3)
    outliers = values[(values < lo_fence) | (values > hi_fence)]
    return ymin, float(q1), float(med), float(q3), ymax, outliers


def _kde_1d(
    values: np.ndarray,
    *,
    n: int = 512,
    adjust: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE on a regular grid (Scott bandwidth × adjust)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = max(8, int(n))
    if x.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if x.size == 1:
        center = float(x[0])
        grid = np.linspace(center - 1.0, center + 1.0, n)
        dens = np.exp(-0.5 * ((grid - center) / 0.2) ** 2)
        dens /= dens.sum() * (grid[1] - grid[0])
        return grid, dens
    std = float(np.std(x, ddof=1)) or 1e-6
    bw = max(1e-9, float(adjust) * 1.06 * std * (x.size ** (-0.2)))
    lo = float(x.min()) - 3.0 * bw
    hi = float(x.max()) + 3.0 * bw
    if hi <= lo:
        hi = lo + 1.0
    grid = np.linspace(lo, hi, n)
    # (n_grid, n_obs)
    u = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * u * u).sum(axis=1) / (
        x.size * bw * math.sqrt(2.0 * math.pi)
    )
    return grid, dens


def _as_discrete_x(table, xcol: str):
    """Cast count-key column to ordered categories so the x scale is discrete.

    ggplot2 users typically map ``factor(x)`` for bar charts; without that,
    integer/numeric codes sit on a continuous axis (ticks at 5, 7, …). Count
    bars are categories of *values*, so we draw them discretely while keeping
    the original labels. Relative ``width`` still defaults to 0.9 (user-settable).

    Returns a small pandas frame (count tables are tiny; categorical levels
    preserve first-appearance order from ``count_by``).
    """
    from plot3.table import as_pandas

    out = as_pandas(table).copy()
    out[xcol] = out[xcol].map(lambda v: "NA" if pd.isna(v) else str(v))
    levels = list(dict.fromkeys(out[xcol].tolist()))
    out[xcol] = pd.Categorical(out[xcol], categories=levels, ordered=True)
    return out


def _hist_counts_left_closed(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Count with left-closed / right-open bins (ggplot2 ``closed = "left"``)."""
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    n = len(edges) - 1
    # searchsorted side='left': edge[i] <= x < edge[i+1] is not default;
    # use side='right' then subtract 1 for left-closed intervals.
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    # Drop values outside [edges[0], edges[-1]]
    inside = (values >= edges[0]) & (values <= edges[-1])
    counts = np.bincount(idx[inside], minlength=n).astype(np.float64)
    return counts


def _edges_from_binwidth(
    lo: float,
    hi: float,
    w: float,
    boundary: float | None,
) -> np.ndarray:
    """Build equal-width edges covering [lo, hi] with optional boundary align."""
    if w <= 0 or not math.isfinite(w):
        raise ValueError("binwidth must be a positive finite number")
    if boundary is not None:
        origin = float(boundary)
        start = origin + math.floor((lo - origin) / w) * w
    else:
        start = lo
    n = max(1, int(math.ceil((hi - start) / w)))
    edges = start + np.arange(n + 1, dtype=np.float64) * w
    if edges[-1] < hi:
        edges = np.append(edges, edges[-1] + w)
    return edges


def _auto_binwidth(values: np.ndarray, method: str = "fd") -> float:
    """Data-driven bin width (Freedman–Diaconis by default).

    Falls back along Scott → Sturges-derived width when a rule is undefined
    (e.g. zero IQR or zero variance).
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n <= 1:
        return 1.0
    lo = float(np.min(x))
    hi = float(np.max(x))
    span = hi - lo
    if span <= 0:
        return 1.0

    method = (method or "fd").lower()
    n_root = n ** (1.0 / 3.0)

    def fd_width() -> float | None:
        q75, q25 = np.percentile(x, [75.0, 25.0])
        iqr = float(q75 - q25)
        if iqr > 0:
            return 2.0 * iqr / n_root
        return None

    def scott_width() -> float | None:
        # Sample SD; Scott's normal-reference rule.
        sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
        if sd > 0:
            return 3.49 * sd / n_root
        return None

    def sturges_width() -> float:
        # Convert Sturges bin count into a width over the data span.
        k = max(1, int(math.ceil(math.log2(n) + 1.0)))
        return span / k

    order: list[str]
    if method in {"fd", "freedman-diaconis", "freedman_diaconis"}:
        order = ["fd", "scott", "sturges"]
    elif method == "scott":
        order = ["scott", "fd", "sturges"]
    elif method == "sturges":
        order = ["sturges"]
    elif method == "auto":
        # Prefer FD, then Scott, then Sturges (same cascade as a robust default).
        order = ["fd", "scott", "sturges"]
    else:
        # Unknown name: still try FD cascade rather than failing hard here;
        # numpy edges path handles named rules when bins is the method string.
        order = ["fd", "scott", "sturges"]

    for name in order:
        if name == "fd":
            w = fd_width()
        elif name == "scott":
            w = scott_width()
        else:
            w = sturges_width()
        if w is not None and w > 0 and math.isfinite(w):
            # At least one bin, at most a fine grid (avoid pathological widths).
            return float(min(max(w, span / 1000.0), span))
    return sturges_width()


def _histogram_breaks(
    values: np.ndarray,
    *,
    bins: int | None,
    binwidth: float | None,
    boundary: float | None,
    method: str = "fd",
) -> np.ndarray:
    """Compute histogram edges.

    Priority (user settings always win when given):

    1. ``binwidth`` — absolute width (optional ``boundary`` alignment)
    2. ``bins`` — explicit bin count (optional ``boundary``)
    3. automatic width from ``method`` (default Freedman–Diaconis), then edges
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        # Single unique value: one bin of unit width around the point.
        mid = lo
        w = 1.0 if binwidth is None else float(binwidth)
        return np.asarray([mid - 0.5 * w, mid + 0.5 * w], dtype=np.float64)

    if binwidth is not None:
        return _edges_from_binwidth(lo, hi, float(binwidth), boundary)

    if bins is not None:
        n_bins = max(1, int(bins))
        if boundary is None:
            return np.linspace(lo, hi, n_bins + 1)
        w = (hi - lo) / n_bins
        return _edges_from_binwidth(lo, hi, w, boundary)

    # Automatic: prefer numpy's named rule when it matches; otherwise FD cascade.
    method = (method or "fd").lower()
    numpy_names = {
        "fd",
        "scott",
        "sturges",
        "auto",
        "doane",
        "stone",
        "rice",
        "sqrt",
    }
    if method in numpy_names and boundary is None:
        try:
            edges = np.histogram_bin_edges(values, bins=method)
            edges = np.asarray(edges, dtype=np.float64)
            if edges.size >= 2 and np.all(np.isfinite(edges)):
                return edges
        except Exception:
            pass
    # FD/Scott/Sturges width cascade (also used when boundary is set).
    w = _auto_binwidth(values, method=method)
    return _edges_from_binwidth(lo, hi, w, boundary)


def expand_stat_geom(
    geom: _Geom,
    base_mapping: aes,
    data,
    domains: dict | None = None,
    transition=None,
    slider=None,
    coord=None,
    addons=None,
):
    """Turn statistical geoms into concrete drawable layers.

    Stats run against the native table backend (pandas / polars / tidy→polars).
    Only selected columns are pulled to arrays; small *result* frames used as
    ``data_override`` are plain pandas (already computed, render-ready).
    Identity geoms (point / line / …) are returned unchanged and materialise
    columns at the render boundary.

    A formula with an integral, tangent, or inequality returns a list of
    layers. A plain curve still returns one geom.
    """
    kind = getattr(geom, "kind", None)
    if kind == "function":
        from plot3.function import expand_function

        layers = expand_function(
            geom, base_mapping, data, domains, transition, slider, coord, addons
        )
        return layers[0] if len(layers) == 1 else layers
    if kind == "vector":
        from plot3.calculus import expand_vector_field

        layers = expand_vector_field(geom, transition, slider)
        return layers[0] if len(layers) == 1 else layers
    mapping = dict(base_mapping)
    mapping.update(geom.mapping)
    if geom.kind == "bar":
        if "x" not in mapping:
            raise ValueError("geom_bar() requires aes(x=)")
        xcol = mapping["x"]
        # Backend-native count: pandas groupby / polars group_by / tidy→polars.
        counts = count_by(data, xcol)
        # ggplot2 stat_count at unique x; draw on a discrete scale (factor(x)
        # style) so integer/numeric codes don't sit on a continuous axis with
        # phantom ticks. Labels keep the original values as strings.
        counts = _as_discrete_x(counts, xcol)
        out = geom_col(
            aes(x=xcol, y="y"),
            width=getattr(geom, "width", 0.9),
            color=geom.const_color,
            colour=None,
            alpha=geom.alpha,
        )
        out.data_override = counts
        out.const_color = geom.const_color
        out.alpha = geom.alpha
        return out
    if geom.kind == "histogram":
        if "x" not in mapping:
            raise ValueError("geom_histogram() requires aes(x=)")
        xcol = mapping["x"]
        values = numeric_array(data, xcol, dropna=True)
        binwidth = getattr(geom, "binwidth", None)
        bins = getattr(geom, "bins", None)
        method = getattr(geom, "method", "fd")
        boundary = getattr(geom, "boundary", None)
        closed = getattr(geom, "closed", "right")
        if values.size == 0:
            frame = pd.DataFrame(
                {"x": np.array([], dtype=float), "y": np.array([], dtype=float)}
            )
            abs_width = 1.0
            edge_lo, edge_hi = 0.0, 1.0
        else:
            edges = _histogram_breaks(
                values,
                bins=bins,
                binwidth=binwidth,
                boundary=boundary,
                method=method,
            )
            # numpy: closed right by default (right edge included except last);
            # closed left uses left-edge convention via density weights unused here.
            # numpy histogram is right-closed (except last bin); ggplot2
            # closed="right" matches that convention.
            if closed == "left":
                # Shift values slightly so membership matches left-closed bins
                # without changing edges: count with inverted edges via
                # searchsorted-based assignment.
                counts = _hist_counts_left_closed(values, edges)
            else:
                counts, _ = np.histogram(values, bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            abs_width = (
                float(np.median(np.diff(edges))) if len(edges) > 1 else 1.0
            )
            edge_lo, edge_hi = float(edges[0]), float(edges[-1])
            frame = pd.DataFrame(
                {"x": centers, "y": counts.astype(np.float64)}
            )
        # Absolute bin width → bars touch (ggplot2 geom_histogram / GeomBar).
        # Relative width stays 1.0 (full bin); users change bins/binwidth, not
        # a gap fraction, for histograms.
        out = geom_col(
            aes(x="x", y="y"),
            width=1.0,
            color=geom.const_color,
            alpha=geom.alpha,
        )
        out.data_override = frame
        out.const_color = geom.const_color
        out.alpha = geom.alpha
        out._bar_width_data = abs_width  # absolute data units (= binwidth)
        # Expand continuous domain to full bin edges (not just centres).
        out._x_domain = (edge_lo, edge_hi)
        return out
    if geom.kind == "boxplot":
        if "x" not in mapping or "y" not in mapping:
            raise ValueError("geom_boxplot() requires aes(x=, y=)")
        xcol, ycol = mapping["x"], mapping["y"]
        require_columns(data, [xcol, ycol])
        colour_col = mapping.get("color")
        group_cols = [xcol]
        if colour_col and colour_col != xcol and has_column(data, colour_col):
            group_cols.append(colour_col)
        coef = float(getattr(geom, "coef", 1.5))
        rows: list[dict] = []
        outlier_rows: list[dict] = []
        for key_tuple, piece in group_pieces(data, group_cols):
            values = numeric_array(piece, ycol, dropna=False)
            stats = _boxplot_stats(values, coef=coef)
            if stats is None:
                continue
            ymin, lower, middle, upper, ymax, outliers = stats
            row = {
                xcol: key_tuple[0],
                "ymin": ymin,
                "lower": lower,
                "middle": middle,
                "upper": upper,
                "ymax": ymax,
            }
            if len(group_cols) > 1:
                row[colour_col] = key_tuple[1]
            rows.append(row)
            for value in outliers:
                out_row = {xcol: key_tuple[0], ycol: float(value)}
                if len(group_cols) > 1:
                    out_row[colour_col] = key_tuple[1]
                outlier_rows.append(out_row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(
                columns=[
                    xcol,
                    "ymin",
                    "lower",
                    "middle",
                    "upper",
                    "ymax",
                    *([colour_col] if colour_col and colour_col != xcol else []),
                ]
            )
        # Concrete drawable layer with fixed stat column names.
        out = _Geom(
            aes(x=xcol, y="middle", colour=colour_col if colour_col else None),
            color=geom.const_color,
            alpha=geom.alpha,
        )
        out.kind = "box"
        out.width = float(getattr(geom, "width", 0.75))
        out.outlier_size = float(getattr(geom, "outlier_size", 3.0))
        out.data_override = frame
        out.const_color = geom.const_color
        out.alpha = geom.alpha
        out._stat_y_cols = ("ymin", "lower", "middle", "upper", "ymax")
        out._outlier_frame = pd.DataFrame(outlier_rows)
        out._y_name = ycol
        return out
    if geom.kind == "density":
        if "x" not in mapping:
            raise ValueError("geom_density() requires aes(x=)")
        xcol = mapping["x"]
        require_columns(data, [xcol])
        colour_col = mapping.get("color")
        n_grid = int(getattr(geom, "n", 512))
        adjust = float(getattr(geom, "adjust", 1.0))
        fill = bool(getattr(geom, "fill", True))
        pieces: list[pd.DataFrame] = []
        if colour_col and has_column(data, colour_col):
            for key_tuple, piece in group_pieces(data, [colour_col]):
                grid, dens = _kde_1d(
                    numeric_array(piece, xcol, dropna=False),
                    n=n_grid,
                    adjust=adjust,
                )
                if grid.size == 0:
                    continue
                frame = pd.DataFrame(
                    {"x": grid, "y": dens, colour_col: key_tuple[0]}
                )
                pieces.append(frame)
        else:
            grid, dens = _kde_1d(
                numeric_array(data, xcol, dropna=False),
                n=n_grid,
                adjust=adjust,
            )
            pieces.append(pd.DataFrame({"x": grid, "y": dens}))
        frame = (
            pd.concat(pieces, ignore_index=True)
            if pieces
            else pd.DataFrame(columns=["x", "y"])
        )
        map_kwargs = {"x": "x", "y": "y"}
        if colour_col and colour_col in frame.columns:
            map_kwargs["colour"] = colour_col
        out = _Geom(
            aes(**map_kwargs),
            color=geom.const_color,
            alpha=geom.alpha,
        )
        out.kind = "area" if fill else "line"
        out.sort_x = True
        out.linewidth = float(getattr(geom, "linewidth", 1.5))
        out.data_override = frame
        out.const_color = geom.const_color
        out.alpha = geom.alpha if geom.alpha is not None else (0.35 if fill else 0.95)
        out._baseline_zero = True
        return out
    if geom.kind == "violin":
        if "x" not in mapping or "y" not in mapping:
            raise ValueError("geom_violin() requires aes(x=, y=)")
        xcol, ycol = mapping["x"], mapping["y"]
        require_columns(data, [xcol, ycol])
        colour_col = mapping.get("color")
        n_grid = int(getattr(geom, "n", 128))
        adjust = float(getattr(geom, "adjust", 1.0))
        width = float(getattr(geom, "width", 0.9))
        levels = category_labels(data, xcol)
        level_index = {level: i for i, level in enumerate(levels)}
        rows: list[dict] = []
        grouping_cols = [xcol]
        if colour_col and colour_col != xcol and has_column(data, colour_col):
            grouping_cols.append(colour_col)
        for key_tuple, piece in group_pieces(data, grouping_cols):
            x_key = key_tuple[0]
            x_pos = float(level_index.get(str(x_key), 0))
            grid_y, dens = _kde_1d(
                numeric_array(piece, ycol, dropna=False),
                n=n_grid,
                adjust=adjust,
            )
            if grid_y.size == 0:
                continue
            peak = float(np.nanmax(dens)) or 1.0
            half = (dens / peak) * (width * 0.5)
            # Closed polygon: left side bottom→top, right side top→bottom.
            group_id = str(key_tuple)
            for yv, hw in zip(grid_y, half):
                row = {"x": x_pos - float(hw), "y": float(yv), "group": group_id}
                if colour_col and colour_col != xcol:
                    row[colour_col] = key_tuple[1]
                elif colour_col == xcol:
                    row[colour_col] = x_key
                rows.append(row)
            for yv, hw in zip(grid_y[::-1], half[::-1]):
                row = {"x": x_pos + float(hw), "y": float(yv), "group": group_id}
                if colour_col and colour_col != xcol:
                    row[colour_col] = key_tuple[1]
                elif colour_col == xcol:
                    row[colour_col] = x_key
                rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=["x", "y", "group"])
        # Force categorical x scale labels via a helper frame for domains:
        # encode x as numeric positions; stash category labels on the geom.
        map_kwargs = {"x": "x", "y": "y", "group": "group"}
        if colour_col and colour_col in frame.columns:
            map_kwargs["colour"] = colour_col
        out = _Geom(aes(**map_kwargs), color=geom.const_color, alpha=geom.alpha)
        out.kind = "poly"
        out.linewidth = float(getattr(geom, "linewidth", 1.0))
        out.data_override = frame
        out.const_color = geom.const_color
        out.alpha = geom.alpha if geom.alpha is not None else 0.45
        out._violin_levels = levels
        return out
    if geom.kind == "surface":
        if "x" not in mapping or "y" not in mapping or "z" not in mapping:
            raise ValueError("geom_surface() requires aes(x=, y=, z=)")
        xcol, ycol, zcol = mapping["x"], mapping["y"], mapping["z"]
        ccol = mapping.get("color")
        require_columns(data, [xcol, ycol, zcol])
        # Only mesh columns cross the table→pandas boundary.
        mesh_df = _surface_input_frame(data, xcol, ycol, zcol, ccol)
        vertices, indices, nx, ny = regular_grid_mesh(
            mesh_df, xcol, ycol, zcol, ccol=ccol
        )
        map_kwargs: dict = {"x": "x", "y": "y", "z": "z"}
        if ccol and "colour" in vertices.columns:
            map_kwargs["colour"] = "colour"
        out = _Geom(
            aes(**map_kwargs),
            color=geom.const_color,
            alpha=geom.alpha,
        )
        out.kind = "surface"
        out.data_override = vertices
        out.const_color = geom.const_color
        out.alpha = geom.alpha if geom.alpha is not None else 0.95
        out.wireframe = bool(getattr(geom, "wireframe", False))
        out._indices = indices
        out._nx = nx
        out._ny = ny
        return out
    if geom.kind == "isosurface":
        if "x" not in mapping or "y" not in mapping or "z" not in mapping:
            raise ValueError("geom_isosurface() requires aes(x=, y=, z=)")
        xcol, ycol, zcol = mapping["x"], mapping["y"], mapping["z"]
        require_columns(data, [xcol, ycol, zcol])
        # Column arrays only — no full-frame pandas conversion.
        xs = numeric_array(data, xcol, dropna=False)
        ys = numeric_array(data, ycol, dropna=False)
        zs = numeric_array(data, zcol, dropna=False)
        pts = np.column_stack([xs, ys, zs])
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        n_bins = int(getattr(geom, "n", 32))
        # Optional stat_density_3d on the figure is applied in build_spec.
        n_bins = int(getattr(geom, "_density_n", n_bins))
        levels = getattr(geom, "levels", (0.25, 0.5, 0.75))
        vertices, indices, used = isosurface_levels(
            pts, levels, n=n_bins, absolute=False
        )
        if vertices.empty:
            vertices = pd.DataFrame(
                columns=["x", "y", "z", "level", "colour"]
            )
            indices = np.zeros((0, 3), dtype=np.int32)
        colour_by = getattr(geom, "colour_by", "level")
        map_kwargs: dict = {"x": "x", "y": "y", "z": "z"}
        if colour_by == "level" and "colour" in vertices.columns:
            map_kwargs["colour"] = "colour"
        out = _Geom(
            aes(**map_kwargs),
            color=geom.const_color,
            alpha=geom.alpha,
        )
        out.kind = "isosurface"
        out.data_override = vertices
        out.const_color = geom.const_color
        out.alpha = geom.alpha if geom.alpha is not None else 0.55
        out.wireframe = bool(getattr(geom, "wireframe", False))
        out._indices = indices
        out._iso_levels = used
        return out
    return geom


def _surface_input_frame(data, xcol: str, ycol: str, zcol: str, ccol: str | None):
    """Build a small pandas frame for ``regular_grid_mesh`` (xyz [+ colour]).

    Drops rows with non-finite x/y/z only; colour may remain null.
    Only the selected mesh columns are converted — not the full source table.
    """
    from plot3.table import _eager_polars, _polars_to_pandas, detect_backend

    cols = [xcol, ycol, zcol]
    if ccol and has_column(data, ccol) and ccol not in cols:
        cols.append(ccol)
    backend = detect_backend(data)
    if backend == "pandas":
        work = data.loc[:, cols].copy()
        for c in (xcol, ycol, zcol):
            work[c] = pd.to_numeric(work[c], errors="coerce")
        return work.dropna(subset=[xcol, ycol, zcol])

    import polars as pl

    frame = _eager_polars(data).select(cols)
    for c in (xcol, ycol, zcol):
        frame = frame.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    frame = frame.drop_nulls(subset=[xcol, ycol, zcol])
    return _polars_to_pandas(frame)


# Geoms that cannot enter a 3D figure (stat expansions use these kinds too).
_2D_ONLY_KINDS = frozenset(
    {"col", "box", "area", "poly", "bar", "histogram", "boxplot", "density", "violin"}
)
_3D_POINT_KINDS = frozenset({"point", "line", "surface", "isosurface"})


def _default_3d_point_size(n: int, *, size_mode: str = "scene") -> float:
    """pcviz-like fine points on the unit-cube scene.

    pcviz uses ~0.06 m marks on clouds whose radius is tens of meters
    (size/radius ≈ 0.001). plot3 places points in a unit cube whose
    bounding-sphere radius is O(1), so the matching world size is ~0.001,
    scaled gently with density.
    """
    n = max(1, int(n))
    if size_mode == "screen":
        # Constant pixel size (Three.js sizeAttenuation=false).
        if n <= 5_000:
            return 2.0
        if n <= 50_000:
            return 1.5
        return 1.25
    # scene mode: world units after [0,1]×aspect encoding
    base = 0.0010
    s = base * (2_000.0 / n) ** (1.0 / 3.0)
    return float(round(min(0.0035, max(0.00035, s)), 5))


def _axis_label(g, base_map: dict, resolved, axis: str, is3d: bool) -> str:
    """Axis title: labs, then the ggplot mapping, then a function's variable."""
    if g.labs.get(axis):
        return g.labs[axis]
    mapped = base_map.get(axis)
    if mapped:
        return mapped
    for geom, _mapping in resolved:
        labels = getattr(geom, "_axis_labels", None)
        if labels and labels.get(axis):
            return str(labels[axis])
    if axis == "z" and not is3d:
        return ""
    return axis


def _coord_spec(coord, is3d: bool, resolved) -> dict | None:
    """Coordinate spec for the viewer.

    3D keeps ``coord_3d``. A formula surface with no coord uses equal aspect
    (a cube); data such as lidar stays proportional. 2D uses ``coord_equal``
    when asked, and also when every layer is an implicit equation, so a
    circle is round in a wide panel.
    """
    if is3d:
        if isinstance(coord, coord_equal):
            raise ValueError(
                "coord_equal() is for 2D figures; use coord_3d(aspect='equal')"
            )
        if isinstance(coord, coord_polar):
            raise ValueError(
                "coord_polar() is for 2D figures. "
                'For example ggplot() + geom_function("r = 1 + cos(theta)") '
                "+ coord_polar()"
            )
        if coord is not None:
            return coord.to_spec()
        # A formula's axes are different quantities (t vs x); true proportions
        # can squash the surface to a sliver. Data stays proportional (lidar).
        if any(getattr(geom, "_function_surface", False) for geom, _ in resolved):
            return {"aspect": "equal", "sizeMode": "scene", "maxPoints": None}
        return {"aspect": "data", "sizeMode": "scene", "maxPoints": None}
    if isinstance(coord, coord_3d):
        raise ValueError(
            "coord_3d() requires a 3D figure (map aes(z=...) on layers)"
        )
    if isinstance(coord, coord_polar):
        return coord.to_spec()
    if isinstance(coord, coord_equal):
        return coord.to_spec()
    if coord is not None:
        raise ValueError(
            "coord_3d() requires a 3D figure (map aes(z=...) on layers)"
        )
    implicit_only = bool(resolved) and all(
        getattr(geom, "_implicit", False) for geom, _mapped in resolved
    )
    if implicit_only:
        return {"aspect": "equal", "ratio": 1.0}
    return None


def _log10_values(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Map positive values to log10. Non-positive finites count as omitted."""
    v = np.asarray(values, dtype=np.float64)
    out = np.full(v.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(v) & (v > 0)
    out[ok] = np.log10(v[ok])
    n_bad = int(np.count_nonzero(np.isfinite(v) & (v <= 0)))
    return out, n_bad


def _last_finite(mat: np.ndarray) -> np.ndarray:
    """Last finite value of each row, or NaN when the row is all missing."""
    out = np.full(mat.shape[0], np.nan, dtype=np.float64)
    for frame in range(mat.shape[1] - 1, -1, -1):
        col = mat[:, frame]
        take = np.isnan(out) & np.isfinite(col)
        out[take] = col[take]
    return out


def _nice_at_most(x: float) -> float:
    if not math.isfinite(x) or x <= 0:
        return 0.0
    mag = 10 ** math.floor(math.log10(x))
    for mult in (5, 2, 1):
        val = mult * mag
        if val <= x * 1.0000001:
            return float(val)
    return float(mag)


def _size_breaks(vmax: float) -> list[float]:
    """A few legend sizes up to ``vmax``, on a 1-2-5 ladder."""
    if not math.isfinite(vmax) or vmax <= 0:
        return []
    breaks: list[float] = []
    for frac in (0.15, 0.4, 0.7, 1.0):
        nice = _nice_at_most(vmax * frac)
        if nice <= 0:
            continue
        if breaks and abs(nice - breaks[-1]) <= abs(breaks[-1]) * 1e-9:
            continue
        breaks.append(nice)
    if not breaks or breaks[-1] < vmax * 0.9:
        breaks.append(float(vmax))
    return breaks[-4:]


def _area_fraction(values: np.ndarray, vmax: float) -> np.ndarray:
    """sqrt(value / vmax), so bubble area tracks the column. Missing stays NaN."""
    v = np.asarray(values, dtype=np.float64)
    out = np.full(v.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(v) & (v >= 0)
    if vmax <= 0:
        out[ok] = 0.0
    else:
        out[ok] = np.sqrt(v[ok] / vmax)
    return out


def _bubble_max(is3d: bool, coord) -> float:
    """Largest bubble diameter: pixels in 2D / screen mode, scene units in 3D."""
    if not is3d:
        return 46.0
    mode = "scene"
    if coord is not None:
        mode = getattr(coord, "size_mode", "scene") or "scene"
    if mode == "screen":
        return 32.0
    return 0.06


def _filter_rows(vals: dict, ok: np.ndarray) -> None:
    """Keep rows where ``ok`` is true. Arrays and colour/group tuples follow."""
    n = int(ok.shape[0])
    for key, item in list(vals.items()):
        if isinstance(item, np.ndarray) and item.shape[:1] == (n,):
            vals[key] = item[ok]
        elif key == "color" and isinstance(item, tuple) and len(item) == 3:
            kind, cv, cats = item
            if isinstance(cv, np.ndarray) and cv.shape[:1] == (n,):
                vals[key] = (kind, cv[ok], cats)
        elif key == "group" and isinstance(item, tuple) and len(item) == 2:
            gv, gcats = item
            if isinstance(gv, np.ndarray) and gv.shape[:1] == (n,):
                vals[key] = (gv[ok], gcats)
        elif key == "ids" and isinstance(item, list) and len(item) == n:
            vals[key] = [item[i] for i, keep in enumerate(ok) if keep]


def _pivot_transition(sub: pd.DataFrame, mapping: dict, transition) -> dict:
    """One row per group, and a matrix per channel shaped (objects, frames).

    Matrices are still in data space. Object order is largest ``size`` first
    when that column is present, so small bubbles draw on top.
    """
    name = (
        "transition_states" if transition.kind == "states" else "transition_time"
    )
    if "group" not in mapping:
        raise ValueError(
            f"{name}() needs aes(group=) so each object keeps its identity "
            "across frames"
        )
    gcol = mapping["group"]
    tcol = transition.column
    if gcol not in sub.columns:
        raise KeyError(f"group column not in data: {gcol!r}")
    if tcol not in sub.columns:
        raise KeyError(f"{name}() column not in data: {tcol!r}")

    g_ok = sub[gcol].notna().to_numpy()
    labels = sub[gcol].astype(str)
    ids = sorted({labels.iloc[i] for i in range(len(sub)) if g_ok[i]})
    if not ids:
        raise ValueError(f"{name}() needs at least one group value")
    id_index = {lab: i for i, lab in enumerate(ids)}
    g_idx = labels.map(id_index).to_numpy(dtype=np.float64)
    g_idx = np.where(g_ok, g_idx, -1).astype(np.int32)

    if transition.kind == "states":
        t_ok = sub[tcol].notna().to_numpy()
        t_labels = sub[tcol].astype(str).tolist()
        times: list = []
        seen: dict[str, int] = {}
        for lab, ok in zip(t_labels, t_ok.tolist()):
            if not ok or lab in seen:
                continue
            seen[lab] = len(times)
            times.append(lab)
        if not times:
            raise ValueError(f"{name}() column has no values")
        t_idx = np.array(
            [seen.get(lab, -1) if ok else -1 for lab, ok in zip(t_labels, t_ok)],
            dtype=np.int32,
        )
        integer = False
        kind = "states"
    else:
        tvals = pd.to_numeric(sub[tcol], errors="coerce").to_numpy(dtype=np.float64)
        finite_t = np.isfinite(tvals)
        if not finite_t.any():
            raise ValueError(f"{name}() column has no finite values")
        times_arr = np.unique(tvals[finite_t])
        t_idx = np.full(len(sub), -1, dtype=np.int32)
        t_idx[finite_t] = np.searchsorted(times_arr, tvals[finite_t])
        scale = np.maximum(1.0, np.abs(times_arr))
        integer = bool(np.all(np.abs(times_arr - np.round(times_arr)) <= 1e-6 * scale))
        if integer:
            times = [int(round(float(t))) for t in times_arr]
        else:
            times = [float(t) for t in times_arr]
        kind = "time"

    n_fr = len(times)
    valid = (g_idx >= 0) & (t_idx >= 0)
    if valid.any():
        key = g_idx[valid].astype(np.int64) * (n_fr + 1) + t_idx[valid]
        if np.unique(key).size != key.size:
            raise ValueError(
                f"{name}() found more than one row for the same group and frame"
            )

    def _mat(values: np.ndarray) -> np.ndarray:
        out = np.full((len(ids), n_fr), np.nan, dtype=np.float64)
        if valid.any():
            out[g_idx[valid], t_idx[valid]] = np.asarray(values, dtype=np.float64)[valid]
        return out

    order = np.arange(len(ids))
    return {
        "ids": ids,
        "times": times,
        "integer": integer,
        "kind": kind,
        "column": transition.column,
        "nFrames": n_fr,
        "g_idx": g_idx,
        "t_idx": t_idx,
        "valid": valid,
        "order": order,
        "_mat": _mat,
    }


def _range_transition_meta(transition) -> dict:
    """Clock for a parameter sweep. Times follow the first range."""
    names = list(transition.ranges)
    first = names[0]
    lo, hi = transition.ranges[first]
    n_frames = int(transition.frames)
    times_arr = np.linspace(float(lo), float(hi), n_frames)
    scale = np.maximum(1.0, np.abs(times_arr))
    integer = bool(
        np.all(np.abs(times_arr - np.round(times_arr)) <= 1e-6 * scale)
    )
    if integer:
        times = [int(round(float(t))) for t in times_arr]
    else:
        times = [float(t) for t in times_arr]
    params = [
        {"name": name, "lo": float(bounds[0]), "hi": float(bounds[1])}
        for name, bounds in transition.ranges.items()
    ]
    return {
        "type": "time",
        "column": first,
        "nFrames": n_frames,
        "times": times,
        "integer": integer,
        "ease": "linear",
        "duration": 12,
        "params": params,
    }


def _slider_meta(slider) -> dict:
    """Independent ranges. The last keyword varies fastest (stride 1)."""
    names = list(slider.ranges)
    steps = int(slider.steps)
    params: list[dict] = []
    stride = 1
    for name in reversed(names):
        lo, hi = slider.ranges[name]
        params.append(
            {
                "name": name,
                "lo": float(lo),
                "hi": float(hi),
                "n": steps,
                "stride": stride,
            }
        )
        stride *= steps
    params.reverse()
    return {"nFrames": int(stride), "params": params}


def _legend_position_spec(value):
    """JSON form of theme(legend_position=). The static default is outside right."""
    if value is None:
        return "right"
    if isinstance(value, tuple):
        return [float(value[0]), float(value[1])]
    return value


def build_spec(g: ggplot) -> tuple[dict, list[tuple[str, str]]]:
    if not g.layers:
        raise ValueError("add a geom: ggplot(df, aes(...)) + geom_point()")
    # Function layers and vector fields sample their own grid, so a figure
    # may have no data frame.
    needs_data = any(
        getattr(layer, "kind", None) not in {"function", "vector"}
        for layer in g.layers
    )
    if g.data is None and needs_data:
        raise ValueError(
            "ggplot has no data; use ggplot(df, aes(...)) or "
            "pipe data with `data >> ggplot(aes(...))`"
        )

    data = g.data
    coord = getattr(g, "coord", None)
    mesh_kinds = {"surface", "isosurface"}
    has_mesh = any(
        getattr(layer, "kind", None) in mesh_kinds for layer in g.layers
    )
    if (
        data is not None
        and coord is not None
        and getattr(coord, "max_points", None)
        and not has_mesh
    ):
        nrows = n_rows(data)
        cap = int(coord.max_points)
        if nrows > cap:
            step = max(1, (nrows + cap - 1) // cap)
            data = subsample_rows(data, step)

    theme = THEMES[g.theme_name]
    # Apply optional stat_density_3d options onto isosurface layers.
    density_stat = getattr(g, "stat_density_3d", None)
    layers_in = []
    for geom in g.layers:
        if getattr(geom, "kind", None) == "isosurface" and density_stat is not None:
            geom = copy_geom_with_density_n(geom, density_stat.n)
        layers_in.append(geom)
    transition = getattr(g, "transition", None)
    slider = getattr(g, "slider", None)
    if slider is not None and transition is not None:
        raise ValueError(
            "slider() cannot be combined with transition_time() "
            "or transition_states()"
        )
    domains = None
    if any(getattr(layer, "kind", None) == "function" for layer in layers_in):
        from plot3.function import data_domains

        domains = data_domains(g, data)
    addon_map: dict[int, list] = {}
    for index, addon in getattr(g, "_addons", None) or []:
        addon_map.setdefault(int(index), []).append(addon)
    expanded = []
    for index, geom in enumerate(layers_in):
        result = expand_stat_geom(
            geom,
            g.mapping,
            data,
            domains,
            transition,
            slider,
            coord,
            addon_map.get(index),
        )
        if isinstance(result, list):
            expanded.extend(result)
        else:
            expanded.append(result)
    from plot3.calculus import mark_intersections

    expanded = mark_intersections(expanded)
    resolved = []  # per layer: (geom, mapping)
    for geom in expanded:
        # Function layers carry their own columns; don't inherit colour/group.
        if getattr(geom, "_replace_mapping", False):
            m = dict(geom.mapping)
        else:
            m = dict(g.mapping)
            m.update(geom.mapping)
        if "x" not in m or "y" not in m:
            raise ValueError("aes(x=, y=) are required (bar/histogram/density supply y)")
        if geom.kind == "surface" and "z" not in m:
            raise ValueError("geom_surface() requires aes(z=)")
        resolved.append((geom, m))

    is3d = any("z" in m for _, m in resolved)
    if is3d and not all("z" in m for _, m in resolved):
        raise ValueError("mix of 2D and 3D layers: every layer needs aes(z=)")
    if is3d and any(geom.kind in _2D_ONLY_KINDS for geom, _ in resolved):
        bad = sorted(
            {geom.kind for geom, _ in resolved if geom.kind in _2D_ONLY_KINDS}
        )
        raise ValueError(
            f"geom kind(s) {bad} are 2D-only; use geom_point / geom_point3d "
            "(or geom_line/path/surface) with aes(z=...) for 3D"
        )
    if is3d and getattr(g, "facet", None) is not None:
        raise ValueError("facet_wrap() is not supported with 3D figures yet")

    if transition is not None:
        tname = (
            "transition_states" if transition.kind == "states" else "transition_time"
        )
        # geom_function has already become a line, path, or surface. A
        # parameter sweep is recognised by the formula stamp, not by kind.
        if getattr(transition, "ranges", None):
            formula_layers = [
                geom
                for geom, _mapped in resolved
                if getattr(geom, "_is_formula", False)
            ]
            if not formula_layers:
                raise ValueError(
                    "transition_time() parameter ranges need a geom_function layer"
                )
            extra = sorted(
                {
                    geom.kind
                    for geom, _mapped in resolved
                    if not getattr(geom, "_is_formula", False)
                }
            )
            if extra:
                raise ValueError(
                    "transition_time() parameter ranges animate geom_function "
                    f"layers only (this figure also has {extra})"
                )
        else:
            kinds = {geom.kind for geom, _ in resolved}
            if "point" not in kinds:
                raise ValueError(f"{tname}() needs a geom_point layer")
            extra = sorted(kinds - {"point"})
            if extra:
                raise ValueError(
                    f"{tname}() animates geom_point layers only "
                    f"(this figure also has {extra})"
                )

    if slider is not None and getattr(slider, "ranges", None):
        formula_layers = [
            geom
            for geom, _mapped in resolved
            if getattr(geom, "_is_formula", False)
        ]
        if not formula_layers:
            raise ValueError("slider() needs a geom_function layer")
        extra = sorted(
            {
                geom.kind
                for geom, _mapped in resolved
                if not getattr(geom, "_is_formula", False)
            }
        )
        if extra:
            raise ValueError(
                "slider() drives geom_function layers only "
                f"(this figure also has {extra})"
            )
        if not any(getattr(geom, "_anim", None) for geom, _mapped in resolved):
            raise ValueError(
                "slider() parameters are not coefficients of a geom_function. "
                'Leave them unbound: geom_function("y = a x^2") + slider(a=(0, 3))'
            )

    axes = ["x", "y", "z"] if is3d else ["x", "y"]
    scales: dict[str, Scale] = {}
    color_scale = None  # ("num", lo, hi) | ("cat", cats)
    num_color_vals: list[np.ndarray] = []
    dropped_log = 0
    size_label = str(g.labs.get("size") or "")
    transition_meta: dict | None = None
    if transition is not None and getattr(transition, "ranges", None):
        transition_meta = _range_transition_meta(transition)
    slider_meta = (
        _slider_meta(slider)
        if slider is not None and getattr(slider, "ranges", None)
        else None
    )

    def _axis_trans(axis: str) -> str | None:
        if axis == "x" and getattr(g, "scale_x", None) is not None:
            return "log10"
        if axis == "y" and getattr(g, "scale_y", None) is not None:
            return "log10"
        return None

    def _absorb_position(axis: str, kind: str, v: np.ndarray, cats: list[str]):
        nonlocal dropped_log
        trans = _axis_trans(axis)
        sc = scales.get(axis)
        if trans == "log10" and kind != "num":
            raise ValueError(
                f"scale_{axis}_log10() needs a numeric {axis} column"
            )
        if sc is None:
            sc = scales[axis] = Scale(kind, trans=trans if kind == "num" else None)
        elif sc.kind != kind:
            raise ValueError(
                f"aes {axis}: layers disagree on scale type "
                f"({sc.kind} vs {kind})"
            )
        if kind == "cat":
            merged = list(dict.fromkeys(sc.cats + cats))
            remap = {
                cats.index(c) if c in cats else None: i
                for i, c in enumerate(merged)
                if c in cats
            }
            out = np.full(np.size(v), np.nan, dtype=np.float64)
            flat = np.asarray(v, dtype=np.float64).ravel()
            for i, c in enumerate(flat):
                if not math.isfinite(c):
                    continue
                out[i] = remap.get(int(c), -1)
            v = out.reshape(np.shape(v))
            sc.cats = merged
        else:
            if trans == "log10":
                v, n_bad = _log10_values(v)
                dropped_log += n_bad
            sc.widen(v)
        return v

    def _take_color(column_values_kind, cv, ccats, *, into):
        """Record a colour channel on ``into`` and widen the shared colour scale."""
        nonlocal color_scale
        kind = column_values_kind
        if kind == "cat" or (kind == "num" and ccats):
            if len(ccats) > len(theme["cat"]):
                raise ValueError(
                    f"{len(ccats)} colour categories > {len(theme['cat'])} "
                    "palette slots — fold rare categories or map a number"
                )
            if color_scale is None:
                color_scale = ["cat", list(ccats)]
            else:
                if color_scale[0] != "cat":
                    raise ValueError("layers disagree on colour scale type")
                color_scale[1] = list(dict.fromkeys(color_scale[1] + ccats))
            into["color"] = ("cat", cv, ccats)
        else:
            finite = np.asarray(cv, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if color_scale is None:
                color_scale = ["num", math.inf, -math.inf]
            elif color_scale[0] != "num":
                raise ValueError("layers disagree on colour scale type")
            if finite.size:
                color_scale[1] = min(color_scale[1], float(finite.min()))
                color_scale[2] = max(color_scale[2], float(finite.max()))
            num_color_vals.append(np.asarray(cv, dtype=np.float64))
            into["color"] = ("num", cv, None)

    def _drop_log_rows(vals: dict) -> None:
        log_on = any(
            getattr(scales.get(a), "trans", None) == "log10" for a in axes
        )
        if not log_on or "frames" in vals:
            return
        n = len(vals["x"])
        ok = np.isfinite(np.asarray(vals["x"], dtype=np.float64))
        if "y" in vals:
            ok = ok & np.isfinite(np.asarray(vals["y"], dtype=np.float64))
        if "z" in vals:
            ok = ok & np.isfinite(np.asarray(vals["z"], dtype=np.float64))
        if ok.all():
            return
        if not ok.any():
            raise ValueError(
                "log scale removed every row; values must be positive"
            )
        _filter_rows(vals, ok)

    def _vals_from_step(anim: dict) -> dict:
        parts_x = []
        parts_y = []
        spans = []
        groups = []
        cursor = 0
        for fr in anim["frames"]:
            xs = np.asarray(fr["x"], dtype=np.float64).ravel()
            ys = np.asarray(fr["y"], dtype=np.float64).ravel()
            parts_x.append(xs)
            parts_y.append(ys)
            spans.append([cursor, int(xs.size)])
            groups.append([[int(a), int(b)] for a, b in fr["groups"]])
            cursor += int(xs.size)
        all_x = np.concatenate(parts_x) if cursor else np.zeros(0)
        all_y = np.concatenate(parts_y) if cursor else np.zeros(0)
        tx = np.asarray(_absorb_position("x", "num", all_x, []), dtype=np.float64)
        ty = np.asarray(_absorb_position("y", "num", all_y, []), dtype=np.float64)
        static_i = int(anim.get("static", len(spans) - 1))
        start, count = spans[static_i]
        return {
            "x": tx[start:start + count].copy(),
            "y": ty[start:start + count].copy(),
            "frames": {
                "mode": "step",
                "nFrames": int(len(spans)),
                "spans": spans,
                "groups": groups,
                "x": tx,
                "y": ty,
            },
        }

    def _vals_from_function_anim(anim: dict) -> dict:
        if anim.get("mode") == "step":
            return _vals_from_step(anim)
        channels = anim["channels"]
        sample = next(iter(channels.values()))
        _n, n_frames = sample.shape
        frames = {
            "mode": "tween",
            "nFrames": int(n_frames),
            "nObj": int(sample.shape[0]),
        }
        vals = {"frames": frames}
        # Sliders show every parameter at its low end. Transitions keep
        # the last frame, which is what a paused chart already showed.
        static_col = int(anim.get("static_col", -1))
        for axis_name, mat in channels.items():
            mat = np.asarray(mat, dtype=np.float64)
            flat = _absorb_position(axis_name, "num", mat.ravel(), [])
            stored = np.asarray(flat, dtype=np.float64).reshape(mat.shape)
            frames[axis_name] = stored
            vals[axis_name] = stored[:, static_col].copy()
        return vals

    # Pass 1 — per-layer values + global scale domains.
    # Only selected columns are materialised to pandas at this boundary.
    layer_vals = []
    for geom, m in resolved:
        anim = getattr(geom, "_anim", None)
        if anim is not None:
            layer_vals.append(_vals_from_function_anim(anim))
            continue
        frame = getattr(geom, "data_override", None)
        if frame is None:
            frame = data
        if geom.kind == "box":
            # Stats frame: x + ymin/lower/middle/upper/ymax (+ optional colour).
            xcol = m["x"]
            y_stat_cols = getattr(
                geom, "_stat_y_cols", ("ymin", "lower", "middle", "upper", "ymax")
            )
            cols = [xcol, *y_stat_cols]
            frame_cols = get_columns(frame)
            if "color" in m and m["color"] in frame_cols:
                cols.append(m["color"])
            sub = materialize_columns(frame, list(dict.fromkeys(cols)))
            # dropna already applied; for box require all stat y cols present
            sub = sub.dropna(subset=[xcol, *y_stat_cols])
            vals = {}
            kind, v, cats = col_values(sub[xcol])
            vals["x"] = _absorb_position("x", kind, v, cats)
            for col_name in y_stat_cols:
                kind_y, v_y, cats_y = col_values(sub[col_name])
                if kind_y != "num":
                    raise ValueError("geom_boxplot() y statistics must be numeric")
                vals[col_name] = _absorb_position("y", kind_y, v_y, cats_y)
            # Use middle for a generic y channel (hover / fallback).
            vals["y"] = vals["middle"]
            if "color" in m and m["color"] in sub.columns:
                kind, cv, ccats = col_values(sub[m["color"]])
                if kind == "cat" or (kind == "num" and ccats):
                    if len(ccats) > len(theme["cat"]):
                        raise ValueError(
                            f"{len(ccats)} colour categories > {len(theme['cat'])} "
                            "palette slots — fold rare categories or map a number"
                        )
                    if color_scale is None:
                        color_scale = ["cat", list(ccats)]
                    else:
                        color_scale[1] = list(
                            dict.fromkeys(color_scale[1] + ccats)
                        )
                    vals["color"] = ("cat", cv, ccats)
                else:
                    if color_scale is None:
                        color_scale = ["num", math.inf, -math.inf]
                    color_scale[1] = min(color_scale[1], float(np.nanmin(cv)))
                    color_scale[2] = max(color_scale[2], float(np.nanmax(cv)))
                    num_color_vals.append(np.asarray(cv, dtype=np.float64))
                    vals["color"] = ("num", cv, None)
            # Outliers share the same scales.
            outliers = getattr(geom, "_outlier_frame", None)
            y_name = getattr(geom, "_y_name", "y")
            if outliers is not None and len(outliers):
                ox_kind, ox, ox_cats = col_values(outliers[xcol])
                oy_kind, oy, oy_cats = col_values(outliers[y_name])
                vals["ox"] = _absorb_position("x", ox_kind, ox, ox_cats)
                vals["oy"] = _absorb_position("y", oy_kind, oy, oy_cats)
                if "color" in m and m["color"] in outliers.columns:
                    okind, ocv, occats = col_values(outliers[m["color"]])
                    if okind == "cat" or (okind == "num" and occats):
                        if color_scale is None:
                            color_scale = ["cat", list(occats)]
                        else:
                            color_scale[1] = list(
                                dict.fromkeys(color_scale[1] + occats)
                            )
                        vals["ocolor"] = ("cat", ocv, occats)
                    else:
                        if color_scale is None:
                            color_scale = ["num", math.inf, -math.inf]
                        color_scale[1] = min(
                            color_scale[1], float(np.nanmin(ocv))
                        )
                        color_scale[2] = max(
                            color_scale[2], float(np.nanmax(ocv))
                        )
                        num_color_vals.append(np.asarray(ocv, dtype=np.float64))
                        vals["ocolor"] = ("num", ocv, None)
            layer_vals.append(vals)
            continue

        cols = [m[a] for a in axes if a in m] + (
            [m["color"]] if "color" in m else []
        ) + ([m["group"]] if "group" in m else [])
        if geom.kind == "point" and "size" in m:
            cols.append(m["size"])
        if transition is not None and getattr(transition, "column", None):
            cols.append(transition.column)
        sub = materialize_columns(frame, list(dict.fromkeys(cols)))

        if (
            transition is not None
            and geom.kind == "point"
            and not getattr(transition, "ranges", None)
        ):
            pivot = _pivot_transition(sub, m, transition)
            channels: dict[str, np.ndarray] = {}
            if "size" in m:
                skind, sv, _ = col_values(sub[m["size"]])
                if skind != "num":
                    raise ValueError("aes(size=) needs a numeric column")
                sv = np.asarray(sv, dtype=np.float64)
                sv = np.where(np.isfinite(sv) & (sv >= 0), sv, np.nan)
                channels["size"] = pivot["_mat"](sv)
                if not size_label:
                    size_label = str(m["size"])
            # Largest bubbles first, so later (smaller) points stay visible.
            if "size" in channels:
                with np.errstate(all="ignore"):
                    score = np.nanmax(channels["size"], axis=1)
                score = np.where(np.isfinite(score), score, -np.inf)
                pivot["order"] = np.argsort(-score, kind="mergesort")
            order = pivot["order"]
            ids = [pivot["ids"][i] for i in order]
            frames = {
                "ids": ids,
                "times": pivot["times"],
                "integer": bool(pivot["integer"]),
                "kind": pivot["kind"],
                "column": pivot["column"],
                "nFrames": int(pivot["nFrames"]),
                "nObj": len(ids),
            }
            vals = {"ids": ids, "frames": frames}
            for a in axes:
                kind, raw, cats = col_values(sub[m[a]])
                mat = pivot["_mat"](raw)[order]
                flat = _absorb_position(a, kind, mat.ravel(), cats)
                frames[a] = np.asarray(flat, dtype=np.float64).reshape(mat.shape)
                # Static fallback is the last keyframe (missing stays missing).
                vals[a] = frames[a][:, -1].copy()
            if "size" in channels:
                frames["size"] = channels["size"][order]
                vals["size"] = _last_finite(frames["size"])
            if "color" in m:
                kind, cv, ccats = col_values(sub[m["color"]])
                color_mat = pivot["_mat"](cv)[order]
                frames["color"] = color_mat
                frames["color_kind"] = "cat" if (kind == "cat" or ccats) else "num"
                frames["color_cats"] = list(ccats)
                shown = _last_finite(color_mat)
                _take_color(kind, shown, ccats, into=vals)
                if frames["color_kind"] == "num":
                    full = np.asarray(cv, dtype=np.float64)
                    num_color_vals[-1] = full
                    finite = full[np.isfinite(full)]
                    if finite.size and color_scale and color_scale[0] == "num":
                        color_scale[1] = min(color_scale[1], float(finite.min()))
                        color_scale[2] = max(color_scale[2], float(finite.max()))
            meta = {
                "type": frames["kind"],
                "column": frames["column"],
                "nFrames": frames["nFrames"],
                "times": list(frames["times"]),
                "integer": frames["integer"],
                "ease": "smooth" if frames["kind"] == "states" else "linear",
                "duration": 12,
            }
            if transition_meta is None:
                transition_meta = meta
            elif transition_meta["times"] != meta["times"] or transition_meta["type"] != meta["type"]:
                raise ValueError(
                    f"{transition_meta['type']} layers disagree on frame values"
                )
            layer_vals.append(vals)
            continue

        vals = {}
        for a in axes:
            kind, v, cats = col_values(sub[m[a]])
            vals[a] = _absorb_position(a, kind, v, cats)
        # Histogram: domain is full bin edges, not just bin centres.
        x_domain = getattr(geom, "_x_domain", None)
        if x_domain is not None and "x" in scales and scales["x"].kind == "num":
            dom = np.asarray([x_domain[0], x_domain[1]], dtype=np.float64)
            if getattr(scales["x"], "trans", None) == "log10":
                dom, n_bad = _log10_values(dom)
                dropped_log += n_bad
            scales["x"].widen(dom)
        # Bars / densities include the baseline at y=0 in the domain.
        # A log axis has no zero; positive bars keep the data domain.
        if (
            geom.kind in {"col", "area"}
            or getattr(geom, "_baseline_zero", False)
        ) and "y" in scales and scales["y"].kind == "num":
            if getattr(scales["y"], "trans", None) != "log10":
                scales["y"].widen(np.asarray([0.0], dtype=np.float64))
        # Violin: numeric x positions with categorical tick labels.
        if getattr(geom, "_violin_levels", None) is not None:
            if _axis_trans("x") == "log10":
                raise ValueError(
                    "scale_x_log10() cannot be used with categorical x"
                )
            levels = list(geom._violin_levels)
            scales["x"] = Scale("cat")
            scales["x"].cats = levels
        if "color" in m:
            kind, cv, ccats = col_values(sub[m["color"]])
            if kind == "cat" or (kind == "num" and ccats):
                if len(ccats) > len(theme["cat"]):
                    raise ValueError(
                        f"{len(ccats)} colour categories > {len(theme['cat'])} "
                        "palette slots — fold rare categories or map a number"
                    )
                if color_scale is None:
                    color_scale = ["cat", list(ccats)]
                else:
                    color_scale[1] = list(dict.fromkeys(color_scale[1] + ccats))
                vals["color"] = ("cat", cv, ccats)
            else:
                if color_scale is None:
                    color_scale = ["num", math.inf, -math.inf]
                color_scale[1] = min(color_scale[1], float(np.nanmin(cv)))
                color_scale[2] = max(color_scale[2], float(np.nanmax(cv)))
                num_color_vals.append(np.asarray(cv, dtype=np.float64))
                vals["color"] = ("num", cv, None)
        if "group" in m:
            _, gv, gcats = col_values(sub[m["group"]])
            vals["group"] = (gv, gcats)
        if geom.kind == "point" and "size" in m:
            skind, sv, _ = col_values(sub[m["size"]])
            if skind != "num":
                raise ValueError("aes(size=) needs a numeric column")
            sv = np.asarray(sv, dtype=np.float64)
            vals["size"] = np.where(np.isfinite(sv) & (sv >= 0), sv, np.nan)
            if not size_label:
                size_label = str(m["size"])
        _drop_log_rows(vals)
        layer_vals.append(vals)

    for a in axes:
        scales[a].finish()

    # Optional forced domains (facet_wrap scales="fixed").
    force = getattr(g, "_force_scales", None) or {}
    for ax, (lo, hi) in force.items():
        if ax in scales and scales[ax].kind == "num":
            lo_f, hi_f = float(lo), float(hi)
            if getattr(scales[ax], "trans", None) == "log10":
                if lo_f <= 0 or hi_f <= 0:
                    raise ValueError(f"scale_{ax}_log10() limits must be positive")
                lo_f, hi_f = math.log10(lo_f), math.log10(hi_f)
            scales[ax].lo = lo_f
            scales[ax].hi = hi_f
            if scales[ax].hi <= scales[ax].lo:
                scales[ax].hi = scales[ax].lo + 1.0

    # A function's ylim/zlim clips the view even when samples sit inside it.
    for geom, _locked_mapping in resolved:
        lock = getattr(geom, "_axis_lock", None)
        if not lock:
            continue
        for axis_name, bounds in lock.items():
            if axis_name not in scales:
                continue
            lo, hi = float(bounds[0]), float(bounds[1])
            if hi < lo:
                lo, hi = hi, lo
            if getattr(scales[axis_name], "trans", None) == "log10":
                if lo <= 0 or hi <= 0:
                    raise ValueError(
                        f"scale_{axis_name}_log10() limits must be positive"
                    )
                lo, hi = math.log10(lo), math.log10(hi)
            if hi <= lo:
                hi = lo + 1.0
            scales[axis_name].lo = lo
            scales[axis_name].hi = hi

    # Numeric colour limits: robust 2-98 percentile by default so skewed data
    # (lidar intensity) actually varies; override via scale_colour_continuous.
    num_color = None
    if color_scale is not None and color_scale[0] == "num":
        allv = np.concatenate(num_color_vals) if num_color_vals else np.array([0.0, 1.0])
        cs = g.cscale or scale_colour_continuous()
        if "color" in force:
            lo_c, hi_c = force["color"]
            lo_c, hi_c = float(lo_c), float(hi_c)
        elif cs.limits == "full":
            lo_c, hi_c = color_scale[1], color_scale[2]
        elif isinstance(cs.limits, (tuple, list)):
            lo_c, hi_c = float(cs.limits[0]), float(cs.limits[1])
        else:
            lo_c = float(np.nanpercentile(allv, 2))
            hi_c = float(np.nanpercentile(allv, 98))
        if hi_c <= lo_c:
            lo_c, hi_c = color_scale[1], color_scale[2]
        if hi_c <= lo_c:
            hi_c = lo_c + 1.0
        tf = {
            "linear": lambda a: a,
            "sqrt": lambda a: np.sqrt(np.maximum(a, 0.0)),
            "log10": lambda a: np.log10(np.maximum(a, 1e-12)),
        }[cs.trans]
        num_color = (lo_c, hi_c, cs.trans, tf)

    # One size scale for the figure. Area: radius follows sqrt(value / max).
    size_pieces = []
    for vals in layer_vals:
        if vals.get("size") is not None:
            size_pieces.append(np.asarray(vals["size"], dtype=np.float64).ravel())
        fr = vals.get("frames")
        if fr is not None and fr.get("size") is not None:
            size_pieces.append(np.asarray(fr["size"], dtype=np.float64).ravel())
    size_max = None
    size_units = None
    if size_pieces:
        all_s = np.concatenate(size_pieces)
        ok_s = all_s[np.isfinite(all_s) & (all_s >= 0)]
        if ok_s.size == 0:
            raise ValueError("aes(size=) has no finite, non-negative values")
        size_max = float(ok_s.max())
        size_units = _bubble_max(is3d, coord)
        for vals in layer_vals:
            if vals.get("size") is not None:
                vals["size_frac"] = _area_fraction(vals["size"], size_max)
            fr = vals.get("frames")
            if fr is not None and fr.get("size") is not None:
                fr["size_frac"] = _area_fraction(fr["size"], size_max)

    # Pass 2 — encode payloads per layer (quantized against the shared scales)
    # Distinct default colours so several formulas can share a legend.
    # A fill or a tangent point then copies the layer it belongs to.
    palette = theme["cat"]
    color_slot = 0
    for geom, _mapped in resolved:
        if getattr(geom, "_legend_label", None) and geom.const_color is None:
            geom.const_color = palette[color_slot % len(palette)]
            color_slot += 1
    labeled: dict[str, str] = {}
    for geom, _mapped in resolved:
        if geom.const_color is None:
            continue
        key = getattr(geom, "_color_key", None)
        label = getattr(geom, "_legend_label", None)
        if key and key not in labeled:
            labeled[key] = geom.const_color
        if label and label not in labeled:
            labeled[label] = geom.const_color
    for geom, _mapped in resolved:
        if geom.const_color is not None:
            continue
        source = getattr(geom, "_inherit_from", None)
        if source and source in labeled:
            geom.const_color = labeled[source]

    payloads: list[tuple[str, str]] = []
    layer_specs = []
    for li, ((geom, m), vals) in enumerate(zip(resolved, layer_vals)):
        n = len(vals["x"])
        order = np.arange(n)
        group_vec = None
        if "group" in vals:
            group_vec = vals["group"][0]
        elif vals.get("color") and vals["color"][0] == "cat":
            group_vec = vals["color"][1]
        # Precomputed breaks (implicit contours, clipped formulas) stay in order.
        if getattr(geom, "_groups", None) is not None:
            order = np.arange(n)
        elif geom.kind in {"line", "area"}:
            keys = []
            if group_vec is not None:
                keys.append(group_vec)
            if geom.kind == "area" or getattr(geom, "sort_x", False):
                keys.append(vals["x"])
            if keys:
                order = np.lexsort(tuple(reversed(keys)))
        # Large bubbles first so smaller ones, drawn later, stay visible.
        if (
            geom.kind == "point"
            and vals.get("size") is not None
            and "frames" not in vals
        ):
            raw_size = np.asarray(vals["size"], dtype=np.float64)
            score = np.where(
                np.isfinite(raw_size) & (raw_size >= 0), raw_size, -np.inf
            )
            order = np.argsort(-score, kind="mergesort")
        # poly keeps authoring order (closed violin contours).

        spec_l = {
            "kind": geom.kind,
            "n": int(n),
            "alpha": geom.alpha,
            "constColor": geom.const_color,
        }

        def _encode_channel(name: str, values: np.ndarray, axis: str):
            sc = scales[axis]
            values = np.asarray(values, dtype=np.float64)
            if not np.isfinite(values).all():
                fill = sc.lo if math.isfinite(sc.lo) else 0.0
                values = np.where(np.isfinite(values), values, fill)
            enc = encode_norm(
                values,
                sc.lo,
                sc.hi,
                quantize=g.quantize,
                compress=g.compress,
            )
            pid = f"p{li}{name}"
            payloads.append((pid, enc["b64"]))
            spec_l[name] = {"id": pid, "dtype": enc["dtype"]}

        def _encode_matrix(tag: str, values: np.ndarray, lo: float, hi: float):
            flat = np.asarray(values, dtype=np.float64).ravel()
            present = np.isfinite(flat)
            fill = lo if math.isfinite(lo) else 0.0
            filled = np.where(present, flat, fill)
            enc = encode_norm(
                filled, lo, hi, quantize=g.quantize, compress=g.compress
            )
            pid = f"p{li}f{tag}"
            mid = f"p{li}f{tag}m"
            payloads.append((pid, enc["b64"]))
            payloads.append((mid, pack_u8(present.astype(np.uint8), g.compress)))
            return {"id": pid, "dtype": enc["dtype"], "mask": mid}

        if geom.kind in {"surface", "isosurface"}:
            for a in axes:
                _encode_channel(a, vals[a][order], a)
            indices = getattr(geom, "_indices", None)
            if indices is None:
                raise ValueError(f"{geom.kind} missing triangle indices")
            flat = np.ascontiguousarray(indices.reshape(-1), dtype=np.uint32)
            pid = f"p{li}idx"
            payloads.append((pid, pack_u32(flat, g.compress)))
            spec_l["indices"] = {
                "id": pid,
                "dtype": "u32",
                "count": int(flat.size),
            }
            spec_l["wireframe"] = bool(getattr(geom, "wireframe", False))
            if getattr(geom, "_nx", None) is not None:
                spec_l["nx"] = int(geom._nx)
                spec_l["ny"] = int(geom._ny)
            if spec_l["alpha"] is None:
                spec_l["alpha"] = 0.95 if geom.kind == "surface" else 0.55
        elif geom.kind == "box":
            for name in ("x", "ymin", "lower", "middle", "upper", "ymax"):
                _encode_channel(name, vals[name][order], "x" if name == "x" else "y")
            # Keep y as middle for shared hover helpers.
            spec_l["y"] = spec_l["middle"]
            if vals.get("ox") is not None:
                n_out = len(vals["ox"])
                spec_l["nOut"] = int(n_out)
                _encode_channel("ox", vals["ox"], "x")
                _encode_channel("oy", vals["oy"], "y")
                if vals.get("ocolor"):
                    ckind, cv, _ = vals["ocolor"]
                    pid = f"p{li}oc"
                    if ckind == "cat":
                        local = vals["ocolor"][2]
                        remap = {
                            i: color_scale[1].index(c)
                            for i, c in enumerate(local)
                        }
                        codes = np.array(
                            [remap.get(int(c), 0) for c in cv], dtype="<u2"
                        )
                        payloads.append((pid, pack_u16(codes, g.compress)))
                        spec_l["ocolor"] = {
                            "id": pid, "dtype": "u16", "kind": "cat"
                        }
                    else:
                        lo_c, hi_c, _trans, tf = num_color
                        cvt = tf(np.clip(cv, lo_c, hi_c))
                        lo_t = float(tf(np.asarray(lo_c)))
                        hi_t = float(tf(np.asarray(hi_c)))
                        enc = encode_norm(
                            cvt, lo_t, hi_t, quantize=True, compress=g.compress
                        )
                        payloads.append((pid, enc["b64"]))
                        spec_l["ocolor"] = {
                            "id": pid, "dtype": "u16", "kind": "num"
                        }
            else:
                spec_l["nOut"] = 0
        elif geom.kind not in {"surface", "isosurface"}:
            for a in axes:
                _encode_channel(a, vals[a][order], a)

        if vals.get("color") and geom.kind not in {"box"}:
            ckind, cv, _ = vals["color"]
            pid = f"p{li}c"
            if ckind == "cat":
                # remap onto the global cat list
                local = vals["color"][2]
                remap = {i: color_scale[1].index(c) for i, c in enumerate(local)}
                codes = np.array([remap.get(int(c), 0) for c in cv[order]],
                                 dtype="<u2")
                payloads.append((pid, pack_u16(codes, g.compress)))
                spec_l["color"] = {"id": pid, "dtype": "u16", "kind": "cat"}
            else:
                lo_c, hi_c, _trans, tf = num_color
                cvt = tf(np.clip(cv[order], lo_c, hi_c))
                lo_t = float(tf(np.asarray(lo_c)))
                hi_t = float(tf(np.asarray(hi_c)))
                enc = encode_norm(cvt, lo_t, hi_t, quantize=True,
                                   compress=g.compress)
                payloads.append((pid, enc["b64"]))
                spec_l["color"] = {"id": pid, "dtype": "u16", "kind": "num"}
        elif vals.get("color") and geom.kind == "box":
            ckind, cv, _ = vals["color"]
            pid = f"p{li}c"
            if ckind == "cat":
                local = vals["color"][2]
                remap = {i: color_scale[1].index(c) for i, c in enumerate(local)}
                codes = np.array(
                    [remap.get(int(c), 0) for c in cv[order]], dtype="<u2"
                )
                payloads.append((pid, pack_u16(codes, g.compress)))
                spec_l["color"] = {"id": pid, "dtype": "u16", "kind": "cat"}
            else:
                lo_c, hi_c, _trans, tf = num_color
                cvt = tf(np.clip(cv[order], lo_c, hi_c))
                lo_t = float(tf(np.asarray(lo_c)))
                hi_t = float(tf(np.asarray(hi_c)))
                enc = encode_norm(
                    cvt, lo_t, hi_t, quantize=True, compress=g.compress
                )
                payloads.append((pid, enc["b64"]))
                spec_l["color"] = {"id": pid, "dtype": "u16", "kind": "num"}

        if geom.kind in {"line", "area", "poly"}:
            preset_groups = getattr(geom, "_groups", None)
            if preset_groups is not None:
                spec_l["groups"] = [
                    [int(start), int(count)] for start, count in preset_groups
                ]
                spec_l["linewidth"] = float(getattr(geom, "linewidth", 2.0))
                if geom.kind in {"area", "poly"} and spec_l["alpha"] is None:
                    spec_l["alpha"] = 0.4 if geom.kind == "area" else 0.45
            elif group_vec is not None:
                gv = group_vec[order]
                cut = np.flatnonzero(np.diff(gv)) + 1
                starts = np.concatenate([[0], cut])
                counts = np.diff(np.concatenate([starts, [n]]))
                spec_l["groups"] = [
                    [int(s), int(c)] for s, c in zip(starts, counts)
                ]
            else:
                spec_l["groups"] = [[0, int(n)]]
            spec_l["linewidth"] = float(getattr(geom, "linewidth", 2.0))
            if geom.kind in {"area", "poly"} and spec_l["alpha"] is None:
                spec_l["alpha"] = 0.4 if geom.kind == "area" else 0.45
            if geom.kind == "area":
                scy = scales["y"]
                baseline = getattr(geom, "_baseline", None)
                if baseline is None:
                    baseline = 0.0
                if scy.kind == "num":
                    y_span = max(scy.hi - scy.lo, 1e-12)
                    spec_l["y0"] = float(
                        np.clip((float(baseline) - scy.lo) / y_span, 0.0, 1.0)
                    )
                else:
                    spec_l["y0"] = 0.0
        elif geom.kind in {"col", "box"}:
            # Bar/box width in normalized [0,1] x-space (ggplot2 resolution × width).
            scx = scales["x"]
            span = max(scx.hi - scx.lo, 1e-12)
            rel = float(
                getattr(
                    geom,
                    "width",
                    0.75 if geom.kind == "box" else 0.9,
                )
            )
            if getattr(geom, "_bar_width_data", None) is not None:
                # Absolute data width from stat (histogram binwidth).
                data_w = float(geom._bar_width_data) * rel
            elif scx.kind == "cat":
                # Discrete scale: unit spacing between categories (resolution = 1).
                data_w = 1.0 * rel
            else:
                # Continuous: data_width = resolution(x) * width (ggplot2).
                xs = np.asarray(vals["x"][order], dtype=np.float64)
                data_w = resolution(xs, zero=False) * rel
            spec_l["width"] = float(np.clip(data_w / span, 1e-4, 1.0))
            if geom.kind == "col":
                scy = scales["y"]
                if scy.kind == "num":
                    y_span = max(scy.hi - scy.lo, 1e-12)
                    spec_l["y0"] = float(
                        np.clip((0.0 - scy.lo) / y_span, 0.0, 1.0)
                    )
                else:
                    spec_l["y0"] = 0.0
            else:
                spec_l["outlierSize"] = float(
                    getattr(geom, "outlier_size", 3.0)
                )
            if spec_l["alpha"] is None:
                spec_l["alpha"] = 0.9
        else:
            if vals.get("size_frac") is not None:
                frac = np.asarray(vals["size_frac"], dtype=np.float64)[order]
                present = np.isfinite(frac)
                filled = np.where(present, frac, 0.0)
                enc = encode_norm(
                    filled, 0.0, 1.0, quantize=g.quantize, compress=g.compress
                )
                pid = f"p{li}sz"
                payloads.append((pid, enc["b64"]))
                spec_l["size"] = {
                    "id": pid,
                    "dtype": enc["dtype"],
                    "scale": "area",
                    "max": float(size_units),
                    "vmax": float(size_max),
                }
            elif getattr(geom, "size", None) is not None:
                spec_l["size"] = float(geom.size)
            elif is3d:
                mode = "scene"
                if coord is not None:
                    mode = getattr(coord, "size_mode", "scene") or "scene"
                spec_l["size"] = _default_3d_point_size(n, size_mode=mode)
            else:
                # pixels
                spec_l["size"] = 6.0 if n <= 2000 else (4.0 if n <= 20000 else 2.5)
            if spec_l["alpha"] is None:
                if is3d:
                    # Opaque marks read sharper for lidar-style clouds (pcviz).
                    spec_l["alpha"] = 1.0 if n <= 200_000 else 0.85
                else:
                    spec_l["alpha"] = 0.85 if n <= 50000 else 0.6
        if spec_l["alpha"] is None:
            spec_l["alpha"] = 1.0
        frames = vals.get("frames")
        if frames and frames.get("mode") == "step":
            spec_l["frames"] = {
                "mode": "step",
                "nFrames": int(frames["nFrames"]),
                "spans": frames["spans"],
                "groups": frames["groups"],
            }
            for axis_name in ("x", "y"):
                spec_l["frames"][axis_name] = _encode_matrix(
                    axis_name,
                    frames[axis_name],
                    scales[axis_name].lo,
                    scales[axis_name].hi,
                )
        elif frames and geom.kind == "point":
            spec_l["frames"] = {
                "nFrames": int(frames["nFrames"]),
                "nObj": int(frames["nObj"]),
            }
            for a in axes:
                spec_l["frames"][a] = _encode_matrix(
                    a, frames[a], scales[a].lo, scales[a].hi
                )
            if frames.get("size_frac") is not None:
                ch = _encode_matrix("sz", frames["size_frac"], 0.0, 1.0)
                ch["scale"] = "area"
                ch["max"] = float(size_units)
                ch["vmax"] = float(size_max)
                spec_l["frames"]["size"] = ch
            if frames.get("color") is not None and color_scale is not None:
                if frames["color_kind"] == "cat":
                    local = frames["color_cats"]
                    remap = {
                        i: color_scale[1].index(c) for i, c in enumerate(local)
                    }
                    raw = np.asarray(frames["color"], dtype=np.float64).ravel()
                    flat = np.full(raw.shape, 65535, dtype=np.uint16)
                    ok = np.isfinite(raw) & (raw >= 0)
                    if ok.any():
                        mapped = np.array(
                            [remap.get(int(c), 0) for c in raw[ok]],
                            dtype=np.uint16,
                        )
                        flat[np.flatnonzero(ok)] = mapped
                    enc = encode_codes(flat, g.compress)
                    pid = f"p{li}fc"
                    payloads.append((pid, enc["b64"]))
                    spec_l["frames"]["color"] = {
                        "id": pid, "dtype": "u16", "kind": "cat",
                    }
                else:
                    lo_c, hi_c, _trans, tf = num_color
                    raw = np.asarray(frames["color"], dtype=np.float64)
                    present = np.isfinite(raw)
                    transformed = np.full(raw.shape, np.nan, dtype=np.float64)
                    if present.any():
                        transformed[present] = tf(np.clip(raw[present], lo_c, hi_c))
                    lo_t = float(tf(np.asarray(lo_c)))
                    hi_t = float(tf(np.asarray(hi_c)))
                    if not math.isfinite(lo_t) or not math.isfinite(hi_t) or hi_t <= lo_t:
                        lo_t, hi_t = 0.0, 1.0
                    ch = _encode_matrix("c", transformed, lo_t, hi_t)
                    ch["kind"] = "num"
                    spec_l["frames"]["color"] = ch
        elif frames:
            spec_l["frames"] = {
                "nFrames": int(frames["nFrames"]),
                "nObj": int(frames["nObj"]),
            }
            for a in axes:
                if frames.get(a) is None:
                    continue
                spec_l["frames"][a] = _encode_matrix(
                    a, frames[a], scales[a].lo, scales[a].hi
                )
        if geom.kind == "point":
            labels = None
            if isinstance(vals.get("ids"), list) and len(vals["ids"]) == n:
                labels = list(vals["ids"])
            elif "group" in vals:
                gv, gcats = vals["group"]
                labels = []
                for c in np.asarray(gv, dtype=np.float64):
                    if not math.isfinite(c):
                        labels.append("")
                        continue
                    idx = int(c)
                    if gcats and 0 <= idx < len(gcats):
                        labels.append(str(gcats[idx]))
                    elif float(c).is_integer():
                        labels.append(str(idx))
                    else:
                        labels.append(str(c))
            if labels is not None and n <= 8000:
                spec_l["ids"] = [labels[i] for i in order]
        tip_pretty = getattr(geom, "_tip_pretty", None)
        if tip_pretty:
            tip = {"pretty": str(tip_pretty)}
            tip_latex = getattr(geom, "_tip_latex", None)
            if tip_latex:
                tip["latex"] = str(tip_latex)
            spec_l["tip"] = tip
        layer_specs.append(spec_l)

    # color spec + legend
    cspec = {"kind": "none"}
    legend = None
    if color_scale is not None:
        if color_scale[0] == "cat":
            cats = color_scale[1]
            cspec = {"kind": "cat", "palette": theme["cat"][: len(cats)],
                     "cats": cats}
            legend = [{"label": c, "color": theme["cat"][i]}
                      for i, c in enumerate(cats)]
        else:
            pal = (g.cscale.palette if g.cscale else "blue")
            ramp = _CONT_PALETTES.get(pal, theme["seq"])
            cspec = {"kind": "num", "lo": num_color[0], "hi": num_color[1],
                     "trans": num_color[2], "ramp": ramp}
    if legend is None:
        entries = []
        for geom, _mapped in resolved:
            label = getattr(geom, "_legend_label", None)
            if not label or not geom.const_color:
                continue
            entry = {"label": str(label), "color": geom.const_color}
            if getattr(geom, "_is_formula", False):
                entry["formula"] = True
                segments = getattr(geom, "_legend_math", None)
                latex = getattr(geom, "_legend_latex", None)
                if segments:
                    entry["math"] = segments
                elif latex:
                    entry["latex"] = str(latex)
            entries.append(entry)
        if entries:
            legend = entries

    base_map = dict(g.mapping)
    coord_spec = _coord_spec(coord, is3d, resolved)
    labs_math: dict[str, list] = {}

    def _take(key: str, text) -> str:
        raw = "" if text is None else str(text)
        if "$" not in raw:
            return raw
        plain, segments = split_math(raw)
        if segments:
            labs_math[key] = segments
        return plain

    formula_geoms = [
        geom for geom, _mapped in resolved if getattr(geom, "_formula_primary", False)
    ]
    raw_title = g.labs.get("title") or ""
    # One function and no title of your own: the formula is the title.
    # A shaded area, a tangent, or roots stay in the legend when they
    # have a label of their own. The curve's row would only repeat the title.
    if not str(raw_title).strip() and len(formula_geoms) == 1:
        shown = formula_geoms[0]
        title = str(getattr(shown, "_legend_label", "") or "")
        segments = getattr(shown, "_legend_math", None)
        if segments:
            labs_math["title"] = segments
        elif getattr(shown, "_legend_latex", None):
            labs_math["title"] = [{"text": title, "latex": str(shown._legend_latex)}]
        if legend:
            legend = [
                entry for entry in legend
                if not (entry.get("formula") and entry.get("label") == title)
            ]
            if not legend:
                legend = None
    else:
        title = _take("title", raw_title)
    if legend:
        for entry in legend:
            entry.pop("formula", None)
    notes: list[str] = []
    for geom, _mapped in resolved:
        for note in getattr(geom, "_notes", None) or ():
            text = str(note)
            if text and text not in notes:
                notes.append(text)
    annotations: list[dict] = []
    for geom, _mapped in resolved:
        for ann in getattr(geom, "_annotations", None) or ():
            text = str(ann.get("text") or "")
            if not text:
                continue
            annotations.append({
                "x": float(ann["x"]),
                "y": float(ann["y"]),
                "text": text,
            })
    if dropped_log:
        word = "value" if dropped_log == 1 else "values"
        notes.append(f"{dropped_log} non-positive {word} omitted on a log scale")

    size_legend = None
    if size_max is not None:
        breaks = []
        for value in _size_breaks(size_max):
            frac = 0.0 if size_max <= 0 else math.sqrt(max(value, 0.0) / size_max)
            breaks.append({
                "value": float(value),
                "label": fmt_num(float(value)),
                "t": float(frac),
            })
        size_legend = {
            "label": size_label or "size",
            "vmax": float(size_max),
            "max": float(size_units),
            "breaks": breaks,
        }

    spec = {
        "v": 1,
        "is3d": is3d,
        "theme": theme,
        "labs": {
            "title": title,
            "x": _take("x", _axis_label(g, base_map, resolved, "x", is3d)),
            "y": _take("y", _axis_label(g, base_map, resolved, "y", is3d)),
            "z": _take("z", _axis_label(g, base_map, resolved, "z", is3d)) if is3d else "",
            "color": _take("color", g.labs.get("color", base_map.get("color", ""))),
        },
        "labsMath": labs_math or None,
        "math": bool(
            labs_math
            or any(layer.get("tip") for layer in layer_specs)
            or any(entry.get("latex") or entry.get("math") for entry in (legend or []))
        ),
        "scales": {a: scales[a].spec() for a in axes},
        "color": cspec,
        "legend": legend,
        "legendPosition": _legend_position_spec(getattr(g, "legend_position", None)),
        "sizeLegend": size_legend,
        "transition": transition_meta,
        "slider": slider_meta,
        "layers": layer_specs,
        "gz": 1 if g.compress else 0,
        "coord": coord_spec,
        "notes": notes,
        "ann": annotations,
    }
    return spec, payloads


def _panel_grid(n: int, ncol: int | None, nrow: int | None) -> tuple[int, int]:
    if ncol is not None and nrow is not None:
        if ncol * nrow < n:
            nrow = int(math.ceil(n / ncol))
        return int(ncol), int(nrow)
    if ncol is not None:
        return int(ncol), int(math.ceil(n / ncol))
    if nrow is not None:
        return int(math.ceil(n / nrow)), int(nrow)
    ncol = int(math.ceil(math.sqrt(n)))
    return ncol, int(math.ceil(n / ncol))


def _clone_ggplot_with_data(g: ggplot, data) -> ggplot:
    import copy

    from plot3.table import detect_backend

    out = copy.copy(g)
    out.data = data
    out.backend = detect_backend(data) if data is not None else None
    out.layers = list(g.layers)
    out.labs = dict(g.labs)
    out.facet = None  # panels are leaf plots
    out.mapping = g.mapping
    out.cscale = g.cscale
    out.stat_density_3d = getattr(g, "stat_density_3d", None)
    out.coord = getattr(g, "coord", None)
    out._force_scales = None
    # keep theme/height/quantize
    return out


def build_doc(g: ggplot) -> str:
    """Build a standalone HTML document for *g*.

    Single-panel figures go through :func:`plot3.payload.build_payload` then
    :func:`plot3.payload.render_payload`. Faceted figures assemble a grid of
    per-panel documents (each panel uses the same payload path).
    """
    facet = getattr(g, "facet", None)
    if facet is not None:
        return _build_doc_faceted(g, facet)

    from plot3.payload import build_payload, render_payload

    return render_payload(build_payload(g), log=True)


def _global_numeric_domains(g: ggplot) -> dict[str, tuple[float, float]]:
    """Axis/colour domains from the full faceted dataset (for scales='fixed')."""
    if g.data is None:
        return {}
    mapping = dict(g.mapping)
    cols: dict[str, str] = {}
    for ax in ("x", "y", "z"):
        if ax in mapping:
            cols[ax] = mapping[ax]
    if "color" in mapping:
        cols["color"] = mapping["color"]
    domains: dict[str, tuple[float, float]] = {}
    data_cols = get_columns(g.data)
    for ax, col in cols.items():
        if col not in data_cols:
            continue
        # Materialise only this column at the domain boundary.
        sub = materialize_columns(g.data, [col])
        series = pd.to_numeric(sub[col], errors="coerce").dropna()
        if series.empty:
            continue
        lo, hi = float(series.min()), float(series.max())
        if hi <= lo:
            hi = lo + 1.0
        domains[ax] = (lo, hi)
    return domains


def _build_doc_faceted(g: ggplot, facet) -> str:
    """Render facet_wrap as a CSS grid of independent panel documents."""
    import html as _htmlesc

    if g.data is None:
        raise ValueError("ggplot has no data")
    col = facet.variable
    if not has_column(g.data, col):
        raise KeyError(f"facet column not in DataFrame: {col!r}")

    levels = unique_levels(g.data, col)

    if not levels:
        raise ValueError("facet_wrap() found no panel levels")

    ncol, nrow = _panel_grid(len(levels), facet.ncol, facet.nrow)
    theme = THEMES[g.theme_name]
    force_scales = (
        _global_numeric_domains(g) if facet.scales == "fixed" else {}
    )
    cells: list[str] = []
    total_kb = 0
    for level in levels:
        is_na = level is None or (isinstance(level, float) and np.isnan(level))
        try:
            is_na = is_na or bool(pd.isna(level))
        except (TypeError, ValueError):
            pass
        if is_na:
            panel_data = filter_equal(g.data, col, None)
            label = "NA"
        else:
            panel_data = filter_equal(g.data, col, level)
            label = str(level)
        panel = _clone_ggplot_with_data(g, panel_data)
        # Surface facet level in the panel title.
        base_title = panel.labs.get("title", "")
        panel.labs = dict(panel.labs)
        panel.labs["title"] = (
            f"{base_title} — {label}" if base_title else label
        )
        if force_scales:
            panel._force_scales = force_scales
        try:
            panel_html = build_doc(panel)
        except Exception as exc:
            panel_html = (
                "<!doctype html><html><body style='font:12px system-ui;"
                f"color:#888;padding:12px'>panel { _htmlesc.escape(label) }: "
                f"{_htmlesc.escape(str(exc))}</body></html>"
            )
        total_kb += len(panel_html) // 1024
        cells.append(
            "<div class='panel'>"
            f"<div class='plab'>{_htmlesc.escape(label)}</div>"
            f"<iframe srcdoc=\"{_htmlesc.escape(panel_html, quote=True)}\" "
            "title=\"panel\"></iframe></div>"
        )

    raw_title = str(g.labs.get("title", "") or "")
    if "$" in raw_title:
        raw_title, _segments = split_math(raw_title)
    title = _htmlesc.escape(raw_title)
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;height:100%;background:{theme["surface"]};color:{theme["ink"]};
  font:12px system-ui,-apple-system,"Segoe UI",sans-serif}}
#wrap{{box-sizing:border-box;height:100%;padding:8px;display:flex;flex-direction:column}}
#ftitle{{font-size:14px;font-weight:600;margin:0 4px 8px}}
#grid{{flex:1;min-height:0;display:grid;gap:8px;
  grid-template-columns:repeat({ncol},minmax(0,1fr));
  grid-template-rows:repeat({nrow},minmax(0,1fr))}}
.panel{{min-height:0;min-width:0;display:flex;flex-direction:column;
  border:1px solid {theme["axis"]};border-radius:6px;overflow:hidden;
  background:{theme["surface"]}}}
.plab{{padding:4px 8px;font-size:11px;color:{theme["ink2"]};
  border-bottom:1px solid {theme["grid"]}}}
.panel iframe{{flex:1;width:100%;border:0;background:{theme["surface"]}}}
</style></head><body><div id="wrap">
<div id="ftitle">{title}</div>
<div id="grid">{"".join(cells)}</div>
</div></body></html>"""
    print(
        f"plot3: facet_wrap {len(levels)} panel(s) in {nrow}x{ncol} "
        f"~{total_kb:,} KB portable HTML"
    )
    if total_kb > 1500:
        print("plot3: warning — faceted figure may exceed sslive's ~1.8 MB cap")
    return doc
