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
from plot3.scales import ordered_levels, Scale, col_values, fmt_num, resolution

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
    raw = out[xcol]
    if isinstance(raw.dtype, pd.CategoricalDtype):
        present = set(raw.dropna().tolist())
        ordered = [c for c in raw.cat.categories if c in present]
        ordered += [None] if raw.isna().any() else []
    else:
        ordered = ordered_levels(raw.tolist())
    levels = ["NA" if v is None else str(v) for v in ordered]
    out[xcol] = raw.map(lambda v: "NA" if pd.isna(v) else str(v))
    out[xcol] = pd.Categorical(out[xcol], categories=levels, ordered=True)
    return out


def _grouped_histogram(geom, data, xcol, group_col, edges, centers, closed):
    """One histogram per group on shared bins, stacked like ggplot2.

    Returns None for a continuous colour, which is not a grouping.
    """
    from plot3 import stat2d

    frame = materialize_columns(data, [xcol, group_col])
    kind, _codes, _cats = col_values(frame[group_col])
    if kind != "cat":
        return None
    rows = []
    for level in ordered_levels(frame[group_col].tolist()):
        values = frame.loc[frame[group_col] == level, xcol].to_numpy(np.float64)
        values = values[np.isfinite(values)]
        if closed == "left":
            counts = _hist_counts_left_closed(values, edges)
        else:
            counts, _ = np.histogram(values, bins=edges)
        for center, count in zip(centers, counts):
            rows.append({"__x": float(center), "__y": float(count), group_col: level})
    table = pd.DataFrame(rows)
    proxy = _Geom(color=geom.const_color, alpha=geom.alpha)
    proxy.kind = "col"
    proxy.width = 1.0  # bars touch, as in a histogram
    proxy.position = getattr(geom, "position", "stack")
    out = stat2d.positioned_bars(proxy, {"x": "__x", "y": "__y", "color": group_col}, table)
    out._axis_labels = {"x": xcol, "y": "count"}
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


# Filled shapes take their colour from aes(fill=). Points, lines, error bars,
# and text use aes(colour=) only (ggplot2's default shapes ignore fill).
_FILL_KINDS = frozenset({
    "col", "bar", "histogram", "boxplot", "box", "violin", "poly", "area",
    "density", "surface", "isosurface", "density_3d_stat", "ribbon",
    "tile", "rect", "area_stat",
})


def _apply_fill(mapping: dict, kind: str) -> dict:
    """Resolve ``fill`` into the one colour channel a layer draws with."""
    out = dict(mapping)
    fill = out.pop("fill", None)
    if fill is None:
        return out
    if kind in _FILL_KINDS:
        out["color"] = fill
    elif kind == "smooth" and "color" not in out:
        out["color"] = fill  # a smoother's band follows its fill
    else:
        # Not drawn in the fill colour, but a discrete fill still groups the
        # layer (ggplot2), so error bars dodge onto the bars they belong to.
        out["__fillgroup"] = fill
    return out


_REF_KINDS = frozenset({"hline", "vline", "abline"})
_MM_TO_PX = 96.0 / 25.4


def _text_annotations(geom, vals, color_scale, theme, palette=None) -> list[dict]:
    """geom_text / geom_label rows as annotation entries in scale units."""
    xs = np.asarray(vals["x"], dtype=np.float64) + float(getattr(geom, "nudge_x", 0.0))
    ys = np.asarray(vals["y"], dtype=np.float64) + float(getattr(geom, "nudge_y", 0.0))
    labels = vals.get("label") or []
    colours: list[str | None] = [geom.const_color or theme["ink"]] * len(labels)
    if vals.get("color") and vals["color"][0] == "cat" and color_scale and color_scale[0] == "cat":
        _kind, codes, local = vals["color"]
        palette = palette or theme["cat"]
        for i, code in enumerate(codes):
            if math.isfinite(code):
                name = local[int(code)]
                index = color_scale[1].index(name) if name in color_scale[1] else 0
                colours[i] = palette[index % len(palette)]
    face = getattr(geom, "fontface", "plain")
    style = {
        "style": "label" if getattr(geom, "_box", False) else "text",
        "size": float(getattr(geom, "size", 3.88)) * _MM_TO_PX,
        "hjust": float(getattr(geom, "hjust", 0.5)),
        "vjust": float(getattr(geom, "vjust", 0.5)),
        "weight": 700 if "bold" in face else 400,
        "italic": "italic" in face,
        "overlap": not bool(getattr(geom, "check_overlap", False)),
        "alpha": 1.0 if geom.alpha is None else float(geom.alpha),
    }
    out = []
    for x, y, text, colour in zip(xs, ys, labels, colours):
        if text and math.isfinite(x) and math.isfinite(y):
            out.append({"x": float(x), "y": float(y), "text": text, "color": colour, **style})
    return out


def _ref_scale_value(scale, value) -> float | None:
    """A reference value in the scale's own units (index, seconds, log10)."""
    if scale.kind == "cat":
        key = str(value)
        return float(scale.cats.index(key)) if key in scale.cats else None
    if scale.kind == "dt":
        try:
            return float(pd.Timestamp(value).timestamp())
        except (TypeError, ValueError):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if getattr(scale, "trans", None) == "log10":
        return math.log10(number) if number > 0 else None
    return number if math.isfinite(number) else None


def _ref_specs(ref_layers, scales, theme) -> list[dict]:
    from plot3.geoms import dash_pattern

    out = []
    for ref in ref_layers:
        style = {
            "color": ref.const_color or theme["ink"],
            "width": float(getattr(ref, "linewidth", 1.0) or 1.0),
            "alpha": 1.0 if ref.alpha is None else float(ref.alpha),
            "dash": list(dash_pattern(getattr(ref, "linetype", None)) or []) or None,
        }
        if ref.kind in {"hline", "vline"}:
            axis = "y" if ref.kind == "hline" else "x"
            if axis not in scales:
                continue
            for value in ref.values:
                v = _ref_scale_value(scales[axis], value)
                if v is not None:
                    out.append({"kind": ref.kind, "value": v, **style})
        else:
            sx, sy = scales.get("x"), scales.get("y")
            linear = all(
                sc is not None and sc.kind == "num" and getattr(sc, "trans", None) is None
                for sc in (sx, sy)
            )
            if not linear:
                raise ValueError("geom_abline() needs numeric, linear x and y axes")
            out.append({
                "kind": "abline", "slope": ref.slope, "intercept": ref.intercept, **style,
            })
    return out


_STAT2D_KINDS = frozenset(
    {"jitter", "errorbar", "linerange", "pointrange", "ribbon", "smooth", "summary",
     "tile", "area_stat", "step", "segment", "rect", "qq", "qq_line", "ecdf"}
)


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
    # Resolve fill per mapping, so a layer's own colour beats a base fill.
    mapping = _apply_fill(dict(base_mapping), geom.kind)
    mapping.update(_apply_fill(dict(geom.mapping), geom.kind))
    if geom.kind in _STAT2D_KINDS or geom.kind in {"col", "bar"}:
        from plot3 import stat2d

        if geom.kind in {"col", "bar"}:
            if stat2d.wants_positioned_bars(geom, mapping, data):
                return stat2d.positioned_bars(geom, mapping, data)
        elif geom.kind == "jitter":
            return stat2d.jitter(geom, mapping, data)
        elif geom.kind in {"errorbar", "linerange", "pointrange"}:
            return stat2d.ranges(geom, mapping, data)
        elif geom.kind == "ribbon":
            return stat2d.ribbon(geom, mapping, data)
        elif geom.kind == "smooth":
            return stat2d.smooth(geom, mapping, data)
        elif geom.kind == "summary":
            return stat2d.summary(geom, mapping, data)
        else:
            handler = {
                "tile": stat2d.tile, "area_stat": stat2d.area, "step": stat2d.step,
                "segment": stat2d.segment, "rect": stat2d.rect, "qq": stat2d.qq,
                "qq_line": stat2d.qq_line, "ecdf": stat2d.ecdf,
            }[geom.kind]
            return handler(geom, mapping, data)
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
        out._axis_labels = {"y": "count"}
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
            group_col = mapping.get("color")
            if group_col and group_col != xcol and has_column(data, group_col):
                grouped = _grouped_histogram(
                    geom, data, xcol, group_col, edges, centers, closed
                )
                if grouped is not None:
                    return grouped
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
        out._axis_labels = {"x": xcol, "y": "count"}
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
        out._axis_labels = {"x": xcol, "y": "density"}
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
    """Axis title: labs, then the ggplot mapping, then a function's variable.

    ``labs(x="")`` removes the title (ggplot2's ``labs(x = NULL)``).
    """
    if axis in g.labs and g.labs[axis] is not None:
        return str(g.labs[axis])
    pscale = getattr(g, f"{axis}scale", None)
    if pscale is not None and pscale.name is not None:
        return str(pscale.name)
    mapped = base_map.get(axis)
    if mapped:
        return mapped
    # A layer's own aes(y="v") names the axis before a computed layer's
    # suggestion (a ribbon's ymin, a smoother's y).
    for geom, mapping in resolved:
        if getattr(geom, "_replace_mapping", False) or getattr(geom, "_axis_labels", None):
            continue
        own = (getattr(geom, "mapping", None) or {}).get(axis)
        if own and own == mapping.get(axis):
            return str(own)
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
        elif key == "label" and isinstance(item, list) and len(item) == n:
            vals[key] = [item[i] for i, keep in enumerate(ok) if keep]
        elif key in {"shape", "linetype"} and isinstance(item, tuple) and len(item) == 2:
            codes, cats = item
            if isinstance(codes, np.ndarray) and codes.shape[:1] == (n,):
                vals[key] = (codes[ok], cats)


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



def _forced_levels(g, ccats) -> list:
    """Colour levels shared by every facet panel, so a group keeps its colour."""
    forced = getattr(g, "_force_color_levels", None)
    if not forced:
        return list(ccats)
    return list(dict.fromkeys([str(level) for level in forced] + list(ccats)))


def _limits_in_scale(pscale, sc) -> tuple[float | None, float | None]:
    """Scale limits in the scale's own units (log10, epoch seconds)."""
    lo, hi = pscale.limits
    def conv(value):
        if value is None:
            return None
        if pscale.kind == "date" or (sc is not None and sc.kind == "dt"):
            return float(pd.Timestamp(value).timestamp())
        number = float(value)
        if sc is not None and getattr(sc, "trans", None) == "log10":
            if number <= 0:
                raise ValueError("log scale limits must be positive")
            return math.log10(number)
        return number
    a, b = conv(lo), conv(hi)
    if a is not None and b is not None and b < a:
        a, b = b, a
    return a, b


def _hex_or_none(colour):
    """#rrggbb for the viewer, which does not know R's 'grey50'."""
    if not colour:
        return colour
    from plot3.scaling import to_hex

    try:
        return to_hex(colour)
    except ValueError:
        return colour


def _label_text(value) -> str:
    """geom_text label: numbers to 4 significant figures, text as given."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if abs(number) >= 1e4:
            return f"{number:,.0f}"
        return f"{number:.4g}".replace("-", "−")
    return str(value)


def _arrow_specs(geom, vals, order, spec_l, cat_colours, theme) -> list[dict]:
    """Arrowheads at the ends of each line group, in scale units.

    The renderers draw them in screen space, so a head keeps its shape
    whatever the axes' aspect ratio.
    """
    style = geom.arrow.spec()
    xs = np.asarray(vals["x"], dtype=np.float64)[order]
    ys = np.asarray(vals["y"], dtype=np.float64)[order]
    codes = None
    if vals.get("color") and vals["color"][0] == "cat" and cat_colours:
        codes = np.asarray(vals["color"][1], dtype=np.float64)[order]
    base = spec_l.get("constColor") or theme["cat"][0]
    out = []
    for start, count in spec_l.get("groups") or []:
        start, count = int(start), int(count)
        if count < 2:
            continue
        colour = base
        if codes is not None and math.isfinite(codes[start]):
            colour = cat_colours[int(codes[start]) % len(cat_colours)]
        ends = []
        if style["ends"] in {"last", "both"}:
            ends.append((start + count - 2, start + count - 1))
        if style["ends"] in {"first", "both"}:
            ends.append((start + 1, start))
        for a, b in ends:
            if all(math.isfinite(v) for v in (xs[a], ys[a], xs[b], ys[b])):
                out.append({
                    "x0": float(xs[a]), "y0": float(ys[a]), "x1": float(xs[b]), "y1": float(ys[b]),
                    "color": colour, "width": float(spec_l.get("linewidth") or 1.0),
                    **{k: style[k] for k in ("angle", "length", "type")},
                })
    return out


def _layer_name(geom) -> str:
    """The geom as the user wrote it, for messages: geom_point, geom_line."""
    name = type(geom).__name__
    return name if name.startswith(("geom_", "stat_")) else f"geom_{geom.kind}"


def _theme_opts(g) -> dict | None:
    """theme() settings both renderers read (grid, label angle, title hjust)."""
    options = getattr(g, "theme_options", None) or {}
    out = {}
    if "panel_grid" in options:
        out["panelGrid"] = bool(options["panel_grid"])
    if "axis_text_x_angle" in options:
        out["xAngle"] = float(options["axis_text_x_angle"])
    if "plot_title_hjust" in options:
        out["titleHjust"] = float(options["plot_title_hjust"])
    return out or None


def _value_part(full: str, base: str, sep: str) -> str | None:
    """``a = 2, b = 5`` from a caption ``<base><sep>(a = 2, b = 5)``."""
    head = f"{base}{sep}("
    if base and full.startswith(head) and full.endswith(")"):
        return full[len(head):-1]
    return None


def _formula_legend(legend, formula_geoms, has_title: bool):
    """Tidy the legend rows that function curves add.

    * One curve under a title of your own: the title already names it, so
      the curve gets no row (ggplot2 draws no legend for an unmapped layer).
    * Several curves of one formula (a loop over ``a``): the formula moves
      to the legend title and each row lists only its values.

    Returns ``(legend, (pretty, latex) | None)``.
    """
    if not legend:
        return legend, None
    primary = [
        entry for entry in legend
        if entry.get("formula")
        and getattr(entry.get("_geom"), "_formula_primary", False)
        and not getattr(entry.get("_geom"), "_legend_math", None)
    ]
    if not primary:
        return legend, None
    if has_title and len(formula_geoms) == 1:
        kept = [entry for entry in legend if not any(entry is p for p in primary)]
        return kept or None, None
    if len(primary) < 2 or len(primary) != len(formula_geoms):
        return legend, None
    geoms = [entry["_geom"] for entry in primary]
    bases = {str(getattr(geom, "_title_label", "") or "") for geom in geoms}
    if len(bases) != 1:
        return legend, None
    base = bases.pop()
    base_latex = str(getattr(geoms[0], "_title_latex", "") or "")
    rows = []
    for geom in geoms:
        pretty = _value_part(str(getattr(geom, "_tip_pretty", "") or ""), base, "  ")
        latex = _value_part(
            str(getattr(geom, "_tip_latex", "") or ""), base_latex, " \\quad "
        )
        if pretty is None:
            return legend, None
        rows.append((pretty, latex))
    for entry, (pretty, latex) in zip(primary, rows):
        entry["label"] = pretty
        entry.pop("math", None)
        if latex:
            entry["latex"] = latex
        else:
            entry.pop("latex", None)
    return legend, (base, base_latex)


def _legend_title_label(current: str, legend_title, labs_math: dict) -> str:
    """Use a shared formula as the legend title unless labs(colour=) is set."""
    if current or not legend_title:
        return current
    pretty, latex = legend_title
    if latex:
        labs_math["color"] = [{"text": pretty, "latex": latex}]
    return pretty


def _discrete_codes(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Category codes and names for shape / linetype (numbers become levels)."""
    kind, values, cats = col_values(series)
    if kind == "cat":
        return np.asarray(values, dtype=np.float64), list(cats)
    labels = series.map(lambda v: "NA" if pd.isna(v) else str(v))
    order = sorted(set(labels), key=lambda t: (_num_key(t), t))
    index = {name: i for i, name in enumerate(order)}
    return labels.map(index).to_numpy(np.float64), order


def _num_key(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.inf


def _linetype_names(cats, key_scale=None) -> list:
    from plot3.geoms import LINETYPE_ORDER

    if key_scale is not None:
        return [key_scale.value_for(i, str(c)) for i, c in enumerate(cats)]
    return [LINETYPE_ORDER[i % len(LINETYPE_ORDER)] for i in range(len(cats))]


def _shape_names(cats, key_scale=None) -> list:
    from plot3.geoms import SHAPE_ORDER

    if key_scale is not None:
        return [key_scale.value_for(i, str(c)) for i, c in enumerate(cats)]
    if len(cats) > len(SHAPE_ORDER):
        raise ValueError(
            f"aes(shape=) has {len(cats)} levels; at most {len(SHAPE_ORDER)} "
            "shapes are distinct. Map a column with fewer levels to shape, or "
            "give scale_shape_manual(values=[...]) one shape per level"
        )
    return [SHAPE_ORDER[i] for i in range(len(cats))]


def _encode_dashes(spec_l: dict, geom, vals: dict, order: np.ndarray, key_scale=None) -> None:
    """Dash patterns for a line layer: one per group, or one for the layer."""
    from plot3.geoms import dash_pattern

    mapped = vals.get("linetype")
    if mapped is not None:
        codes = np.asarray(mapped[0], dtype=np.float64)[order]
        names = _linetype_names(mapped[1], key_scale)
        dashes = []
        for start, _count in spec_l.get("groups") or []:
            code = codes[int(start)]
            name = names[int(code) % len(names)] if math.isfinite(code) else "solid"
            pattern = dash_pattern(name)
            dashes.append(list(pattern) if pattern else None)
        if any(dashes):
            spec_l["dashes"] = dashes
        return
    pattern = dash_pattern(getattr(geom, "linetype", None))
    if pattern:
        spec_l["dash"] = list(pattern)


def _encode_shapes(spec_l: dict, geom, vals: dict, order: np.ndarray, li: int, payloads: list, compress: bool, key_scale=None) -> None:
    """Point symbols: per-point codes with their names, or one shape."""
    mapped = vals.get("shape")
    if mapped is not None:
        codes, cats = mapped
        names = _shape_names(cats, key_scale)
        ordered = np.asarray(codes, dtype=np.float64)[order]
        packed = np.where(np.isfinite(ordered), ordered, 0).astype("<u2")
        pid = f"p{li}sh"
        payloads.append((pid, pack_u16(packed, compress)))
        spec_l["shape"] = {"id": pid, "dtype": "u16", "names": names}
    elif getattr(geom, "shape", None):
        spec_l["shape"] = str(geom.shape)


def _aux_legends(resolved, layer_vals, legend, shape_scale=None, linetype_scale=None):
    """Shape and linetype keys: on the colour legend when they map the same
    column, otherwise a legend of their own."""
    from plot3.geoms import dash_pattern

    extra = {}
    for (geom, m), vals in zip(resolved, layer_vals):
        for aes_name, key in (("shape", "shape"), ("linetype", "dash")):
            mapped = vals.get(aes_name)
            if mapped is None or aes_name in extra:
                continue
            cats = mapped[1]
            if key == "dash":
                styles = [list(dash_pattern(n) or []) for n in _linetype_names(cats, linetype_scale)]
            else:
                styles = _shape_names(cats, shape_scale)
            column = m.get(aes_name)
            key_scale = shape_scale if key == "shape" else linetype_scale
            title = key_scale.name if key_scale is not None and key_scale.name else column
            by_label = dict(zip(cats, styles))
            if legend and column == m.get("color") and all(e["label"] in by_label for e in legend):
                for entry in legend:
                    entry[key] = by_label[entry["label"]]
                extra[aes_name] = None
            else:
                extra[aes_name] = {
                    "label": str(title),
                    "entries": [{"label": c, key: st} for c, st in zip(cats, styles)],
                }
    # One constant symbol for every point layer (geom_point(shape="triangle"))
    # also belongs on the colour keys.
    if legend and "shape" not in extra:
        constant = {
            getattr(geom, "shape", None)
            for geom, _m in resolved
            if geom.kind == "point" and getattr(geom, "shape", None)
        }
        if len(constant) == 1 and all(not e.get("shape") for e in legend):
            shape = constant.pop()
            for entry in legend:
                entry["shape"] = shape
    return legend, extra.get("shape"), extra.get("linetype")


def build_spec(g: ggplot) -> tuple[dict, list[tuple[str, str]]]:
    if not g.layers:
        raise ValueError("add a geom: ggplot(df, aes(...)) + geom_point()")
    # Function layers and vector fields sample their own grid, so a figure
    # may have no data frame.
    needs_data = any(
        getattr(layer, "kind", None) not in {"function", "vector", "hline", "vline", "abline"}
        and getattr(layer, "layer_data", None) is None
        and not getattr(layer, "_annotation", False)
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
    ref_layers = []
    for geom in g.layers:
        if getattr(geom, "kind", None) in _REF_KINDS:
            ref_layers.append(geom)  # drawn across the panel, not from rows
            continue
        if getattr(geom, "kind", None) == "isosurface" and density_stat is not None:
            geom = copy_geom_with_density_n(geom, density_stat.n)
        layers_in.append(geom)
    if not layers_in:
        raise ValueError(
            "geom_hline(), geom_vline(), and geom_abline() draw on a plot: "
            "add a data layer such as geom_point()"
        )
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
        layer_data = getattr(geom, "layer_data", None)
        result = expand_stat_geom(
            geom,
            g.mapping,
            data if layer_data is None else layer_data,
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
    from plot3.geoms import coord_flip as _coord_flip

    if isinstance(coord, _coord_flip):
        # Stats ran upright; now every layer exchanges x and y.
        from plot3.flip import flip_layers, flip_refs, flipped_figure

        expanded = flip_layers(expanded, g)
        g = flipped_figure(g)
        ref_layers = flip_refs(ref_layers)
        coord = None
    resolved = []  # per layer: (geom, mapping)
    for geom in expanded:
        # Function layers carry their own columns; don't inherit colour/group.
        if getattr(geom, "_replace_mapping", False):
            m = dict(geom.mapping)
        else:
            m = _apply_fill(dict(g.mapping), geom.kind)
            m.update(_apply_fill(dict(geom.mapping), geom.kind))
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
    # scale_x_discrete(limits=[...]) / xlim("a", "b"): that order, first.
    for _axis_name, _pscale in (("x", getattr(g, "xscale", None)), ("y", getattr(g, "yscale", None))):
        if _pscale is not None and _pscale.kind == "discrete" and _pscale.limits:
            scales[_axis_name] = Scale("cat")
            scales[_axis_name].cats = list(_pscale.limits)
    color_scale = None  # ("num", lo, hi) | ("cat", cats)
    num_color_vals: list[np.ndarray] = []
    dropped_log = 0
    missing_notes: list[str] = []
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

    def _absorb_level_positions(v: np.ndarray, levels: list[str], axis: str = "x") -> np.ndarray:
        if _axis_trans(axis) == "log10":
            raise ValueError(f"scale_{axis}_log10() cannot be used with categorical {axis}")
        sc = scales.get(axis)
        if sc is None or (sc.kind == "num" and not math.isfinite(sc.lo)):
            sc = scales[axis] = Scale("cat")
        elif sc.kind != "cat":
            raise ValueError(
                "aes x: layers disagree on scale type (cat vs num)"
            )
        merged = list(dict.fromkeys(sc.cats + [str(level) for level in levels]))
        sc.cats = merged
        where = {str(level): merged.index(str(level)) for level in levels}
        flat = np.asarray(v, dtype=np.float64)
        # Nearest category, clamped: a tile's outer edge sits at n - 0.5,
        # which rounds past the last level. Keep the offset from it.
        top = max(len(levels) - 1, 0)
        out = np.full(flat.shape, np.nan)
        for i, p in enumerate(flat):
            if not math.isfinite(p) or not levels:
                continue
            k = int(min(max(math.floor(p + 0.5), 0), top))
            out[i] = where[str(levels[k])] + (p - k)
        return out

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
            if color_scale is None:
                color_scale = ["cat", _forced_levels(g, ccats)]
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

    def _limit_rows(vals: dict, geom) -> None:
        """Rows outside scale limits are dropped, as in ggplot2, and reported."""
        if "frames" in vals or getattr(geom, "data_override", None) is not None:
            return
        n = len(vals["x"])
        ok = np.ones(n, dtype=bool)
        for axis_name in ("x", "y"):
            pscale = getattr(g, f"{axis_name}scale", None)
            if pscale is None or axis_name not in vals or pscale.limits is None:
                continue
            v = np.asarray(vals[axis_name], dtype=np.float64)
            if pscale.kind == "discrete":
                ok &= np.isfinite(v) & (v < len(pscale.limits))
                continue
            lo, hi = _limits_in_scale(pscale, scales.get(axis_name))
            tol = 1e-9 * max(1.0, abs(lo or 0.0), abs(hi or 0.0))
            if lo is not None:
                ok &= v >= lo - tol
            if hi is not None:
                ok &= v <= hi + tol
        if ok.all():
            return
        removed = int((~ok).sum())
        word = "row" if removed == 1 else "rows"
        missing_notes.append(
            f"Removed {removed} {word} outside the scale limits ({_layer_name(geom)})"
        )
        _filter_rows(vals, ok)

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
            frame = getattr(geom, "layer_data", None)
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
                    if color_scale is None:
                        color_scale = ["cat", _forced_levels(g, ccats)]
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
        if geom.kind == "text":
            if "label" not in m:
                raise ValueError("geom_text() requires aes(label=)")
            cols.append(m["label"])
        if geom.kind == "point" and "shape" in m:
            cols.append(m["shape"])
        if geom.kind == "line" and "linetype" in m:
            cols.append(m["linetype"])
        if transition is not None and getattr(transition, "column", None):
            cols.append(transition.column)
        sub = materialize_columns(frame, list(dict.fromkeys(cols)))
        if getattr(geom, "data_override", None) is None and frame is not None:
            # ggplot2 says when rows are dropped; plot3 used to do it silently.
            removed = n_rows(frame) - len(sub)
            if removed > 0:
                word = "row" if removed == 1 else "rows"
                missing_notes.append(
                    f"Removed {removed} {word} containing missing values "
                    f"({_layer_name(geom)})"
                )

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
            levels = (
                getattr(geom, "_violin_levels", None) if a == "x"
                else getattr(geom, "_y_levels", None) if a == "y"
                else None
            )
            if levels is not None and kind == "num":
                # Numeric positions on a categorical axis (violins, dodged
                # bars, error bars, heatmap rows, a flipped plot). Join the
                # shared category list by name, so layers agree on order.
                vals[a] = _absorb_level_positions(v, list(levels), a)
                continue
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
        # Violin, dodged bars, error bars: numeric x positions on a
        # categorical axis were joined to the category list above.
        if "color" in m:
            kind, cv, ccats = col_values(sub[m["color"]])
            if kind == "cat" or (kind == "num" and ccats):
                if color_scale is None:
                    color_scale = ["cat", _forced_levels(g, ccats)]
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
        if geom.kind == "point" and "shape" in m:
            vals["shape"] = _discrete_codes(sub[m["shape"]])
        if geom.kind == "line" and "linetype" in m:
            vals["linetype"] = _discrete_codes(sub[m["linetype"]])
        if geom.kind == "text":
            vals["label"] = [_label_text(v) for v in sub[m["label"]].tolist()]
        if geom.kind == "point" and "size" in m:
            skind, sv, _ = col_values(sub[m["size"]])
            if skind != "num":
                raise ValueError("aes(size=) needs a numeric column")
            sv = np.asarray(sv, dtype=np.float64)
            vals["size"] = np.where(np.isfinite(sv) & (sv >= 0), sv, np.nan)
            if not size_label:
                size_label = str(m["size"])
        _drop_log_rows(vals)
        _limit_rows(vals, geom)
        layer_vals.append(vals)

    # Reference lines are part of the picture: ggplot2 widens the scales so
    # geom_hline(yintercept=0) is never off the panel.
    for ref in ref_layers:
        axis = {"hline": "y", "vline": "x"}.get(ref.kind)
        if axis and axis in scales and scales[axis].kind in {"num", "dt"}:
            values = [_ref_scale_value(scales[axis], v) for v in ref.values]
            scales[axis].widen(np.asarray([v for v in values if v is not None], dtype=np.float64))

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

    # scale_x_continuous(limits=, breaks=, labels=), xlim(), scale_y_reverse()…
    for axis_name in ("x", "y"):
        pscale = getattr(g, f"{axis_name}scale", None)
        sc = scales.get(axis_name)
        if pscale is None or sc is None:
            continue
        if pscale.kind == "discrete" and sc.kind == "cat" and pscale.limits:
            sc.cats = list(pscale.limits)
            sc.finish()
        elif pscale.limits is not None and sc.kind in {"num", "dt"}:
            lo, hi = _limits_in_scale(pscale, sc)
            if lo is not None:
                sc.lo = lo
            if hi is not None:
                sc.hi = hi
            if sc.hi <= sc.lo:
                sc.hi = sc.lo + 1.0
        sc.custom = pscale
        if pscale.trans == "reverse" and sc.kind == "num":
            sc.lo, sc.hi = sc.hi, sc.lo

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
        if g.cscale is None and any(getattr(geom, "_default_ramp", None) for geom, _m in resolved):
            # A formula surface coloured by height spans its whole range.
            cs = scale_colour_continuous(limits="full")
        user_scale = getattr(g, "colour_scale", None)
        if user_scale is not None and user_scale.kind == "discrete":
            raise ValueError(
                "this colour is numeric; scale_colour_manual() and brewer are for "
                "groups. Use scale_colour_gradient(), or map a text column"
            )
        if user_scale is not None and user_scale.kind == "continuous":
            # ggplot2 maps the whole data range, unless limits are given.
            cs = scale_colour_continuous(limits=user_scale.limits or "full")
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
    arrows: list[dict] = []
    # One palette for the colour/fill groups: a scale_*_manual / brewer /
    # viridis_d scale, else the theme's colours extended past eight groups.
    cat_colours: list[str] = []
    if color_scale is not None and color_scale[0] == "cat":
        user_scale = getattr(g, "colour_scale", None)
        if user_scale is not None and user_scale.kind == "continuous":
            raise ValueError(
                "scale_colour_gradient() is for numbers; this colour is categorical. "
                "Use scale_colour_manual() or scale_colour_brewer()"
            )
        if user_scale is not None:
            cat_colours = user_scale.colours(list(color_scale[1]), theme["cat"])
        else:
            from plot3.scaling import extend_palette

            cat_colours = extend_palette(theme["cat"], len(color_scale[1]))

    # Text layers are positioned through the scales like any layer (a label
    # at x="Sat" lands on Sat), then drawn as annotations by both renderers.
    text_anns: list[dict] = []
    kept_pairs = []
    for (geom, m), vals in zip(resolved, layer_vals):
        if geom.kind == "text":
            if is3d:
                raise ValueError("geom_text() is for 2D figures")
            text_anns.extend(_text_annotations(geom, vals, color_scale, theme, cat_colours))
        else:
            kept_pairs.append(((geom, m), vals))
    if len(kept_pairs) != len(resolved):
        resolved = [pair for pair, _vals in kept_pairs]
        layer_vals = [vals for _pair, vals in kept_pairs]

    for li, ((geom, m), vals) in enumerate(zip(resolved, layer_vals)):
        n = len(vals["x"])
        order = np.arange(n)
        group_vec = None
        if "group" in vals:
            group_vec = vals["group"][0]
        elif vals.get("color") and vals["color"][0] == "cat":
            group_vec = vals["color"][1]
        if vals.get("linetype") is not None:
            # Each linetype level is its own line, as in ggplot2.
            lt_codes, lt_cats = vals["linetype"]
            group_vec = (
                lt_codes if group_vec is None
                else np.asarray(group_vec, dtype=np.float64) * (len(lt_cats) + 1) + lt_codes
            )
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
            # Error bars, reference lines, and text default to the ink colour
            # (black on a light theme), as in ggplot2.
            "constColor": _hex_or_none(
                geom.const_color
                or (theme["ink"] if getattr(geom, "_ink_default", False) else None)
            ),
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
            if geom.kind == "line":
                _encode_dashes(spec_l, geom, vals, order, getattr(g, "linetype_scale", None))
                if getattr(geom, "arrow", None) is not None:
                    arrows.extend(_arrow_specs(geom, vals, order, spec_l, cat_colours, theme))
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
            if geom.kind == "point":
                _encode_shapes(
                    spec_l, geom, vals, order, li, payloads, g.compress,
                    getattr(g, "shape_scale", None),
                )
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
            cspec = {"kind": "cat", "palette": cat_colours, "cats": cats}
            user_scale = getattr(g, "colour_scale", None)
            if user_scale is not None:
                legend = user_scale.legend_entries(list(cats), cat_colours)
                for entry in legend:
                    entry.pop("_level", None)
            else:
                legend = [{"label": c, "color": cat_colours[i]} for i, c in enumerate(cats)]
        else:
            default_ramp = next(
                (getattr(geom, "_default_ramp") for geom, _m in resolved if getattr(geom, "_default_ramp", None)),
                "blue",
            )
            pal = (g.cscale.palette if g.cscale else default_ramp)
            ramp = _CONT_PALETTES.get(pal, theme["seq"])
            user_scale = getattr(g, "colour_scale", None)
            if user_scale is not None and user_scale.kind == "continuous":
                ramp = user_scale.ramp(num_color[0], num_color[1])
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
                entry["_geom"] = geom
                segments = getattr(geom, "_legend_math", None)
                latex = getattr(geom, "_legend_latex", None)
                if segments:
                    entry["math"] = segments
                elif latex:
                    entry["latex"] = str(latex)
            entries.append(entry)
        if entries:
            legend = entries

    legend, shape_legend, linetype_legend = _aux_legends(
        resolved, layer_vals, legend,
        getattr(g, "shape_scale", None), getattr(g, "linetype_scale", None),
    )

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
        # Symbolic formula, not the value list. That list stays in the legend
        # when the two differ, so the title does not become a long caption.
        title = str(
            getattr(shown, "_title_label", None)
            or getattr(shown, "_legend_label", "")
            or ""
        )
        segments = getattr(shown, "_legend_math", None)
        title_latex = getattr(shown, "_title_latex", None) or getattr(shown, "_legend_latex", None)
        if segments:
            labs_math["title"] = segments
        elif title_latex:
            labs_math["title"] = [{"text": title, "latex": str(title_latex)}]
        if legend:
            legend = [
                entry for entry in legend
                if not (entry.get("formula") and entry.get("label") == title)
            ]
            if not legend:
                legend = None
    else:
        title = _take("title", raw_title)
    legend, legend_title = _formula_legend(legend, formula_geoms, bool(str(raw_title).strip()))
    if legend:
        for entry in legend:
            entry.pop("formula", None)
            entry.pop("_geom", None)
    notes: list[str] = list(dict.fromkeys(missing_notes))
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
                "style": "area",
            })
    annotations.extend(text_anns)
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
            "color": (
                _legend_title_label(
                    _take(
                        "color",
                        g.labs.get(
                            "color",
                            getattr(getattr(g, "colour_scale", None), "name", None)
                            or base_map.get("color") or base_map.get("fill")
                            or next(
                                (getattr(geom, "_colour_title") for geom, _m in resolved
                                 if getattr(geom, "_colour_title", None)),
                                "",
                            ),
                        ),
                    ),
                    legend_title,
                    labs_math,
                )
                if (getattr(g, "theme_options", None) or {}).get("legend_title", True)
                else ""
            ),
            "subtitle": _take("subtitle", g.labs.get("subtitle")),
            "caption": _take("caption", g.labs.get("caption")),
            "tag": _take("tag", g.labs.get("tag")),
        },
        "themeOpts": _theme_opts(g),
        "facetChild": bool(getattr(g, "_facet_child", False)) or None,
        "labsMath": labs_math or None,
        "refs": _ref_specs(ref_layers, scales, theme) or None,
        "arrows": arrows or None,
        "shapeLegend": shape_legend,
        "linetypeLegend": linetype_legend,
        "math": bool(
            labs_math
            or any(layer.get("tip") for layer in layer_specs)
            or any(entry.get("latex") or entry.get("math") for entry in (legend or []))
        ),
        "scales": {a: scales[a].spec() for a in axes},
        "color": cspec,
        "legend": legend,
        "legendPosition": (
            "none" if getattr(g, "_facet_child", False)
            else _legend_position_spec(getattr(g, "legend_position", None))
        ),
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


def _level_text(level) -> str:
    try:
        if level is None or bool(pd.isna(level)):
            return "NA"
    except (TypeError, ValueError):
        pass
    return str(level)


def _subset(data, column, level):
    from plot3.table import filter_equal

    if column is None:
        return data
    return filter_equal(data, column, None if _level_text(level) == "NA" else level)


def _facet_levels(data, column) -> list:
    """Panel order: a categorical column's own order, else ggplot2's sort."""
    from plot3.table import detect_backend

    levels = unique_levels(data, column)

    keep = False
    if detect_backend(data) == "pandas":
        keep = isinstance(data[column].dtype, pd.CategoricalDtype)
    return ordered_levels(levels, keep_order=keep)


def _facet_colour_levels(g: ggplot) -> list[str] | None:
    """Categorical colour/fill levels of the whole dataset, in scale order."""
    candidates = [g.mapping.get("color"), g.mapping.get("fill")]
    for layer in g.layers:
        mapping = getattr(layer, "mapping", None) or {}
        candidates += [mapping.get("color"), mapping.get("fill")]
    for column in candidates:
        if column and has_column(g.data, column):
            kind, _values, cats = col_values(materialize_columns(g.data, [column])[column])
            if kind == "cat":
                return list(cats)
    return None


def _facet_legend_html(spec: dict, theme: dict) -> str:
    """One colour legend for a faceted HTML figure."""
    import html as _htmlesc

    entries = spec.get("legend") or []
    if not entries:
        return ""
    esc = _htmlesc.escape
    title = str((spec.get("labs") or {}).get("color") or "")
    rows = "".join(
        f"<div><span class='sw' style='background:{esc(str(e.get('color')))}'></span>"
        f"{esc(str(e.get('label')))}</div>"
        for e in entries
    )
    head = f"<b style='color:{theme['ink']}'>{esc(title)}</b>" if title else ""
    return f"<div id='flegend'>{head}{rows}</div>"


def facet_cells(g: ggplot) -> dict:
    """Panels of a faceted figure, for the HTML viewer and for ggsave.

    ``cells`` lists ``{"row", "col", "fig", "strip"}``; ``fig`` is None for
    an empty facet_grid combination. facet_wrap puts each label on its own
    panel (``strip``); facet_grid uses ``col_strips`` above the top row and
    ``row_strips`` to the right, as in ggplot2.
    """
    from plot3.geoms import facet_grid as _facet_grid

    facet = g.facet
    if g.data is None:
        raise ValueError("ggplot has no data")
    force = _global_numeric_domains(g) if facet.scales == "fixed" else {}
    colour_levels = _facet_colour_levels(g)

    def child(panel):
        # Panels draw only data; the figure draws title, legend, axis titles.
        panel._facet_child = True
        if colour_levels:
            panel._force_color_levels = colour_levels
        if force:
            panel._force_scales = force
        return panel
    header = {
        key: g.labs.get(key) for key in ("title", "subtitle", "caption", "tag") if g.labs.get(key)
    }
    cells: list[dict] = []
    if not isinstance(facet, _facet_grid):
        column = facet.variable
        if not has_column(g.data, column):
            raise KeyError(f"facet column not in DataFrame: {column!r}")
        levels = _facet_levels(g.data, column)
        if not levels:
            raise ValueError("facet_wrap() found no panel levels")
        ncol, nrow = _panel_grid(len(levels), facet.ncol, facet.nrow)
        for index, level in enumerate(levels):
            label = _level_text(level)
            panel = child(_clone_ggplot_with_data(g, _subset(g.data, column, level)))
            panel.labs = {
                k: v for k, v in panel.labs.items()
                if k not in {"title", "subtitle", "caption", "tag"}
            }
            row, col = divmod(index, ncol)
            cells.append({"row": row, "col": col, "fig": panel, "strip": label})
        return {"ncol": ncol, "nrow": nrow, "cells": cells, "col_strips": None,
                "row_strips": None, "header": header, "kind": "wrap"}

    for column in (facet.rows, facet.cols):
        if column is not None and not has_column(g.data, column):
            raise KeyError(f"facet column not in DataFrame: {column!r}")
    row_levels = _facet_levels(g.data, facet.rows) if facet.rows else [None]
    col_levels = _facet_levels(g.data, facet.cols) if facet.cols else [None]
    nrow, ncol = len(row_levels), len(col_levels)
    for r, row_level in enumerate(row_levels):
        rows_data = _subset(g.data, facet.rows, row_level)
        for c, col_level in enumerate(col_levels):
            piece = _subset(rows_data, facet.cols, col_level)
            if n_rows(piece) == 0:
                cells.append({"row": r, "col": c, "fig": None, "strip": None})
                continue
            panel = child(_clone_ggplot_with_data(g, piece))
            panel.labs = {
                k: v for k, v in panel.labs.items()
                if k not in {"title", "subtitle", "caption", "tag"}
            }
            cells.append({"row": r, "col": c, "fig": panel, "strip": None})
    return {
        "ncol": ncol,
        "nrow": nrow,
        "cells": cells,
        "col_strips": [_level_text(v) for v in col_levels] if facet.cols else None,
        "row_strips": [_level_text(v) for v in row_levels] if facet.rows else None,
        "header": header,
        "kind": "grid",
    }


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
    """facet_wrap / facet_grid as a CSS grid of independent panel documents."""
    import html as _htmlesc

    layout = facet_cells(g)
    ncol, nrow = layout["ncol"], layout["nrow"]
    theme = THEMES[g.theme_name]
    col_strips = layout.get("col_strips")
    row_strips = layout.get("row_strips")
    by_pos = {(c["row"], c["col"]): c for c in layout["cells"]}
    first = next(c["fig"] for c in layout["cells"] if c["fig"] is not None)
    first_spec, _pairs = build_spec(first)
    shared_x = str((first_spec.get("labs") or {}).get("x") or "")
    shared_y = str((first_spec.get("labs") or {}).get("y") or "")
    legend_html = _facet_legend_html(first_spec, theme)
    cells: list[str] = []
    total_kb = 0
    count = 0
    esc = _htmlesc.escape
    # facet_grid: a strip row above the panels and a strip column to their right.
    if col_strips:
        cells += [f"<div class='cstrip'>{esc(t)}</div>" for t in col_strips]
        if row_strips:
            cells.append("<div></div>")
    for row in range(nrow):
        for col in range(ncol):
            cell = by_pos.get((row, col))
            if cell is None or cell["fig"] is None:
                cells.append("<div class='empty'></div>")
                continue
            label = cell.get("strip")
            try:
                panel_html = build_doc(cell["fig"])
            except Exception as exc:
                panel_html = (
                    "<!doctype html><html><body style='font:12px system-ui;"
                    f"color:#888;padding:12px'>panel {esc(str(label or ''))}: "
                    f"{esc(str(exc))}</body></html>"
                )
            total_kb += len(panel_html) // 1024
            count += 1
            strip = f"<div class='plab'>{esc(label)}</div>" if label else ""
            cells.append(
                "<div class='panel'>" + strip
                + f"<iframe srcdoc=\"{esc(panel_html, quote=True)}\" title=\"panel\"></iframe></div>"
            )
        if row_strips:
            cells.append(f"<div class='rstrip'><span>{esc(row_strips[row])}</span></div>")

    header = layout.get("header") or {}
    def text(key: str) -> str:
        raw = str(header.get(key, "") or "")
        if "$" in raw:
            raw, _segments = split_math(raw)
        return esc(raw)
    columns = f"repeat({ncol},minmax(0,1fr))" + (" 26px" if row_strips else "")
    rows = ("24px " if col_strips else "") + f"repeat({nrow},minmax(0,1fr))"
    tag = f"<b style='margin-right:8px'>{text('tag')}</b>" if header.get("tag") else ""
    subtitle = f"<div id='fsub'>{text('subtitle')}</div>" if header.get("subtitle") else ""
    caption = f"<div id='fcap'>{text('caption')}</div>" if header.get("caption") else ""
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;height:100%;background:{theme["surface"]};color:{theme["ink"]};
  font:12px system-ui,-apple-system,"Segoe UI",sans-serif}}
#wrap{{box-sizing:border-box;height:100%;padding:8px;display:flex;flex-direction:column}}
#ftitle{{font-size:14px;font-weight:600;margin:0 4px 2px}}
#fsub{{font-size:12px;color:{theme["ink2"]};margin:0 4px 6px}}
#fcap{{font-size:11px;color:{theme["muted"]};text-align:right;margin:4px 4px 0}}
#body{{flex:1;min-height:0;display:flex;gap:6px}}
#main{{flex:1;min-width:0;display:flex;flex-direction:column}}
#ytitle{{width:16px;display:flex;align-items:center;justify-content:center;color:{theme["ink2"]}}}
#ytitle span{{writing-mode:vertical-rl;transform:rotate(180deg)}}
#xtitle{{text-align:center;color:{theme["ink2"]};padding-top:4px}}
#flegend{{align-self:flex-start;margin-top:24px;padding:6px 9px;border:1px solid {theme["grid"]};
  border-radius:6px;font-size:11px;line-height:1.7;color:{theme["ink2"]}}}
#flegend .sw{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}}
#grid{{flex:1;min-height:0;display:grid;gap:8px;
  grid-template-columns:{columns};grid-template-rows:{rows}}}
.panel{{min-height:0;min-width:0;display:flex;flex-direction:column;
  border:1px solid {theme["axis"]};border-radius:6px;overflow:hidden;
  background:{theme["surface"]}}}
.plab{{padding:4px 8px;font-size:11px;color:{theme["ink2"]};
  border-bottom:1px solid {theme["grid"]}}}
.cstrip,.rstrip{{background:{theme["grid"]};color:{theme["ink2"]};font-size:11px;
  font-weight:600;display:flex;align-items:center;justify-content:center;border-radius:4px}}
.rstrip span{{writing-mode:vertical-rl}}
.panel iframe{{flex:1;width:100%;border:0;background:{theme["surface"]}}}
</style></head><body><div id="wrap">
<div id="ftitle">{tag}{text("title")}</div>{subtitle}
<div id="body"><div id="ytitle"><span>{esc(shared_y)}</span></div>
<div id="main"><div id="grid">{"".join(cells)}</div><div id="xtitle">{esc(shared_x)}</div></div>
{legend_html}</div>{caption}
</div></body></html>"""
    kind = "facet_grid" if layout.get("kind") == "grid" else "facet_wrap"
    print(
        f"plot3: {kind} {count} panel(s) in {nrow}x{ncol} "
        f"~{total_kb:,} KB portable HTML"
    )
    if total_kb > 1500:
        print("plot3: warning — faceted figure may exceed sslive's ~1.8 MB cap")
    return doc
