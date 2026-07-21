"""Grammar → wire format: stats, layer specs, HTML document."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import copy

from plot3.encode import encode_norm, pack_u16, pack_u32
from plot3.geoms import _Geom, aes, geom_col, scale_colour_continuous
from plot3.scales import Scale, col_values
from plot3.stats3d import isosurface_levels, regular_grid_mesh
from plot3.themes import _CONT_PALETTES, _THEMES as THEMES
from plot3.viewer import _DOC_TEMPLATE as DOC_TEMPLATE


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


def expand_stat_geom(geom: _Geom, base_mapping: aes, data: pd.DataFrame) -> _Geom:
    """Turn statistical geoms into concrete drawable layers."""
    mapping = dict(base_mapping)
    mapping.update(geom.mapping)
    if geom.kind == "bar":
        if "x" not in mapping:
            raise ValueError("geom_bar() requires aes(x=)")
        xcol = mapping["x"]
        if xcol not in data.columns:
            raise KeyError(f"column(s) not in DataFrame: {[xcol]}")
        counts = (
            data.groupby(xcol, dropna=False, observed=True, sort=False)
            .size()
            .rename("y")
            .reset_index()
        )
        out = geom_col(aes(x=xcol, y="y"), width=getattr(geom, "width", 0.9),
                       color=geom.const_color, colour=None, alpha=geom.alpha)
        out.data_override = counts
        out.const_color = geom.const_color
        out.alpha = geom.alpha
        return out
    if geom.kind == "histogram":
        if "x" not in mapping:
            raise ValueError("geom_histogram() requires aes(x=)")
        xcol = mapping["x"]
        if xcol not in data.columns:
            raise KeyError(f"column(s) not in DataFrame: {[xcol]}")
        values = pd.to_numeric(data[xcol], errors="coerce").dropna().to_numpy(
            dtype=np.float64
        )
        if values.size == 0:
            frame = pd.DataFrame({"x": np.array([], dtype=float),
                                  "y": np.array([], dtype=float)})
            width = 1.0
        else:
            bins = max(1, int(getattr(geom, "bins", 30)))
            counts, edges = np.histogram(values, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            width = float(np.median(np.diff(edges))) if len(edges) > 1 else 1.0
            frame = pd.DataFrame({"x": centers, "y": counts.astype(np.float64)})
        out = geom_col(aes(x="x", y="y"), width=getattr(geom, "width", 1.0),
                       color=geom.const_color, alpha=geom.alpha)
        out.data_override = frame
        out.const_color = geom.const_color
        out.alpha = geom.alpha
        out._bar_width_data = width  # absolute data units
        return out
    if geom.kind == "boxplot":
        if "x" not in mapping or "y" not in mapping:
            raise ValueError("geom_boxplot() requires aes(x=, y=)")
        xcol, ycol = mapping["x"], mapping["y"]
        missing = [c for c in (xcol, ycol) if c not in data.columns]
        if missing:
            raise KeyError(f"column(s) not in DataFrame: {missing}")
        colour_col = mapping.get("color")
        group_cols = [xcol]
        if colour_col and colour_col != xcol and colour_col in data.columns:
            group_cols.append(colour_col)
        coef = float(getattr(geom, "coef", 1.5))
        rows: list[dict] = []
        outlier_rows: list[dict] = []
        grouping = data.groupby(
            group_cols, dropna=False, observed=True, sort=False
        )
        for key, piece in grouping:
            key_tuple = key if isinstance(key, tuple) else (key,)
            values = pd.to_numeric(piece[ycol], errors="coerce").to_numpy(
                dtype=np.float64
            )
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
        if xcol not in data.columns:
            raise KeyError(f"column(s) not in DataFrame: {[xcol]}")
        colour_col = mapping.get("color")
        n_grid = int(getattr(geom, "n", 512))
        adjust = float(getattr(geom, "adjust", 1.0))
        fill = bool(getattr(geom, "fill", True))
        pieces: list[pd.DataFrame] = []
        if colour_col and colour_col in data.columns:
            groups = data.groupby(colour_col, dropna=False, observed=True, sort=False)
            for key, piece in groups:
                grid, dens = _kde_1d(
                    pd.to_numeric(piece[xcol], errors="coerce").to_numpy(),
                    n=n_grid,
                    adjust=adjust,
                )
                if grid.size == 0:
                    continue
                frame = pd.DataFrame({"x": grid, "y": dens, colour_col: key})
                pieces.append(frame)
        else:
            grid, dens = _kde_1d(
                pd.to_numeric(data[xcol], errors="coerce").to_numpy(),
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
        missing = [c for c in (xcol, ycol) if c not in data.columns]
        if missing:
            raise KeyError(f"column(s) not in DataFrame: {missing}")
        colour_col = mapping.get("color")
        n_grid = int(getattr(geom, "n", 128))
        adjust = float(getattr(geom, "adjust", 1.0))
        width = float(getattr(geom, "width", 0.9))
        # Stable category order of appearance.
        if isinstance(data[xcol].dtype, pd.CategoricalDtype):
            levels = [str(c) for c in data[xcol].cat.categories]
        else:
            levels = list(dict.fromkeys(data[xcol].astype(str).tolist()))
        level_index = {level: i for i, level in enumerate(levels)}
        rows: list[dict] = []
        grouping_cols = [xcol]
        if colour_col and colour_col != xcol and colour_col in data.columns:
            grouping_cols.append(colour_col)
        for key, piece in data.groupby(
            grouping_cols, dropna=False, observed=True, sort=False
        ):
            key_tuple = key if isinstance(key, tuple) else (key,)
            x_key = key_tuple[0]
            x_pos = float(level_index.get(str(x_key), 0))
            grid_y, dens = _kde_1d(
                pd.to_numeric(piece[ycol], errors="coerce").to_numpy(),
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
        vertices, indices, nx, ny = regular_grid_mesh(
            data, xcol, ycol, zcol, ccol=ccol
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
        for col in (xcol, ycol, zcol):
            if col not in data.columns:
                raise KeyError(f"column(s) not in DataFrame: {[col]}")
        pts = (
            data.loc[:, [xcol, ycol, zcol]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .to_numpy(dtype=np.float64)
        )
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


# Geoms that cannot enter a 3D figure (stat expansions use these kinds too).
_2D_ONLY_KINDS = frozenset(
    {"col", "box", "area", "poly", "bar", "histogram", "boxplot", "density", "violin"}
)
_3D_POINT_KINDS = frozenset({"point", "line", "surface", "isosurface"})


def build_spec(g: ggplot) -> tuple[dict, list[tuple[str, str]]]:
    if g.data is None:
        raise ValueError(
            "ggplot has no data; use ggplot(df, aes(...)) or "
            "pipe data with `data >> ggplot(aes(...))`"
        )
    if not g.layers:
        raise ValueError("add a geom: ggplot(df, aes(...)) + geom_point()")

    data = g.data
    coord = getattr(g, "coord", None)
    mesh_kinds = {"surface", "isosurface"}
    has_mesh = any(
        getattr(layer, "kind", None) in mesh_kinds for layer in g.layers
    )
    if (
        coord is not None
        and getattr(coord, "max_points", None)
        and not has_mesh
    ):
        n_rows = len(data)
        cap = int(coord.max_points)
        if n_rows > cap:
            step = max(1, (n_rows + cap - 1) // cap)
            data = data.iloc[::step].copy()

    theme = THEMES[g.theme_name]
    # Apply optional stat_density_3d options onto isosurface layers.
    density_stat = getattr(g, "stat_density_3d", None)
    layers_in = []
    for geom in g.layers:
        if getattr(geom, "kind", None) == "isosurface" and density_stat is not None:
            geom = copy_geom_with_density_n(geom, density_stat.n)
        layers_in.append(geom)
    expanded = [
        expand_stat_geom(geom, g.mapping, data) for geom in layers_in
    ]
    resolved = []  # per layer: (geom, mapping)
    for geom in expanded:
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

    axes = ["x", "y", "z"] if is3d else ["x", "y"]
    scales: dict[str, Scale] = {}
    color_scale = None  # ("num", lo, hi) | ("cat", cats)
    num_color_vals: list[np.ndarray] = []

    def _absorb_position(axis: str, kind: str, v: np.ndarray, cats: list[str]):
        sc = scales.get(axis)
        if sc is None:
            sc = scales[axis] = Scale(kind)
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
            v = np.array([remap.get(int(c), -1) for c in v], dtype=np.float64)
            sc.cats = merged
        else:
            sc.widen(v)
        return v

    # Pass 1 — per-layer values + global scale domains
    layer_vals = []
    for geom, m in resolved:
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
            if "color" in m and m["color"] in frame.columns:
                cols.append(m["color"])
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise KeyError(f"column(s) not in DataFrame: {missing}")
            sub = frame[list(dict.fromkeys(cols))].dropna(
                subset=[xcol, *y_stat_cols]
            )
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
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"column(s) not in DataFrame: {missing}")
        sub = frame[list(dict.fromkeys(cols))].dropna()
        vals = {}
        for a in axes:
            kind, v, cats = col_values(sub[m[a]])
            vals[a] = _absorb_position(a, kind, v, cats)
        # Bars / densities include the baseline at y=0 in the domain.
        if (
            geom.kind in {"col", "area"}
            or getattr(geom, "_baseline_zero", False)
        ) and "y" in scales and scales["y"].kind == "num":
            scales["y"].widen(np.asarray([0.0], dtype=np.float64))
        # Violin: numeric x positions with categorical tick labels.
        if getattr(geom, "_violin_levels", None) is not None:
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
        layer_vals.append(vals)

    for a in axes:
        scales[a].finish()

    # Optional forced domains (facet_wrap scales="fixed").
    force = getattr(g, "_force_scales", None) or {}
    for ax, (lo, hi) in force.items():
        if ax in scales and scales[ax].kind == "num":
            scales[ax].lo = float(lo)
            scales[ax].hi = float(hi)
            if scales[ax].hi <= scales[ax].lo:
                scales[ax].hi = scales[ax].lo + 1.0

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

    # Pass 2 — encode payloads per layer (quantized against the shared scales)
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
        if geom.kind in {"line", "area"}:
            keys = []
            if group_vec is not None:
                keys.append(group_vec)
            if geom.kind == "area" or getattr(geom, "sort_x", False):
                keys.append(vals["x"])
            if keys:
                order = np.lexsort(tuple(reversed(keys)))
        # poly keeps authoring order (closed violin contours).

        spec_l = {
            "kind": geom.kind,
            "n": int(n),
            "alpha": geom.alpha,
            "constColor": geom.const_color,
        }

        def _encode_channel(name: str, values: np.ndarray, axis: str):
            sc = scales[axis]
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
            if group_vec is not None:
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
                if scy.kind == "num":
                    y_span = max(scy.hi - scy.lo, 1e-12)
                    spec_l["y0"] = float(
                        np.clip((0.0 - scy.lo) / y_span, 0.0, 1.0)
                    )
                else:
                    spec_l["y0"] = 0.0
        elif geom.kind in {"col", "box"}:
            # Bar/box width in normalized [0,1] x-space for the renderer.
            scx = scales["x"]
            span = max(scx.hi - scx.lo, 1e-12)
            if hasattr(geom, "_bar_width_data"):
                data_w = float(geom._bar_width_data) * float(
                    getattr(geom, "width", 1.0)
                )
            elif scx.kind == "cat":
                data_w = float(getattr(geom, "width", 0.75 if geom.kind == "box" else 0.9))
            else:
                xs = np.asarray(vals["x"][order], dtype=np.float64)
                if len(xs) >= 2:
                    gaps = np.diff(np.sort(np.unique(xs)))
                    step = float(np.median(gaps)) if len(gaps) else 1.0
                else:
                    step = span * 0.08
                data_w = step * float(
                    getattr(geom, "width", 0.75 if geom.kind == "box" else 0.9)
                )
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
            if getattr(geom, "size", None) is not None:
                spec_l["size"] = float(geom.size)
            elif is3d:
                # scene units; density-scaled so dense scans stay crisp
                spec_l["size"] = round(
                    min(0.02, max(0.0012,
                                  0.02 * (500.0 / max(1, n)) ** (1.0 / 3.0))), 5)
            else:
                # pixels
                spec_l["size"] = 6.0 if n <= 2000 else (4.0 if n <= 20000 else 2.5)
            if spec_l["alpha"] is None:
                spec_l["alpha"] = 0.85 if n <= 50000 else 0.6
        if spec_l["alpha"] is None:
            spec_l["alpha"] = 1.0
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

    base_map = dict(g.mapping)
    if is3d:
        coord_spec = (
            coord.to_spec()
            if coord is not None
            else {"aspect": "data", "sizeMode": "scene", "maxPoints": None}
        )
    else:
        coord_spec = None
        if coord is not None:
            raise ValueError(
                "coord_3d() requires a 3D figure (map aes(z=...) on layers)"
            )

    spec = {
        "v": 1,
        "is3d": is3d,
        "theme": theme,
        "labs": {
            "title": g.labs.get("title", ""),
            "x": g.labs.get("x", base_map.get("x", "x")),
            "y": g.labs.get("y", base_map.get("y", "y")),
            "z": g.labs.get("z", base_map.get("z", "z")) if is3d else "",
            "color": g.labs.get("color", base_map.get("color", "")),
        },
        "scales": {a: scales[a].spec() for a in axes},
        "color": cspec,
        "legend": legend,
        "layers": layer_specs,
        "gz": 1 if g.compress else 0,
        "coord": coord_spec,
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


def _clone_ggplot_with_data(g: ggplot, data: pd.DataFrame) -> ggplot:
    import copy

    from plot3.ggplot import ggplot as Ggplot

    out = copy.copy(g)
    out.data = data
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
    facet = getattr(g, "facet", None)
    if facet is not None:
        return _build_doc_faceted(g, facet)

    spec, payloads = build_spec(g)
    blocks = "\n".join(
        f'<script type="text/plain" id="{pid}">{b64}</script>'
        for pid, b64 in payloads
    )
    doc = (
        DOC_TEMPLATE
        .replace("__SPEC__", json.dumps(spec, separators=(",", ":")))
        .replace("__PAYLOADS__", blocks)
    )
    kb = len(doc) // 1024
    rows = sum(sp["n"] for sp in spec["layers"])
    print(f"plot3: {len(spec['layers'])} layer(s), {rows:,} rows -> {kb:,} KB "
          f"portable HTML{' (3D)' if spec['is3d'] else ''}")
    if kb > 1500:
        print("plot3: warning — figure may exceed sslive's ~1.8 MB in-slide cap")
    return doc


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
    for ax, col in cols.items():
        if col not in g.data.columns:
            continue
        series = pd.to_numeric(g.data[col], errors="coerce").dropna()
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
    if col not in g.data.columns:
        raise KeyError(f"facet column not in DataFrame: {col!r}")

    if isinstance(g.data[col].dtype, pd.CategoricalDtype):
        levels = [c for c in g.data[col].cat.categories if (g.data[col] == c).any()]
        # also include NaN panel if present
        if g.data[col].isna().any():
            levels = list(levels) + [pd.NA]
    else:
        levels = list(dict.fromkeys(g.data[col].tolist()))

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
        if pd.isna(level):
            mask = g.data[col].isna()
            label = "NA"
        else:
            mask = g.data[col] == level
            label = str(level)
        panel_data = g.data.loc[mask].copy()
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

    title = _htmlesc.escape(str(g.labs.get("title", "")))
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
