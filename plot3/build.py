"""Grammar → wire format: stats, layer specs, HTML document."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from plot3.encode import encode_norm, pack_u16
from plot3.geoms import _Geom, aes, geom_col, scale_colour_continuous
from plot3.scales import Scale, col_values
from plot3.themes import _CONT_PALETTES, _THEMES as THEMES
from plot3.viewer import _DOC_TEMPLATE as DOC_TEMPLATE


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


def expand_stat_geom(geom: _Geom, base_mapping: aes, data: pd.DataFrame) -> _Geom:
    """Turn bar/histogram/boxplot stats into concrete drawable layers."""
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
    return geom


def build_spec(g: ggplot) -> tuple[dict, list[tuple[str, str]]]:
    if g.data is None:
        raise ValueError(
            "ggplot has no data; use ggplot(df, aes(...)) or "
            "pipe data with `data >> ggplot(aes(...))`"
        )
    if not g.layers:
        raise ValueError("add a geom: ggplot(df, aes(...)) + geom_point()")

    theme = THEMES[g.theme_name]
    expanded = [
        expand_stat_geom(geom, g.mapping, g.data) for geom in g.layers
    ]
    resolved = []  # per layer: (geom, mapping)
    for geom in expanded:
        m = dict(g.mapping)
        m.update(geom.mapping)
        if "x" not in m or "y" not in m:
            raise ValueError("aes(x=, y=) are required (bar/histogram supply y)")
        resolved.append((geom, m))

    is3d = any("z" in m for _, m in resolved)
    if is3d and not all("z" in m for _, m in resolved):
        raise ValueError("mix of 2D and 3D layers: every layer needs aes(z=)")
    if is3d and any(geom.kind in {"col", "box"} for geom, _ in resolved):
        raise ValueError(
            "geom_col/geom_bar/geom_histogram/geom_boxplot are 2D only"
        )

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
            frame = g.data
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
        # Bars include the baseline at y=0 in the domain.
        if geom.kind == "col" and "y" in scales and scales["y"].kind == "num":
            scales["y"].widen(np.asarray([0.0], dtype=np.float64))
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

    # Numeric colour limits: robust 2-98 percentile by default so skewed data
    # (lidar intensity) actually varies; override via scale_colour_continuous.
    num_color = None
    if color_scale is not None and color_scale[0] == "num":
        allv = np.concatenate(num_color_vals)
        cs = g.cscale or scale_colour_continuous()
        if cs.limits == "full":
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
        if geom.kind == "line":
            keys = []
            if group_vec is not None:
                keys.append(group_vec)
            if geom.sort_x:
                keys.append(vals["x"])
            if keys:
                order = np.lexsort(tuple(reversed(keys)))

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

        if geom.kind == "box":
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
        else:
            for a in axes:
                _encode_channel(a, vals[a][order], a)

        if vals.get("color") and geom.kind != "box":
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

        if geom.kind == "line":
            if group_vec is not None:
                gv = group_vec[order]
                cut = np.flatnonzero(np.diff(gv)) + 1
                starts = np.concatenate([[0], cut])
                counts = np.diff(np.concatenate([starts, [n]]))
                spec_l["groups"] = [[int(s), int(c)]
                                    for s, c in zip(starts, counts)]
            else:
                spec_l["groups"] = [[0, int(n)]]
            spec_l["linewidth"] = float(getattr(geom, "linewidth", 2.0))
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
                # cube units; density-scaled so dense scans stay crisp
                spec_l["size"] = round(
                    min(0.02, max(0.0012,
                                  0.02 * (500.0 / max(1, n)) ** (1.0 / 3.0))), 5)
            else:
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
    }
    return spec, payloads


def build_doc(g: ggplot) -> str:
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
