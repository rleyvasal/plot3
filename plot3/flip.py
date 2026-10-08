"""coord_flip: draw every layer with x and y exchanged.

Stats run in the usual orientation first (a boxplot summarises y by x).
Then bars, boxes, and filled densities are lowered to rectangles, lines, and
polygons, and every layer swaps its x and y. Both renderers then draw an
ordinary plot: no sideways drawing code is needed anywhere.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

from plot3 import stat2d
from plot3.geoms import _Geom, aes

_SWAP = {
    "x": "y", "y": "x", "xmin": "ymin", "ymin": "xmin",
    "xmax": "ymax", "ymax": "xmax", "xend": "yend", "yend": "xend",
}


def swap_mapping(mapping) -> dict:
    return {_SWAP.get(key, key): value for key, value in dict(mapping or {}).items()}


def flipped_figure(g):
    """The ggplot with its own mapping, titles, and scales exchanged."""
    out = copy.copy(g)
    out.mapping = aes()
    out.mapping.update(swap_mapping(g.mapping))
    out.labs = {_SWAP.get(k, k): v for k, v in dict(g.labs).items()}
    def moved(scale, axis):
        if scale is None:
            return None
        scale = copy.copy(scale)
        scale.axis = axis
        return scale

    out.xscale = moved(getattr(g, "yscale", None), "x")
    out.yscale = moved(getattr(g, "xscale", None), "y")
    out.scale_x, out.scale_y = getattr(g, "scale_y", None), getattr(g, "scale_x", None)
    return out


def flip_refs(refs: list) -> list:
    out = []
    for ref in refs:
        clone = copy.copy(ref)
        if ref.kind == "hline":
            clone.kind = "vline"
        elif ref.kind == "vline":
            clone.kind = "hline"
        elif ref.kind == "abline":
            # y = a + b x drawn with axes exchanged is x = a + b y.
            if ref.slope == 0:
                clone.kind = "vline"
                clone.values = [ref.intercept]
            else:
                clone.slope = 1.0 / ref.slope
                clone.intercept = -ref.intercept / ref.slope
        out.append(clone)
    return out


def flip_layers(layers: list, g) -> list:
    out = []
    for geom in layers:
        for piece in _lower(geom, g):
            out.append(_swap(piece))
    return out


def _merged(geom, g) -> dict:
    from plot3.build import _apply_fill

    if getattr(geom, "_replace_mapping", False):
        return dict(geom.mapping)
    mapping = _apply_fill(dict(g.mapping), geom.kind)
    mapping.update(_apply_fill(dict(geom.mapping), geom.kind))
    return mapping


def _lower(geom, g) -> list:
    kind = getattr(geom, "kind", "")
    data = geom.data_override if getattr(geom, "data_override", None) is not None else g.data
    if kind == "col":
        mapping = _merged(geom, g)
        proxy = _Geom(color=geom.const_color, alpha=geom.alpha)
        proxy.kind = "col"
        proxy.width = float(getattr(geom, "width", 0.9) or 0.9)
        proxy.position = "identity"
        bars = stat2d.positioned_bars(proxy, mapping, data)
        bars._axis_labels = dict(getattr(geom, "_axis_labels", None) or {}) or getattr(bars, "_axis_labels", None)
        return [bars]
    if kind == "box":
        return _lower_box(geom, g)
    if kind == "area":
        return [_lower_area(geom, g, data)]
    if kind == "line" and getattr(geom, "sort_x", False) and getattr(geom, "_groups", None) is None:
        # geom_line joins points in order of x; keep that order once x is vertical.
        mapping = _merged(geom, g)
        cols = [c for c in (mapping.get("x"), mapping.get("y"), mapping.get("color"), mapping.get("group")) if c]
        from plot3.table import materialize_columns

        frame = materialize_columns(data, cols)
        keys = [c for c in (mapping.get("group"), mapping.get("color")) if c] + [mapping["x"]]
        frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
        clone = copy.copy(geom)
        clone.data_override = frame
        clone.sort_x = False
        clone.mapping = aes()
        clone.mapping.update(mapping)
        clone._replace_mapping = True
        return [clone]
    return [geom]


def _lower_box(geom, g) -> list:
    """A boxplot as a box polygon, median and whisker lines, and outliers."""
    stats = geom.data_override
    xcol = geom.mapping.get("x")
    colour = geom.mapping.get("color")
    colour = colour if colour and colour in stats.columns else None
    axis = stat2d._axis(stats[xcol])
    groups, levels = stat2d._colour_groups(stats, colour if colour != xcol else None)
    n_groups = len(levels) if colour and colour != xcol else 1
    width = float(getattr(geom, "width", 0.75)) / max(n_groups, 1)
    offsets = stat2d._dodge_offsets(n_groups, float(getattr(geom, "width", 0.75)))
    centres = axis.values + (offsets[groups] if n_groups > 1 else 0.0)
    half = width / 2.0
    ymin, lower, middle, upper, ymax = (stats[c].to_numpy(np.float64) for c in ("ymin", "lower", "middle", "upper", "ymax"))
    keep = {colour: stats[colour].tolist()} if colour else {}
    rx, ry, rect_groups, extra = stat2d._rect_rows(centres - half, centres + half, lower, upper, keep)
    box_frame = pd.DataFrame({"x": axis.out(np.asarray(rx)), "y": ry})
    box_map = {"x": "x", "y": "y"}
    if colour:
        box_frame[colour] = extra[colour]
        box_map["colour"] = colour
    box = stat2d._layer("poly", box_frame, box_map, geom, _groups=rect_groups, linewidth=1.2)
    box.alpha = geom.alpha if geom.alpha is not None else 0.35
    stat2d._levels_hook(box, axis)
    lx, ly, lc, starts = [], [], [], []
    for i in range(len(stats)):
        c = centres[i]
        for path_x, path_y in (
            ([c - half, c + half], [middle[i], middle[i]]),   # median
            ([c, c], [ymin[i], lower[i]]),                   # lower whisker
            ([c, c], [upper[i], ymax[i]]),                   # upper whisker
        ):
            if not all(math.isfinite(v) for v in path_y):
                continue
            starts.append([len(lx), 2])
            lx += path_x
            ly += path_y
            if colour:
                lc += [stats[colour].iloc[i]] * 2
    line_frame = pd.DataFrame({"x": axis.out(np.asarray(lx)), "y": ly})
    line_map = {"x": "x", "y": "y"}
    if colour:
        line_frame[colour] = lc
        line_map["colour"] = colour
    lines = stat2d._layer("line", line_frame, line_map, geom, _groups=starts, linewidth=1.5)
    stat2d._levels_hook(lines, axis)
    layers = [box, lines]
    outliers = getattr(geom, "_outlier_frame", None)
    y_name = getattr(geom, "_y_name", None)
    if outliers is not None and len(outliers) and y_name in outliers.columns:
        index = {str(level): i for i, level in enumerate(axis.levels or [])}
        ox = np.array([index.get(str(v), np.nan) for v in outliers[xcol]], dtype=np.float64)
        dots = pd.DataFrame({"x": ox, "y": outliers[y_name].to_numpy(np.float64)})
        dot_map = {"x": "x", "y": "y"}
        if colour and colour in outliers.columns:
            dots[colour] = outliers[colour].to_numpy()
            dot_map["colour"] = colour
        dot = stat2d._layer("point", dots, dot_map, geom)
        dot.size = float(getattr(geom, "outlier_size", 3.0))
        stat2d._levels_hook(dot, axis)
        layers.append(dot)
    for layer in layers:
        layer._axis_labels = {"x": xcol, "y": y_name or "y"}
    return layers


def _lower_area(geom, g, data) -> _Geom:
    """A filled density or area as a band from its baseline."""
    mapping = _merged(geom, g)
    xcol, ycol = mapping["x"], mapping["y"]
    colour = mapping.get("color")
    colour = colour if colour and colour in data.columns else None
    frame = data
    axis = stat2d._axis(frame[xcol])
    baseline = float(getattr(geom, "_baseline", 0.0) or 0.0)
    groups, levels = stat2d._colour_groups(frame, colour)
    pieces = []
    for gi in range(len(levels)):
        rows = np.flatnonzero(groups == gi)
        xs = axis.values[rows]
        ys = frame[ycol].to_numpy(np.float64)[rows]
        value = frame[colour].iloc[rows[0]] if colour and rows.size else None
        pieces.append(stat2d._band(xs, np.full_like(ys, baseline), ys, value, colour, axis))
    band = stat2d._bands_layer(pieces, colour, geom, axis, geom.alpha if geom.alpha is not None else 0.35)
    band._axis_labels = dict(getattr(geom, "_axis_labels", None) or {"x": xcol, "y": ycol})
    band._baseline_zero = True
    return band


def _swap(geom) -> _Geom:
    out = copy.copy(geom)
    out.mapping = aes()
    out.mapping.update(swap_mapping(geom.mapping))
    out._violin_levels = getattr(geom, "_y_levels", None)
    out._y_levels = getattr(geom, "_violin_levels", None)
    labels = getattr(geom, "_axis_labels", None)
    if labels:
        out._axis_labels = {_SWAP.get(k, k): v for k, v in labels.items()}
    if getattr(geom, "_annotations", None):
        out._annotations = [dict(a, x=a["y"], y=a["x"]) for a in geom._annotations]
    lock = getattr(geom, "_axis_lock", None)
    if lock:
        out._axis_lock = {_SWAP.get(k, k): v for k, v in lock.items()}
    if hasattr(geom, "nudge_x") or hasattr(geom, "nudge_y"):
        out.nudge_x, out.nudge_y = getattr(geom, "nudge_y", 0.0), getattr(geom, "nudge_x", 0.0)
    if getattr(geom, "_x_domain", None) is not None:
        out._x_domain = None
    out.sort_x = False
    return out


_SIDE_SWAP = {"b": "l", "l": "b", "t": "r", "r": "t"}


def flip_rugs(rugs: list) -> list:
    """geom_rug under coord_flip: x values move to the left side, y to the bottom."""
    out = []
    for rug in rugs:
        clone = copy.copy(rug)
        clone.mapping = aes()
        clone.mapping.update(swap_mapping(rug.mapping))
        clone.sides = "".join(_SIDE_SWAP[side] for side in rug.sides)
        out.append(clone)
    return out
