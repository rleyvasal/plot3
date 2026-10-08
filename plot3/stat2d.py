"""Positioned bars, ranges, ribbons, smoothers, and summaries.

Every layer here expands into kinds the viewer and the static exporter
already draw: ``poly`` (bars as rectangles, ribbons), ``line`` (error bars,
smoothers), and ``point`` (jitter, the dot of a point range). Nothing new has
to be taught to WebGL or to the PNG/SVG/PDF writer.

Categorical x is drawn at integer positions with the category names on the
axis (the ``_violin_levels`` hook build_spec already honours), so dodged
bars and the error bars on top of them can sit between the ticks.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from plot3.geoms import _Geom, aes
from plot3.scales import col_values, resolution
from plot3.special import qt
from plot3.table import has_column, materialize_columns

_ALPHA_BAR = 0.9
_ALPHA_BAND = 0.25


# ── positions ────────────────────────────────────────────────────────────────


def position_kind(value: Any, default: str) -> tuple[str, float | None]:
    """``("stack" | "dodge" | "fill" | "identity", dodge width or None)``."""
    if value is None:
        return default, None
    if isinstance(value, str):
        kind = value.strip().lower()
        width = None
    else:
        kind = str(getattr(value, "kind", "") or "")
        width = getattr(value, "width", None)
    if kind not in {"stack", "dodge", "fill", "identity"}:
        raise ValueError(
            "position must be 'stack', 'dodge', 'fill', or 'identity' "
            f"(or position_dodge() etc.), not {value!r}"
        )
    return kind, None if width is None else float(width)


# ── axis helpers ─────────────────────────────────────────────────────────────


class _Axis:
    """x as numbers, plus the category names when x is discrete."""

    def __init__(self, kind: str, values: np.ndarray, levels: list[str] | None):
        self.kind = kind  # "num" | "dt" | "cat"
        self.values = values
        self.levels = levels

    def out(self, positions: np.ndarray) -> Any:
        """Positions back in the column type the x scale expects."""
        if self.kind == "dt":
            return pd.to_datetime(np.asarray(positions, dtype=np.float64), unit="s")
        return np.asarray(positions, dtype=np.float64)

    def step(self) -> float:
        if self.kind == "cat":
            return 1.0
        finite = np.unique(self.values[np.isfinite(self.values)])
        return float(resolution(finite, zero=False)) if finite.size else 1.0


def _axis(series: pd.Series, *, discrete: bool = False) -> _Axis:
    if discrete and not isinstance(series.dtype, pd.CategoricalDtype):
        # Count bars keep first-appearance order, like geom_bar does today.
        labels = series.map(lambda v: "NA" if pd.isna(v) else str(v))
        levels = list(dict.fromkeys(labels.tolist()))
        index = {level: i for i, level in enumerate(levels)}
        return _Axis("cat", labels.map(index).to_numpy(np.float64), levels)
    kind, values, cats = col_values(series)
    return _Axis(kind, np.asarray(values, dtype=np.float64), cats if kind == "cat" else None)


def _colour_groups(frame: pd.DataFrame, colour: str | None) -> tuple[np.ndarray, list[Any]]:
    """Group index per row and the levels in legend order (one group if none)."""
    if not colour or colour not in frame.columns:
        return np.zeros(len(frame), dtype=int), [None]
    kind, codes, cats = col_values(frame[colour])
    if kind != "cat":
        # A numeric colour is a continuous gradient, not a group.
        return np.zeros(len(frame), dtype=int), [None]
    return codes.astype(int), cats


def _frame(data: Any, cols: list[str]) -> pd.DataFrame:
    return materialize_columns(data, [c for c in dict.fromkeys(cols) if c])


def _layer(kind: str, frame: pd.DataFrame, mapping: dict, geom: _Geom, **extra) -> _Geom:
    out = _Geom(aes(**mapping), color=geom.const_color, alpha=geom.alpha)
    out.kind = kind
    out.data_override = frame.reset_index(drop=True)
    out.const_color = geom.const_color
    out.alpha = geom.alpha
    out._replace_mapping = True
    out.sort_x = False
    for key, value in extra.items():
        setattr(out, key, value)
    return out


def _title(out: _Geom, axis: str, text: Any) -> None:
    """Axis title when the ggplot() mapping does not name one."""
    if text:
        labels = dict(getattr(out, "_axis_labels", None) or {})
        labels[axis] = str(text)
        out._axis_labels = labels


def _colour_mapping(colour: str | None) -> dict:
    return {"colour": colour} if colour else {}


def _levels_hook(out: _Geom, axis: _Axis) -> _Geom:
    if axis.kind == "cat" and axis.levels is not None:
        out._violin_levels = list(axis.levels)
    return out


def _dodge_offsets(n_groups: int, width: float) -> np.ndarray:
    """Centre offset of each group inside a dodged slot of ``width``."""
    if n_groups <= 1:
        return np.zeros(1)
    slot = width / n_groups
    return (np.arange(n_groups) - (n_groups - 1) / 2.0) * slot


# ── bars ─────────────────────────────────────────────────────────────────────


def wants_positioned_bars(geom: _Geom, mapping: dict, data: Any) -> bool:
    """Stacked, dodged, or filled bars need the rectangle path."""
    kind, _w = position_kind(getattr(geom, "position", None), "stack")
    colour = mapping.get("color")
    x = mapping.get("x")
    grouped = bool(colour) and colour != x and has_column(data, colour)
    if kind == "fill":
        return True
    if not grouped:
        return False
    if geom.kind == "bar":
        return True  # counts must be split by the colour group
    return kind in {"stack", "dodge"}


def positioned_bars(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    xcol = mapping.get("x")
    if not xcol:
        raise ValueError(f"geom_{geom.kind}() requires aes(x=)")
    colour = mapping.get("color")
    colour = colour if colour and colour != xcol and has_column(data, colour) else None
    kind, dodge_width = position_kind(getattr(geom, "position", None), "stack")
    rel = float(getattr(geom, "width", 0.9))
    if geom.kind == "bar":
        frame = _frame(data, [xcol, colour])
        keys = [xcol] + ([colour] if colour else [])
        counts = frame.groupby(keys, sort=False, dropna=False, observed=True).size()
        frame = counts.reset_index(name="__y")
        axis = _axis(frame[xcol], discrete=True)
    else:
        ycol = mapping.get("y")
        if not ycol:
            raise ValueError("geom_col() requires aes(x=, y=)")
        frame = _frame(data, [xcol, ycol, colour])
        frame = frame.rename(columns={ycol: "__y"}) if ycol != "__y" else frame
        axis = _axis(frame[xcol])
    groups, levels = _colour_groups(frame, colour)
    n_groups = len(levels)
    xs = axis.values
    ys = frame["__y"].to_numpy(np.float64)
    full = axis.step() * rel
    bottoms = np.zeros_like(ys)
    tops = ys.copy()
    lefts = xs - full / 2.0
    rights = xs + full / 2.0
    if kind in {"stack", "fill"}:
        # ggplot2 puts the first level on top: stack from the last level up.
        # Negative values stack downward on their own.
        rank = (n_groups - 1) - groups
        bottoms = np.empty_like(ys)
        tops = np.empty_like(ys)
        for x_value in np.unique(xs):
            here = np.flatnonzero(xs == x_value)
            here = here[np.argsort(rank[here], kind="stable")]
            up = down = 0.0
            for i in here:
                if ys[i] >= 0:
                    bottoms[i], tops[i] = up, up + ys[i]
                    up += ys[i]
                else:
                    bottoms[i], tops[i] = down, down + ys[i]
                    down += ys[i]
            if kind == "fill":
                total = up if up > 0 else 1.0
                bottoms[here] /= total
                tops[here] /= total
    elif kind == "dodge":
        width = full if dodge_width is None else axis.step() * dodge_width
        offsets = _dodge_offsets(n_groups, width)
        slot = full / max(n_groups, 1)
        centres = xs + offsets[groups]
        lefts = centres - slot / 2.0
        rights = centres + slot / 2.0
    rows_x: list[float] = []
    rows_y: list[float] = []
    rows_c: list[Any] = []
    starts: list[list[int]] = []
    colour_values = frame[colour].tolist() if colour else None
    for i in range(len(frame)):
        if not (math.isfinite(bottoms[i]) and math.isfinite(tops[i])):
            continue
        start = len(rows_x)
        # Left edge up, right edge down: the pairing poly drawing expects.
        rows_x += [lefts[i], lefts[i], rights[i], rights[i]]
        rows_y += [bottoms[i], tops[i], tops[i], bottoms[i]]
        if colour_values is not None:
            rows_c += [colour_values[i]] * 4
        starts.append([start, 4])
    out_frame = pd.DataFrame({"x": axis.out(np.asarray(rows_x)), "y": rows_y})
    mapping_out = {"x": "x", "y": "y"}
    if colour:
        out_frame[colour] = rows_c
        mapping_out["colour"] = colour
    out = _layer(
        "poly", out_frame, mapping_out, geom,
        _groups=starts, _baseline_zero=True, linewidth=0.0,
    )
    out.alpha = geom.alpha if geom.alpha is not None else _ALPHA_BAR
    _title(out, "y", "count" if geom.kind == "bar" else None)
    if kind == "fill":
        _title(out, "y", "proportion")
    return _levels_hook(out, axis)


# ── jitter ───────────────────────────────────────────────────────────────────


def jitter(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    xcol, ycol = mapping.get("x"), mapping.get("y")
    if not xcol or not ycol:
        raise ValueError("geom_jitter() requires aes(x=, y=)")
    keep = [c for c in (mapping.get("color"), mapping.get("size"), mapping.get("group")) if c]
    frame = _frame(data, [xcol, ycol, *keep])
    axis = _axis(frame[xcol])
    yaxis = _axis(frame[ycol])
    rng = np.random.default_rng(getattr(geom, "seed", 0))
    width = getattr(geom, "width", None)
    height = getattr(geom, "height", None)
    # ggplot2: 40% of the resolution in each direction, both ways.
    width = 0.4 * axis.step() if width is None else float(width) * (axis.step() if axis.kind == "cat" else 1.0)
    height = 0.4 * yaxis.step() if height is None else float(height)
    xs = axis.values + rng.uniform(-width, width, len(frame))
    ys = yaxis.values + (rng.uniform(-height, height, len(frame)) if height > 0 else 0.0)
    out_frame = pd.DataFrame({"x": axis.out(xs), "y": yaxis.out(ys)})
    mapping_out = {"x": "x", "y": "y"}
    for key, col in (("colour", mapping.get("color")), ("size", mapping.get("size")), ("group", mapping.get("group"))):
        if col:
            out_frame[col] = frame[col].to_numpy()
            mapping_out[key] = col
    out = _layer("point", out_frame, mapping_out, geom)
    out.size = getattr(geom, "size", None)
    out = _levels_hook(out, axis)
    if yaxis.kind == "cat" and yaxis.levels is not None:
        raise ValueError("geom_jitter() needs a numeric y")
    return out


# ── ranges ───────────────────────────────────────────────────────────────────


def _range_frame(geom: _Geom, mapping: dict, data: Any, need_y: bool):
    name = f"geom_{geom.kind}"
    xcol = mapping.get("x")
    lo, hi = mapping.get("ymin"), mapping.get("ymax")
    ycol = mapping.get("y")
    if not xcol or not lo or not hi or (need_y and not ycol):
        wanted = "x=, y=, ymin=, ymax=" if need_y else "x=, ymin=, ymax="
        raise ValueError(f"{name}() requires aes({wanted})")
    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    frame = _frame(data, [xcol, lo, hi, ycol if need_y else None, colour])
    return frame, xcol, lo, hi, ycol, colour


def _dodged_x(geom: _Geom, axis: _Axis, groups: np.ndarray, n_groups: int) -> np.ndarray:
    kind, width = position_kind(getattr(geom, "position", None), "identity")
    if kind != "dodge" or n_groups <= 1:
        return axis.values
    width = 0.9 if width is None else width
    return axis.values + _dodge_offsets(n_groups, axis.step() * width)[groups]


def ranges(geom: _Geom, mapping: dict, data: Any):
    """geom_errorbar, geom_linerange, geom_pointrange."""
    kind = geom.kind
    frame, xcol, lo, hi, ycol, colour = _range_frame(geom, mapping, data, kind == "pointrange")
    axis = _axis(frame[xcol])
    groups, levels = _colour_groups(frame, colour)
    xs = _dodged_x(geom, axis, groups, len(levels))
    y_lo = frame[lo].to_numpy(np.float64)
    y_hi = frame[hi].to_numpy(np.float64)
    cap = 0.0
    if kind == "errorbar":
        dodge_kind, _w = position_kind(getattr(geom, "position", None), "identity")
        slot = axis.step() * (1.0 / len(levels) if dodge_kind == "dodge" else 1.0)
        cap = 0.5 * slot * float(getattr(geom, "width", 0.5))
    rows_x: list[float] = []
    rows_y: list[float] = []
    rows_c: list[Any] = []
    starts: list[list[int]] = []
    colour_values = frame[colour].tolist() if colour else None
    for i in range(len(frame)):
        x, a, b = xs[i], y_lo[i], y_hi[i]
        if not (math.isfinite(x) and math.isfinite(a) and math.isfinite(b)):
            continue
        if cap > 0:
            # One path per bar: top cap, down the stem, bottom cap.
            path_x = [x - cap, x + cap, x, x, x - cap, x + cap]
            path_y = [b, b, b, a, a, a]
        else:
            path_x, path_y = [x, x], [a, b]
        starts.append([len(rows_x), len(path_x)])
        rows_x += path_x
        rows_y += path_y
        if colour_values is not None:
            rows_c += [colour_values[i]] * len(path_x)
    line_frame = pd.DataFrame({"x": axis.out(np.asarray(rows_x)), "y": rows_y})
    line_map = {"x": "x", "y": "y"}
    if colour:
        line_frame[colour] = rows_c
        line_map["colour"] = colour
    linewidth = float(getattr(geom, "linewidth", 1.0) or 1.0)
    line = _layer("line", line_frame, line_map, geom, _groups=starts, linewidth=linewidth)
    inherited = (getattr(geom, "_axis_labels", None) or {}).get("y")
    _title(line, "y", inherited or ycol or lo)
    _levels_hook(line, axis)
    if kind != "pointrange":
        return line
    dot_frame = pd.DataFrame({"x": axis.out(xs), "y": frame[ycol].to_numpy(np.float64)})
    dot_map = {"x": "x", "y": "y"}
    if colour:
        dot_frame[colour] = frame[colour].to_numpy()
        dot_map["colour"] = colour
    dot = _layer("point", dot_frame, dot_map, geom)
    dot.size = getattr(geom, "size", None) or 7.0
    dot.alpha = geom.alpha if geom.alpha is not None else 1.0
    _title(dot, "y", inherited or ycol)
    _levels_hook(dot, axis)
    return [line, dot]


# ── ribbons ──────────────────────────────────────────────────────────────────


def _band(xs, lo, hi, colour_value, colour, axis: _Axis):
    order = np.argsort(xs, kind="stable")
    xs, lo, hi = xs[order], lo[order], hi[order]
    ok = np.isfinite(xs) & np.isfinite(lo) & np.isfinite(hi)
    xs, lo, hi = xs[ok], lo[ok], hi[ok]
    if xs.size < 2:
        return None
    # Lower edge left to right, upper edge right to left: pairs (i, n-1-i)
    # share an x, which the poly strip expects.
    px = np.concatenate([xs, xs[::-1]])
    py = np.concatenate([lo, hi[::-1]])
    frame = pd.DataFrame({"x": axis.out(px), "y": py})
    if colour:
        frame[colour] = [colour_value] * len(px)
    return frame


def _bands_layer(pieces, colour, geom, axis, alpha):
    frames, starts, at = [], [], 0
    for piece in pieces:
        if piece is None:
            continue
        starts.append([at, len(piece)])
        at += len(piece)
        frames.append(piece)
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    mapping = {"x": "x", "y": "y", **_colour_mapping(colour)}
    out = _layer("poly", frame, mapping, geom, _groups=starts, linewidth=0.0)
    out.alpha = geom.alpha if geom.alpha is not None else alpha
    return _levels_hook(out, axis)


def ribbon(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    xcol, lo, hi = mapping.get("x"), mapping.get("ymin"), mapping.get("ymax")
    if not xcol or not lo or not hi:
        raise ValueError("geom_ribbon() requires aes(x=, ymin=, ymax=)")
    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    frame = _frame(data, [xcol, lo, hi, colour])
    axis = _axis(frame[xcol])
    groups, levels = _colour_groups(frame, colour)
    colour_values = frame[colour].to_numpy() if colour else None
    pieces = []
    for g in range(len(levels)):
        rows = np.flatnonzero(groups == g)
        pieces.append(
            _band(
                axis.values[rows],
                frame[lo].to_numpy(np.float64)[rows],
                frame[hi].to_numpy(np.float64)[rows],
                colour_values[rows[0]] if colour and rows.size else None,
                colour,
                axis,
            )
        )
    out = _bands_layer(pieces, colour, geom, axis, 0.3)
    if out is None:
        raise ValueError("geom_ribbon() needs at least two rows per group")
    _title(out, "y", lo)
    return out


# ── smoothers ────────────────────────────────────────────────────────────────


def _lm(x: np.ndarray, y: np.ndarray, grid: np.ndarray, level: float):
    n = x.size
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fit = coef[0] + coef[1] * grid
    if n <= 2:
        return fit, None
    resid = y - design @ coef
    sigma2 = float(resid @ resid) / (n - 2)
    inv = np.linalg.pinv(design.T @ design)
    g = np.column_stack([np.ones(grid.size), grid])
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", g, inv, g) * sigma2, 0.0))
    return fit, (se, n - 2)


def _loess_row(x: np.ndarray, x0: float, q: int, degree: int) -> np.ndarray | None:
    """Weights l(x0) so that fit(x0) = l @ y (local polynomial, tricube)."""
    dist = np.abs(x - x0)
    if q >= x.size:
        radius = float(dist.max()) * (q / x.size)
    else:
        radius = float(np.partition(dist, q - 1)[q - 1])
    radius = max(radius, 1e-12)
    w = np.clip(1.0 - (dist / radius) ** 3, 0.0, None) ** 3
    keep = w > 0
    if np.count_nonzero(keep) <= degree:
        return None
    xc = x[keep] - x0
    basis = np.column_stack([xc ** k for k in range(degree + 1)])
    wb = basis * w[keep, None]
    gram = basis.T @ wb
    try:
        row = np.linalg.solve(gram, wb.T)[0]
    except np.linalg.LinAlgError:
        row = (np.linalg.pinv(gram) @ wb.T)[0]
    full = np.zeros(x.size)
    full[keep] = row
    return full


def _loess(x: np.ndarray, y: np.ndarray, grid: np.ndarray, level: float, span: float):
    """R's loess defaults: degree 2, tricube weights, ``span`` of the points."""
    n = x.size
    degree = 2 if n > 3 else 1
    q = max(degree + 2, int(math.ceil(span * n)))
    rows = [_loess_row(x, g, q, degree) for g in grid]
    fit = np.array([float(r @ y) if r is not None else math.nan for r in rows])
    # Residual scale and equivalent degrees of freedom from the smoother's
    # own rows at the data (a subsample keeps big n affordable).
    idx = np.arange(n) if n <= 600 else np.linspace(0, n - 1, 600).astype(int)
    diag, resid = [], []
    for i in idx:
        r = _loess_row(x, float(x[i]), q, degree)
        if r is None:
            continue
        diag.append(r[i])
        resid.append(y[i] - float(r @ y))
    if len(resid) < 3:
        return fit, None
    trace = float(np.mean(diag)) * n
    df = max(n - trace, 1.0)
    sigma2 = float(np.mean(np.square(resid))) * n / df
    se = np.array([
        math.sqrt(sigma2 * float(r @ r)) if r is not None else math.nan for r in rows
    ])
    return fit, (se, df)


def smooth(geom: _Geom, mapping: dict, data: Any):
    xcol, ycol = mapping.get("x"), mapping.get("y")
    if not xcol or not ycol:
        raise ValueError("geom_smooth() requires aes(x=, y=)")
    colour = mapping.get("color")
    colour = colour if colour and colour not in {xcol, ycol} and has_column(data, colour) else None
    frame = _frame(data, [xcol, ycol, colour])
    axis = _axis(frame[xcol])
    if axis.kind == "cat":
        raise ValueError("geom_smooth() needs a numeric or date x")
    method = str(getattr(geom, "method", "loess")).lower()
    if method not in {"loess", "lm"}:
        raise ValueError("geom_smooth(method=) is 'loess' or 'lm'")
    level = float(getattr(geom, "level", 0.95))
    span = float(getattr(geom, "span", 0.75))
    n_grid = int(getattr(geom, "n", 80))
    se_on = bool(getattr(geom, "se", True))
    groups, levels = _colour_groups(frame, colour)
    colour_values = frame[colour].to_numpy() if colour else None
    yv = frame[ycol].to_numpy(np.float64)
    lines, bands, starts, at = [], [], [], 0
    for g in range(len(levels)):
        rows = np.flatnonzero(groups == g)
        x, y = axis.values[rows], yv[rows]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if np.unique(x).size < 2:
            continue
        grid = np.linspace(float(x.min()), float(x.max()), n_grid)
        if method == "lm" or np.unique(x).size < 5:
            fit, spread = _lm(x, y, grid, level)
        else:
            fit, spread = _loess(x, y, grid, level, span)
        value = colour_values[rows[0]] if colour else None
        line = pd.DataFrame({"x": axis.out(grid), "y": fit})
        if colour:
            line[colour] = [value] * grid.size
        starts.append([at, grid.size])
        at += grid.size
        lines.append(line)
        if se_on and spread is not None:
            se, df = spread
            t = float(qt(0.5 + level / 2.0, df))
            bands.append(_band(grid, fit - t * se, fit + t * se, value, colour, axis))
    if not lines:
        raise ValueError("geom_smooth() needs at least two distinct x values")
    line_map = {"x": "x", "y": "y", **_colour_mapping(colour)}
    curve = _layer(
        "line", pd.concat(lines, ignore_index=True), line_map, geom,
        _groups=starts, linewidth=float(getattr(geom, "linewidth", 2.0) or 2.0),
    )
    _title(curve, "y", ycol)
    _title(curve, "x", xcol)
    if geom.const_color is None and not colour:
        curve.const_color = "#3366FF"  # ggplot2's smoother blue
    band_layer = _bands_layer(bands, colour, geom, axis, _ALPHA_BAND) if bands else None
    if band_layer is None:
        return curve
    if geom.const_color is None and not colour:
        band_layer.const_color = "#999999"
        band_layer.alpha = geom.alpha if geom.alpha is not None else 0.4
    _title(band_layer, "y", ycol)
    return [band_layer, curve]


# ── summaries ────────────────────────────────────────────────────────────────


def _mean_se(v: np.ndarray, mult: float = 1.0):
    m = float(np.mean(v))
    se = float(np.std(v, ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0
    return m, m - mult * se, m + mult * se


def _mean_cl_normal(v: np.ndarray, level: float = 0.95):
    m = float(np.mean(v))
    if v.size < 2:
        return m, m, m
    se = float(np.std(v, ddof=1) / math.sqrt(v.size))
    t = float(qt(0.5 + level / 2.0, v.size - 1))
    return m, m - t * se, m + t * se


def _mean_sdl(v: np.ndarray, mult: float = 2.0):
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return m, m - mult * sd, m + mult * sd


def _median_hilow(v: np.ndarray, conf: float = 0.95):
    lo, mid, hi = np.quantile(v, [0.5 - conf / 2.0, 0.5, 0.5 + conf / 2.0])
    return float(mid), float(lo), float(hi)


SUMMARIES: dict[str, Callable[..., tuple[float, float, float]]] = {
    "mean_se": _mean_se,
    "mean_cl_normal": _mean_cl_normal,
    "mean_sdl": _mean_sdl,
    "median_hilow": _median_hilow,
}


def summary(geom: _Geom, mapping: dict, data: Any):
    xcol, ycol = mapping.get("x"), mapping.get("y")
    if not xcol or not ycol:
        raise ValueError("stat_summary() requires aes(x=, y=)")
    fun_data = getattr(geom, "fun_data", "mean_se")
    if callable(fun_data):
        fn = fun_data
    else:
        if fun_data not in SUMMARIES:
            raise ValueError(
                f"stat_summary(fun_data=) is one of {sorted(SUMMARIES)}, not {fun_data!r}"
            )
        fn = SUMMARIES[fun_data]
    fun_args = dict(getattr(geom, "fun_args", None) or {})
    colour = mapping.get("color")
    colour = colour if colour and colour not in {xcol, ycol} and has_column(data, colour) else None
    frame = _frame(data, [xcol, ycol, colour])
    keys = [xcol] + ([colour] if colour else [])
    rows = []
    for key, piece in frame.groupby(keys, sort=False, dropna=False, observed=True):
        values = piece[ycol].to_numpy(np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        y, lo, hi = fn(values, **fun_args)
        key = key if isinstance(key, tuple) else (key,)
        row = {xcol: key[0], "__y": y, "__ymin": lo, "__ymax": hi}
        if colour:
            row[colour] = key[1]
        rows.append(row)
    stats = pd.DataFrame(rows)
    if stats.empty:
        raise ValueError("stat_summary() found no numeric y values")
    shape = str(getattr(geom, "geom", "pointrange"))
    stat_map = {"x": xcol, "y": "__y", "ymin": "__ymin", "ymax": "__ymax"}
    if colour:
        stat_map["color"] = colour
    proxy = _Geom(color=geom.const_color, alpha=geom.alpha)
    proxy.position = getattr(geom, "position", None)
    proxy.width = getattr(geom, "width", 0.5)
    proxy.linewidth = getattr(geom, "linewidth", 1.0)
    proxy.size = getattr(geom, "size", None)
    _title(proxy, "y", ycol)
    if shape in {"pointrange", "errorbar", "linerange"}:
        proxy.kind = shape
        return ranges(proxy, stat_map, stats)
    if shape in {"col", "bar"}:
        proxy.kind = "col"
        proxy.width = getattr(geom, "width", 0.9)
        if colour:
            proxy.position = proxy.position or "dodge"
            return positioned_bars(proxy, {"x": xcol, "y": "__y", "color": colour}, stats)
        # One bar per x: the ordinary column path keeps its hover tooltips.
        out = _Geom(aes(x=xcol, y="__y"), color=geom.const_color, alpha=geom.alpha)
        out.kind = "col"
        out.width = proxy.width
        out.data_override = stats
        out._replace_mapping = True
        out.const_color = geom.const_color
        _title(out, "y", ycol)
        return out
    if shape == "point":
        dot = _layer("point", stats.rename(columns={"__y": "y", xcol: "x"}),
                     {"x": "x", "y": "y", **_colour_mapping(colour)}, geom)
        dot.size = getattr(geom, "size", None)
        return dot
    raise ValueError("stat_summary(geom=) is 'pointrange', 'errorbar', 'linerange', 'col', or 'point'")

