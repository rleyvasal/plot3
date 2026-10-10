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
from plot3.scales import col_values, ordered_levels, resolution
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
        # Counted values are categories, in ggplot2's level order (4, 6, 8).
        levels = ["NA" if v is None else str(v) for v in ordered_levels(series.tolist())]
        labels = series.map(lambda v: "NA" if pd.isna(v) else str(v))
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
        # position_dodge2(padding=): a gap between neighbouring bars.
        slot *= 1.0 - float(getattr(getattr(geom, "position", None), "padding", 0.0) or 0.0)
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


def _point_position(geom: _Geom):
    """(kind, settings) for how points move: jitter, jitterdodge, or nudge."""
    pos = getattr(geom, "position", None)
    if geom.kind == "jitter":
        return "jitter", geom
    if isinstance(pos, str):
        kind = pos.strip().lower()
        if kind not in {"identity", "jitter"}:
            raise ValueError(
                f"geom_point(position={pos!r}): use 'identity', 'jitter', position_jitter(), "
                "position_jitterdodge(), or position_nudge()"
            )
        return kind, None
    if pos is None:
        return "identity", None
    return str(getattr(pos, "kind", "identity")), pos


def jitter(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_jitter, and geom_point with position_jitter(),
    position_jitterdodge(), or position_nudge()."""
    name = "geom_jitter" if geom.kind == "jitter" else "geom_point"
    xcol, ycol = mapping.get("x"), mapping.get("y")
    if not xcol or not ycol:
        raise ValueError(f"{name}() requires aes(x=, y=)")
    kind, pos = _point_position(geom)
    dodge = mapping.get("color") or mapping.get("__fillgroup") or mapping.get("group")
    dodge = dodge if dodge and has_column(data, dodge) else None
    keep = [c for c in (mapping.get("color"), mapping.get("size"), mapping.get("group"),
                        mapping.get("shape"), dodge) if c]
    frame = _frame(data, [xcol, ycol, *keep])
    axis = _axis(frame[xcol])
    yaxis = _axis(frame[ycol])
    rng = np.random.default_rng(getattr(pos if pos is not None else geom, "seed", 0))
    xs, ys = axis.values.astype(np.float64), yaxis.values.astype(np.float64)
    if kind == "nudge":
        xs = xs + float(getattr(pos, "x", 0.0))
        ys = ys + float(getattr(pos, "y", 0.0))
    elif kind == "jitterdodge":
        # ggplot2: dodge by group within each x, then jitter inside the
        # group's slot (40% of the spacing, shared among n + 2 slots).
        groups, levels = _colour_groups(frame, dodge)
        n = max(1, len(levels))
        step = axis.step()
        xs = xs + _dodge_offsets(n, step * float(pos.dodge_width))[groups]
        width = 0.4 * step if pos.jitter_width is None else float(pos.jitter_width)
        width = width / (n + 2)
        xs = xs + rng.uniform(-width, width, len(frame))
        if pos.jitter_height > 0:
            ys = ys + rng.uniform(-pos.jitter_height, pos.jitter_height, len(frame))
    else:
        source = pos if pos is not None else geom
        width = getattr(source, "width", None)
        height = getattr(source, "height", None)
        # ggplot2: 40% of the resolution in each direction, both ways.
        width = 0.4 * axis.step() if width is None else float(width) * (axis.step() if axis.kind == "cat" else 1.0)
        height = 0.4 * yaxis.step() if height is None else float(height)
        xs = xs + rng.uniform(-width, width, len(frame))
        if height > 0:
            ys = ys + rng.uniform(-height, height, len(frame))
    out_frame = pd.DataFrame({"x": axis.out(xs), "y": yaxis.out(ys)})
    mapping_out = {"x": "x", "y": "y"}
    for key, col in (("colour", mapping.get("color")), ("size", mapping.get("size")),
                     ("group", mapping.get("group")), ("shape", mapping.get("shape"))):
        if col:
            out_frame[col] = frame[col].to_numpy()
            mapping_out[key] = col
    out = _layer("point", out_frame, mapping_out, geom)
    out.size = getattr(geom, "size", None)
    out.shape = getattr(geom, "shape", None)
    out = _levels_hook(out, axis)
    if yaxis.kind == "cat" and yaxis.levels is not None and kind != "nudge":
        raise ValueError(f"{name}() needs a numeric y to jitter")
    if yaxis.kind == "cat" and yaxis.levels is not None:
        out._y_levels = list(yaxis.levels)
    _title(out, "x", xcol)
    _title(out, "y", ycol)
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
    # Dodge by the colour group, or by a fill the layer is not coloured by.
    dodge = colour or mapping.get("__fillgroup") or mapping.get("group")
    dodge = dodge if dodge and has_column(data, dodge) else None
    frame = _frame(data, [xcol, lo, hi, ycol if need_y else None, colour, dodge])
    return frame, xcol, lo, hi, ycol, colour, dodge


def _dodged_x(geom: _Geom, axis: _Axis, groups: np.ndarray, n_groups: int) -> np.ndarray:
    kind, width = position_kind(getattr(geom, "position", None), "identity")
    if kind != "dodge" or n_groups <= 1:
        return axis.values
    width = 0.9 if width is None else width
    return axis.values + _dodge_offsets(n_groups, axis.step() * width)[groups]


def ranges(geom: _Geom, mapping: dict, data: Any):
    """geom_errorbar, geom_linerange, geom_pointrange."""
    kind = geom.kind
    frame, xcol, lo, hi, ycol, colour, dodge = _range_frame(
        geom, mapping, data, kind == "pointrange"
    )
    axis = _axis(frame[xcol])
    groups, levels = _colour_groups(frame, dodge)
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
    line = _layer(
        "line", line_frame, line_map, geom, _groups=starts, linewidth=linewidth,
        _ink_default=not colour,
    )
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
    dot = _layer("point", dot_frame, dot_map, geom, _ink_default=not colour)
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
        linetype=getattr(geom, "linetype", None),
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
    fill_group = mapping.get("__fillgroup")
    fill_group = (
        fill_group
        if not colour and fill_group and fill_group not in {xcol, ycol} and has_column(data, fill_group)
        else None
    )
    split = colour or fill_group
    frame = _frame(data, [xcol, ycol, split])
    keys = [xcol] + ([split] if split else [])
    rows = []
    for key, piece in frame.groupby(keys, sort=False, dropna=False, observed=True):
        values = piece[ycol].to_numpy(np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        y, lo, hi = fn(values, **fun_args)
        key = key if isinstance(key, tuple) else (key,)
        row = {xcol: key[0], "__y": y, "__ymin": lo, "__ymax": hi}
        if split:
            row[split] = key[1]
        rows.append(row)
    stats = pd.DataFrame(rows)
    if stats.empty:
        raise ValueError("stat_summary() found no numeric y values")
    shape = str(getattr(geom, "geom", "pointrange"))
    stat_map = {"x": xcol, "y": "__y", "ymin": "__ymin", "ymax": "__ymax"}
    if colour:
        stat_map["color"] = colour
    elif fill_group:
        stat_map["__fillgroup"] = fill_group
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
        if split:
            # Bars are filled shapes: the fill group colours and dodges them.
            proxy.position = proxy.position or "dodge"
            return positioned_bars(proxy, {"x": xcol, "y": "__y", "color": split}, stats)
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



# ── tiles, areas, steps, segments, rectangles ────────────────────────────────


def _rect_rows(lefts, rights, bottoms, tops, keep_cols: dict):
    """Rectangles as 4-point polygons (left edge up, right edge down)."""
    xs, ys, groups = [], [], []
    extra = {name: [] for name in keep_cols}
    for i in range(len(lefts)):
        values = (lefts[i], rights[i], bottoms[i], tops[i])
        if not all(np.isfinite(values)):
            continue
        groups.append([len(xs), 4])
        xs += [lefts[i], lefts[i], rights[i], rights[i]]
        ys += [bottoms[i], tops[i], tops[i], bottoms[i]]
        for name, column in keep_cols.items():
            extra[name] += [column[i]] * 4
    return xs, ys, groups, extra


def tile(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_tile / geom_raster: a heatmap cell per row."""
    xcol, ycol = mapping.get("x"), mapping.get("y")
    if not xcol or not ycol:
        raise ValueError("geom_tile() requires aes(x=, y=) and usually fill=")
    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    frame = _frame(data, [xcol, ycol, colour])
    xaxis, yaxis = _axis(frame[xcol]), _axis(frame[ycol])
    width = getattr(geom, "width", None)
    height = getattr(geom, "height", None)
    w = xaxis.step() if width is None else float(width) * (1.0 if xaxis.kind != "cat" else 1.0)
    h = yaxis.step() if height is None else float(height)
    xs, ys = xaxis.values, yaxis.values
    keep = {colour: frame[colour].tolist()} if colour else {}
    rx, ry, groups, extra = _rect_rows(xs - w / 2, xs + w / 2, ys - h / 2, ys + h / 2, keep)
    out_frame = pd.DataFrame({"x": xaxis.out(np.asarray(rx)), "y": yaxis.out(np.asarray(ry))})
    out_map = {"x": "x", "y": "y"}
    if colour:
        out_frame[colour] = extra[colour]
        out_map["colour"] = colour
    alpha = geom.alpha if geom.alpha is not None else 1.0
    # Opaque cells get a hairline in their own colour: neighbours then meet
    # without the pale antialiased seam between them.
    out = _layer("poly", out_frame, out_map, geom, _groups=groups,
                 linewidth=0.6 if alpha >= 0.9 else 0.01)
    out.alpha = alpha
    _title(out, "x", xcol)
    _title(out, "y", ycol)
    _levels_hook(out, xaxis)
    if yaxis.kind == "cat" and yaxis.levels is not None:
        out._y_levels = list(yaxis.levels)
    return out


def area(geom: _Geom, mapping: dict, data: Any):
    """geom_area: filled from 0, groups stacked on a shared x grid (ggplot2)."""
    xcol, ycol = mapping.get("x"), mapping.get("y")
    if not xcol or not ycol:
        raise ValueError("geom_area() requires aes(x=, y=)")
    colour = mapping.get("color")
    colour = colour if colour and colour not in {xcol, ycol} and has_column(data, colour) else None
    frame = _frame(data, [xcol, ycol, colour])
    axis = _axis(frame[xcol])
    if axis.kind == "cat":
        raise ValueError("geom_area() needs a numeric or date x")
    groups, levels = _colour_groups(frame, colour)
    kind, _w = position_kind(getattr(geom, "position", None), "stack")
    grid = np.unique(axis.values[np.isfinite(axis.values)])
    yv = frame[ycol].to_numpy(np.float64)
    series = []
    for g in range(len(levels)):
        rows = np.flatnonzero(groups == g)
        order = np.argsort(axis.values[rows])
        gx, gy = axis.values[rows][order], yv[rows][order]
        ok = np.isfinite(gx) & np.isfinite(gy)
        gx, gy = gx[ok], gy[ok]
        # Outside a group's own x range it adds nothing to the stack.
        series.append(np.interp(grid, gx, gy, left=0.0, right=0.0) if gx.size else np.zeros_like(grid))
    bottoms = [np.zeros_like(grid) for _ in series]
    tops = [s.copy() for s in series]
    if kind in {"stack", "fill"} and len(series) > 1:
        running = np.zeros_like(grid)
        for g in reversed(range(len(series))):  # first level on top
            bottoms[g] = running.copy()
            running = running + series[g]
            tops[g] = running.copy()
        if kind == "fill":
            total = np.where(running > 0, running, 1.0)
            bottoms = [b / total for b in bottoms]
            tops = [t / total for t in tops]
    colour_values = frame[colour].to_numpy() if colour else None
    pieces = []
    for g in range(len(series)):
        rows = np.flatnonzero(groups == g)
        value = colour_values[rows[0]] if colour and rows.size else None
        pieces.append(_band(grid, bottoms[g], tops[g], value, colour, axis))
    out = _bands_layer(pieces, colour, geom, axis, 0.85)
    if out is None:
        raise ValueError("geom_area() needs at least two x values")
    out._baseline_zero = True
    _title(out, "x", xcol)
    _title(out, "y", "proportion" if kind == "fill" else ycol)
    return out


def step(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_step: horizontal then vertical ("hv"), "vh", or "mid"."""
    xcol, ycol = mapping.get("x"), mapping.get("y")
    if not xcol or not ycol:
        raise ValueError("geom_step() requires aes(x=, y=)")
    direction = str(getattr(geom, "direction", "hv"))
    if direction not in {"hv", "vh", "mid"}:
        raise ValueError("geom_step(direction=) is 'hv', 'vh', or 'mid'")
    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    group = mapping.get("group")
    split = colour or (group if group and has_column(data, group) else None)
    frame = _frame(data, [xcol, ycol, split])
    axis = _axis(frame[xcol])
    groups, levels = _colour_groups(frame, split)
    xs_all, ys_all = axis.values, frame[ycol].to_numpy(np.float64)
    out_x, out_y, out_c, starts = [], [], [], []
    for g in range(len(levels)):
        rows = np.flatnonzero(groups == g)
        order = rows[np.argsort(xs_all[rows], kind="stable")]
        px, py = _step_path(xs_all[order], ys_all[order], direction)
        if len(px) < 2:
            continue
        starts.append([len(out_x), len(px)])
        out_x += px
        out_y += py
        if colour:
            out_c += [frame[colour].iloc[order[0]]] * len(px)
    frame_out = pd.DataFrame({"x": axis.out(np.asarray(out_x)), "y": out_y})
    out_map = {"x": "x", "y": "y"}
    if colour:
        frame_out[colour] = out_c
        out_map["colour"] = colour
    out = _layer("line", frame_out, out_map, geom, _groups=starts,
                 linewidth=float(getattr(geom, "linewidth", 2.0) or 2.0))
    out.linetype = getattr(geom, "linetype", None)
    _title(out, "x", xcol)
    _title(out, "y", getattr(geom, "_y_name", None) or ycol)
    _levels_hook(out, axis)
    return out


def _step_path(xs, ys, direction):
    px, py = [], []
    for i in range(len(xs)):
        if i == 0:
            px.append(float(xs[0]))
            py.append(float(ys[0]))
            continue
        x0, y0, x1, y1 = float(xs[i - 1]), float(ys[i - 1]), float(xs[i]), float(ys[i])
        if direction == "hv":
            px += [x1, x1]
            py += [y0, y1]
        elif direction == "vh":
            px += [x0, x1]
            py += [y1, y1]
        else:
            mid = (x0 + x1) / 2
            px += [mid, mid, x1]
            py += [y0, y1, y1]
    return px, py


def segment(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_segment: a line from (x, y) to (xend, yend) for every row."""
    need = ["x", "y", "xend", "yend"]
    if any(not mapping.get(k) for k in need):
        raise ValueError("geom_segment() requires aes(x=, y=, xend=, yend=)")
    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    frame = _frame(data, [mapping[k] for k in need] + [colour])
    cols = [frame[mapping[k]].to_numpy(np.float64) for k in need]
    out_x, out_y, out_c, starts = [], [], [], []
    for i in range(len(frame)):
        x, y, xe, ye = (c[i] for c in cols)
        if not all(np.isfinite([x, y, xe, ye])):
            continue
        starts.append([len(out_x), 2])
        out_x += [x, xe]
        out_y += [y, ye]
        if colour:
            out_c += [frame[colour].iloc[i]] * 2
    frame_out = pd.DataFrame({"x": out_x, "y": out_y})
    out_map = {"x": "x", "y": "y"}
    if colour:
        frame_out[colour] = out_c
        out_map["colour"] = colour
    out = _layer("line", frame_out, out_map, geom, _groups=starts,
                 linewidth=float(getattr(geom, "linewidth", 1.0) or 1.0), _ink_default=not colour)
    out.linetype = getattr(geom, "linetype", None)
    out.arrow = getattr(geom, "arrow", None)
    _title(out, "x", mapping["x"])
    _title(out, "y", mapping["y"])
    return out


def rect(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_rect: a rectangle from xmin..xmax and ymin..ymax for every row."""
    need = ["xmin", "xmax", "ymin", "ymax"]
    if any(not mapping.get(k) for k in need):
        raise ValueError("geom_rect() requires aes(xmin=, xmax=, ymin=, ymax=)")
    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    frame = _frame(data, [mapping[k] for k in need] + [colour])
    x0, x1, y0, y1 = (frame[mapping[k]].to_numpy(np.float64) for k in need)
    keep = {colour: frame[colour].tolist()} if colour else {}
    rx, ry, groups, extra = _rect_rows(x0, x1, y0, y1, keep)
    out_frame = pd.DataFrame({"x": rx, "y": ry})
    out_map = {"x": "x", "y": "y"}
    if colour:
        out_frame[colour] = extra[colour]
        out_map["colour"] = colour
    out = _layer("poly", out_frame, out_map, geom, _groups=groups, linewidth=0.01)
    out.alpha = geom.alpha if geom.alpha is not None else 0.6
    _title(out, "x", mapping["xmin"])
    _title(out, "y", mapping["ymin"])
    return out


# ── distributions: Q-Q and ECDF ──────────────────────────────────────────────


def _ppoints(n: int) -> np.ndarray:
    """R's ppoints: (i - a) / (n + 1 - 2a), a = 3/8 up to 10 points, else 1/2."""
    a = 3.0 / 8.0 if n <= 10 else 0.5
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)


def _sample_groups(geom, mapping, data, who):
    sample = mapping.get("sample") or mapping.get("y")
    if not sample:
        raise ValueError(f"{who}() requires aes(sample=)")
    colour = mapping.get("color")
    colour = colour if colour and colour != sample and has_column(data, colour) else None
    frame = _frame(data, [sample, colour])
    groups, levels = _colour_groups(frame, colour)
    values = frame[sample].to_numpy(np.float64)
    out = []
    for g in range(len(levels)):
        rows = np.flatnonzero(groups == g)
        v = np.sort(values[rows][np.isfinite(values[rows])])
        label = frame[colour].iloc[rows[0]] if colour and rows.size else None
        if v.size:
            out.append((v, label))
    return sample, colour, out


def qq(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_qq / stat_qq: sample quantiles against the normal distribution."""
    from plot3.special import qnorm

    sample, colour, parts = _sample_groups(geom, mapping, data, "geom_qq")
    xs, ys, cs = [], [], []
    for values, label in parts:
        theory = np.asarray(qnorm(_ppoints(values.size)), dtype=np.float64)
        xs += theory.tolist()
        ys += values.tolist()
        cs += [label] * values.size
    frame = pd.DataFrame({"x": xs, "y": ys})
    out_map = {"x": "x", "y": "y"}
    if colour:
        frame[colour] = cs
        out_map["colour"] = colour
    out = _layer("point", frame, out_map, geom)
    out.size = getattr(geom, "size", None)
    _title(out, "x", "theoretical")
    _title(out, "y", "sample")
    return out


def qq_line(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_qq_line: the line through the first and third quartiles."""
    from plot3.special import qnorm

    sample, colour, parts = _sample_groups(geom, mapping, data, "geom_qq_line")
    xs, ys, cs, starts = [], [], [], []
    q_theory = np.asarray(qnorm(np.array([0.25, 0.75])), dtype=np.float64)
    for values, label in parts:
        if values.size < 2:
            continue
        q_sample = np.quantile(values, [0.25, 0.75])  # R's quantile type 7
        slope = (q_sample[1] - q_sample[0]) / (q_theory[1] - q_theory[0])
        intercept = q_sample[0] - slope * q_theory[0]
        theory = np.asarray(qnorm(_ppoints(values.size)), dtype=np.float64)
        ends = np.array([theory.min(), theory.max()])
        starts.append([len(xs), 2])
        xs += ends.tolist()
        ys += (intercept + slope * ends).tolist()
        cs += [label] * 2
    frame = pd.DataFrame({"x": xs, "y": ys})
    out_map = {"x": "x", "y": "y"}
    if colour:
        frame[colour] = cs
        out_map["colour"] = colour
    out = _layer("line", frame, out_map, geom, _groups=starts,
                 linewidth=float(getattr(geom, "linewidth", 1.5) or 1.5), _ink_default=not colour)
    out.linetype = getattr(geom, "linetype", None)
    _title(out, "x", "theoretical")
    _title(out, "y", "sample")
    return out


def ecdf(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """stat_ecdf: the empirical cumulative distribution as a step line."""
    xcol = mapping.get("x")
    if not xcol:
        raise ValueError("stat_ecdf() requires aes(x=)")
    colour = mapping.get("color")
    colour = colour if colour and colour != xcol and has_column(data, colour) else None
    frame = _frame(data, [xcol, colour])
    groups, levels = _colour_groups(frame, colour)
    values = frame[xcol].to_numpy(np.float64)
    finite = values[np.isfinite(values)]
    span = float(finite.max() - finite.min()) if finite.size else 1.0
    pad = 0.04 * (span or 1.0) if getattr(geom, "pad", True) else 0.0
    rows_out = []
    for g in range(len(levels)):
        rows = np.flatnonzero(groups == g)
        v = np.sort(values[rows][np.isfinite(values[rows])])
        if v.size == 0:
            continue
        # F(x) at each distinct value: ties make one step, not several.
        distinct = np.unique(v)
        heights = np.searchsorted(v, distinct, side="right") / v.size
        xs = np.concatenate([[distinct[0] - pad], distinct, [distinct[-1] + pad]])
        ys = np.concatenate([[0.0], heights, [1.0]])
        label = frame[colour].iloc[rows[0]] if colour else None
        for x, y in zip(xs, ys):
            rows_out.append({"__x": x, "__y": y, **({colour: label} if colour else {})})
    table = pd.DataFrame(rows_out)
    step_geom = _Geom(color=geom.const_color, alpha=geom.alpha)
    step_geom.kind = "step"
    step_geom.direction = "hv"
    step_geom.linewidth = getattr(geom, "linewidth", 2.0)
    step_geom._y_name = "ECDF"
    out = step(step_geom, {"x": "__x", "y": "__y", **({"color": colour} if colour else {})}, table)
    _title(out, "x", xcol)
    _title(out, "y", "ECDF")
    return out


# ── crossbars, horizontal error bars, polygons ───────────────────────────────


def crossbar(geom: _Geom, mapping: dict, data: Any):
    """geom_crossbar: a box from ymin to ymax and a thick line at y."""
    frame, xcol, lo, hi, ycol, colour, dodge = _range_frame(geom, mapping, data, True)
    fill = mapping.get("__fillgroup")
    fill = fill if fill and has_column(data, fill) and fill != colour else None
    if fill and fill not in frame.columns:
        frame = _frame(data, [xcol, lo, hi, ycol, colour, fill])
    dodge = dodge or fill
    axis = _axis(frame[xcol])
    groups, levels = _colour_groups(frame, dodge)
    xs = _dodged_x(geom, axis, groups, len(levels))
    dodge_kind, _w = position_kind(getattr(geom, "position", None), "identity")
    slot = axis.step() * (1.0 / len(levels) if dodge_kind == "dodge" else 1.0)
    half = 0.5 * slot * float(getattr(geom, "width", 0.9))
    y_lo, y_hi = frame[lo].to_numpy(np.float64), frame[hi].to_numpy(np.float64)
    y_mid = frame[ycol].to_numpy(np.float64)
    box_x, box_y, box_c, box_starts = [], [], [], []
    mid_x, mid_y, mid_c, mid_starts = [], [], [], []
    tag = colour or fill
    tags = frame[tag].tolist() if tag else None
    for i in range(len(frame)):
        x, a, b, m = xs[i], y_lo[i], y_hi[i], y_mid[i]
        if not all(math.isfinite(v) for v in (x, a, b, m)):
            continue
        box_starts.append([len(box_x), 5])
        box_x += [x - half, x + half, x + half, x - half, x - half]
        box_y += [a, a, b, b, a]
        mid_starts.append([len(mid_x), 2])
        mid_x += [x - half, x + half]
        mid_y += [m, m]
        if tags is not None:
            box_c += [tags[i]] * 5
            mid_c += [tags[i]] * 2
    layers = []
    linewidth = float(getattr(geom, "linewidth", 1.0) or 1.0)
    if fill:
        # Filled boxes under dark lines, as ggplot2 draws crossbars.
        fill_frame = pd.DataFrame({"x": axis.out(np.asarray(box_x)), "y": box_y, fill: box_c})
        body = _layer("poly", fill_frame, {"x": "x", "y": "y", "colour": fill}, geom,
                      _groups=box_starts, linewidth=0.0, _polygon=True)
        body.alpha = geom.alpha if geom.alpha is not None else 1.0
        layers.append(_levels_hook(body, axis))
    line_colour = colour if colour else None
    for xs_part, ys_part, cs_part, starts, width in (
        (box_x, box_y, box_c, box_starts, linewidth),
        (mid_x, mid_y, mid_c, mid_starts, linewidth * float(getattr(geom, "fatten", 2.5))),
    ):
        part = pd.DataFrame({"x": axis.out(np.asarray(xs_part)), "y": ys_part})
        part_map = {"x": "x", "y": "y"}
        if line_colour:
            part[line_colour] = cs_part
            part_map["colour"] = line_colour
        line = _layer("line", part, part_map, geom, _groups=starts, linewidth=width,
                      _ink_default=not line_colour)
        if not line_colour:
            line.const_color = geom.const_color
        _title(line, "y", ycol)
        layers.append(_levels_hook(line, axis))
    return layers


def errorbarh(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_errorbarh: from xmin to xmax at y, with caps up and down."""
    ycol, lo, hi = mapping.get("y"), mapping.get("xmin"), mapping.get("xmax")
    if not ycol or not lo or not hi:
        raise ValueError("geom_errorbarh() requires aes(y=, xmin=, xmax=)")
    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    frame = _frame(data, [ycol, lo, hi, colour])
    yaxis = _axis(frame[ycol])
    cap = 0.5 * yaxis.step() * float(getattr(geom, "height", 0.5))
    x_lo, x_hi = frame[lo].to_numpy(np.float64), frame[hi].to_numpy(np.float64)
    ys = yaxis.values.astype(np.float64)
    rows_x, rows_y, rows_c, starts = [], [], [], []
    tags = frame[colour].tolist() if colour else None
    for i in range(len(frame)):
        y, a, b = ys[i], x_lo[i], x_hi[i]
        if not all(math.isfinite(v) for v in (y, a, b)):
            continue
        starts.append([len(rows_x), 6])
        rows_x += [a, a, a, b, b, b]
        rows_y += [y - cap, y + cap, y, y, y - cap, y + cap]
        if tags is not None:
            rows_c += [tags[i]] * 6
    out_frame = pd.DataFrame({"x": rows_x, "y": yaxis.out(np.asarray(rows_y))})
    out_map = {"x": "x", "y": "y"}
    if colour:
        out_frame[colour] = rows_c
        out_map["colour"] = colour
    out = _layer("line", out_frame, out_map, geom, _groups=starts,
                 linewidth=float(getattr(geom, "linewidth", 1.0) or 1.0), _ink_default=not colour)
    if yaxis.kind == "cat" and yaxis.levels is not None:
        out._y_levels = list(yaxis.levels)
    _title(out, "x", lo)
    _title(out, "y", ycol)
    return out


def polygon(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_polygon: one closed shape per group, corners in row order."""
    xcol, ycol = mapping.get("x"), mapping.get("y")
    if not xcol or not ycol:
        raise ValueError("geom_polygon() requires aes(x=, y=)")
    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    group = mapping.get("group")
    group = group if group and has_column(data, group) else None
    frame = _frame(data, [xcol, ycol, colour, group]).reset_index(drop=True)
    key_cols = [c for c in (group, colour) if c]
    if key_cols:
        keys = frame[key_cols].astype(str).agg("\x1f".join, axis=1)
        order = pd.unique(keys)
        rank = {k: i for i, k in enumerate(order)}
        frame = frame.iloc[np.argsort(keys.map(rank).to_numpy(), kind="stable")].reset_index(drop=True)
        keys = frame[key_cols].astype(str).agg("\x1f".join, axis=1).to_numpy()
        cut = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        bounds = np.concatenate([[0], cut, [len(frame)]])
    else:
        bounds = np.array([0, len(frame)])
    starts = [[int(a), int(b - a)] for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 3]
    out_frame = pd.DataFrame({"x": frame[xcol].to_numpy(np.float64), "y": frame[ycol].to_numpy(np.float64)})
    out_map = {"x": "x", "y": "y"}
    if colour:
        out_frame[colour] = frame[colour].to_numpy()
        out_map["colour"] = colour
    out = _layer("poly", out_frame, out_map, geom, _groups=starts,
                 linewidth=float(getattr(geom, "linewidth", 0.5)), _polygon=True)
    out.alpha = geom.alpha if geom.alpha is not None else 1.0
    _title(out, "x", xcol)
    _title(out, "y", ycol)
    return out


# ── 2D distributions: bins, hexagons, densities, contours, ellipses ──────────


def _xy_numeric(geom: _Geom, mapping: dict, data: Any, name: str, extra=()):
    xcol, ycol = mapping.get("x"), mapping.get("y")
    if not xcol or not ycol:
        raise ValueError(f"{name}() requires aes(x=, y=)")
    frame = _frame(data, [xcol, ycol, *[c for c in extra if c]])
    xs = frame[xcol].to_numpy(np.float64)
    ys = frame[ycol].to_numpy(np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        raise ValueError(f"{name}() needs numeric x and y")
    return frame.loc[ok].reset_index(drop=True), xcol, ycol, xs[ok], ys[ok]


def _bin_edges(values: np.ndarray, bins, binwidth) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    if binwidth is not None:
        w = float(binwidth)
        start = math.floor(lo / w) * w
        count = max(1, int(math.ceil((hi - start) / w + 1e-9)))
        return start + w * np.arange(count + 1)
    return np.linspace(lo, hi, int(bins) + 1)


def bin_2d(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_bin_2d: rows counted in rectangles, drawn as tiles by count."""
    frame, xcol, ycol, xs, ys = _xy_numeric(geom, mapping, data, "geom_bin_2d")
    bins = getattr(geom, "bins", 30)
    bw = getattr(geom, "binwidth", None)
    bx, by = (bins, bins) if not isinstance(bins, (tuple, list)) else bins
    wx, wy = (bw, bw) if not isinstance(bw, (tuple, list)) else bw
    ex, ey = _bin_edges(xs, bx, wx), _bin_edges(ys, by, wy)
    counts, _, _ = np.histogram2d(xs, ys, bins=[ex, ey])
    cx, cy = 0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:])
    ii, jj = np.nonzero(counts)
    cells = pd.DataFrame({"x": cx[ii], "y": cy[jj], "count": counts[ii, jj]})
    proxy = _Geom(alpha=geom.alpha)
    proxy.kind = "tile"
    proxy.width = float(np.median(np.diff(ex)))
    proxy.height = float(np.median(np.diff(ey)))
    out = tile(proxy, {"x": "x", "y": "y", "color": "count"}, cells)
    out._colour_title = "count"
    _title(out, "x", xcol)
    _title(out, "y", ycol)
    return out


def hex_bins(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_hex: rows counted in hexagons (pointy side up), filled by count."""
    frame, xcol, ycol, xs, ys = _xy_numeric(geom, mapping, data, "geom_hex")
    bins = getattr(geom, "bins", 30)
    bw = getattr(geom, "binwidth", None)
    if bw is not None:
        w, h = (bw, bw) if not isinstance(bw, (tuple, list)) else (float(bw[0]), float(bw[1]))
    else:
        bx, by = (bins, bins) if not isinstance(bins, (tuple, list)) else bins
        w = (xs.max() - xs.min()) / bx or 1.0
        h = (ys.max() - ys.min()) / by or 1.0
    # In (u, v), with u = x / w and v = y / h * sqrt(3)/2, the hexagons are
    # regular, so the nearest centre is the hexagon a point falls in.
    u = (xs - xs.min()) / w
    v = (ys - ys.min()) / h
    k = math.sqrt(3.0) / 2.0
    row0 = np.floor(v)
    best_r = np.zeros_like(u)
    best_c = np.zeros_like(u)
    best_d = np.full(u.shape, np.inf)
    for dr in (0.0, 1.0):
        r = row0 + dr
        offset = np.where(np.mod(r, 2) == 1, 0.5, 0.0)
        c = np.round(u - offset)
        d = (u - (c + offset)) ** 2 + ((v - r) * k) ** 2
        better = d < best_d
        best_r, best_c, best_d = np.where(better, r, best_r), np.where(better, c, best_c), np.where(better, d, best_d)
    keys, counts = np.unique(np.stack([best_r, best_c], axis=1), axis=0, return_counts=True)
    rows = []
    for (r, c), n in zip(keys, counts):
        cx = xs.min() + (c + (0.5 if int(r) % 2 else 0.0)) * w
        cy = ys.min() + r * h
        for dx, dy in ((0, 2 / 3), (0.5, 1 / 3), (0.5, -1 / 3), (0, -2 / 3), (-0.5, -1 / 3), (-0.5, 1 / 3)):
            rows.append((cx + dx * w, cy + dy * h, f"{int(r)}:{int(c)}", float(n)))
    cells = pd.DataFrame(rows, columns=["x", "y", "hex", "count"])
    proxy = _Geom(alpha=geom.alpha)
    proxy.kind = "polygon"
    proxy.linewidth = 0.3
    out = polygon(proxy, {"x": "x", "y": "y", "group": "hex", "color": "count"}, cells)
    out._colour_title = "count"
    _title(out, "x", xcol)
    _title(out, "y", ycol)
    return out


def _bandwidth(values: np.ndarray) -> float:
    """MASS::bandwidth.nrd / 4: the normal kernel's sd, as kde2d uses it."""
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    q75, q25 = np.percentile(values, [75, 25])
    spread = min(sd, (q75 - q25) / 1.34) or sd or 1.0
    return 1.06 * spread * values.size ** (-0.2)


def _kde2d(xs, ys, n, lims=None, h=None):
    """Gaussian kernel density on an n x n grid (MASS::kde2d)."""
    x0, x1 = (xs.min(), xs.max()) if lims is None else lims[:2]
    y0, y1 = (ys.min(), ys.max()) if lims is None else lims[2:]
    gx, gy = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    hx, hy = (_bandwidth(xs), _bandwidth(ys)) if h is None else (float(h[0]) / 4, float(h[1]) / 4)
    kx = np.exp(-0.5 * ((gx[:, None] - xs[None, :]) / hx) ** 2) / (hx * math.sqrt(2 * math.pi))
    ky = np.exp(-0.5 * ((gy[:, None] - ys[None, :]) / hy) ** 2) / (hy * math.sqrt(2 * math.pi))
    z = (ky @ kx.T) / xs.size  # (ny, nx), the shape the contour code takes
    return gx, gy, z


def _levels(z: np.ndarray, bins, breaks) -> list[float]:
    if breaks is not None:
        return [float(b) for b in breaks]
    from plot3.scales import nice_ticks

    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    return [t for t in nice_ticks(lo, hi, int(bins or 10)) if lo < t < hi]


def _contour_layer(geom, pieces, colour_col, xcol, ycol, default_colour):
    from plot3.contour import _contour_lines

    rows, starts, values = [], [], []
    for gx, gy, z, level_value in pieces:
        for level in _levels(z, getattr(geom, "bins", None), getattr(geom, "breaks", None)):
            for line in _contour_lines(gx, gy, z, level):
                if len(line) < 2:
                    continue
                starts.append([len(rows), len(line)])
                rows.extend((x, y) for x, y in line)
                values.extend([level_value] * len(line))
    if not rows:
        raise ValueError("no contour lines: the surface is flat")
    frame = pd.DataFrame(rows, columns=["x", "y"])
    mapping = {"x": "x", "y": "y"}
    if colour_col:
        frame[colour_col] = values
        mapping["colour"] = colour_col
    out = _layer("line", frame, mapping, geom, _groups=starts,
                 linewidth=float(getattr(geom, "linewidth", 1.0) or 1.0))
    if not colour_col and geom.const_color is None:
        out.const_color = default_colour
    _title(out, "x", xcol)
    _title(out, "y", ycol)
    return out


def density_2d(geom: _Geom, mapping: dict, data: Any):
    """geom_density_2d: contour lines of a 2D kernel density, per group."""
    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    frame, xcol, ycol, xs, ys = _xy_numeric(geom, mapping, data, "geom_density_2d", [colour])
    n = int(getattr(geom, "n", 100))
    lims = (xs.min(), xs.max(), ys.min(), ys.max())
    groups, levels = _colour_groups(frame, colour)
    if getattr(geom, "kind", "") == "density_2d_filled":
        return _density_bands(geom, xs, ys, n, lims, xcol, ycol)
    pieces = []
    for g, level in enumerate(levels):
        pick = groups == g
        if pick.sum() < 3:
            continue
        gx, gy, z = _kde2d(xs[pick], ys[pick], n, lims, getattr(geom, "h", None))
        pieces.append((gx, gy, z, level))
    return _contour_layer(geom, pieces, colour if levels != [None] else None, xcol, ycol, "#3366FF")


def _density_bands(geom, xs, ys, n, lims, xcol, ycol):
    """geom_density_2d_filled: the density in bands between contour levels,
    one colour per band (viridis, as ggplot2), drawn as fine cells."""
    from plot3.scales import fmt_ticks

    n = min(n, 60)
    gx, gy, z = _kde2d(xs, ys, n, lims, getattr(geom, "h", None))
    top = float(z.max())
    # Fewer bands than lines by default, so the legend stays beside the panel.
    inner = _levels(z, getattr(geom, "bins", None) or 6, getattr(geom, "breaks", None))
    edges = [0.0] + [v for v in inner if v > 0] + [top]
    band = np.clip(np.searchsorted(edges, z, side="left") - 1, 0, len(edges) - 2)
    labels = fmt_ticks(edges[:-1]) + [f"{top:.3g}"]
    names = [f"({labels[i]}, {labels[i + 1]}]" for i in range(len(edges) - 1)]
    X, Y = np.meshgrid(gx, gy)
    cells = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "level": [names[b] for b in band.ravel()]})
    proxy = _Geom(alpha=geom.alpha)
    proxy.kind = "tile"
    proxy.width = float(gx[1] - gx[0])
    proxy.height = float(gy[1] - gy[0])
    out = tile(proxy, {"x": "x", "y": "y", "color": "level"}, cells)
    # Bands in order, lowest first, whatever their labels sort as.
    out.data_override["level"] = pd.Categorical(out.data_override["level"], categories=names)
    out._default_discrete = "viridis"
    out._colour_title = "level"
    _title(out, "x", xcol)
    _title(out, "y", ycol)
    return out


def contour(geom: _Geom, mapping: dict, data: Any):
    """geom_contour: contour lines of z on a regular x-y grid."""
    xcol, ycol, zcol = mapping.get("x"), mapping.get("y"), mapping.get("z")
    if not xcol or not ycol or not zcol:
        raise ValueError("geom_contour() requires aes(x=, y=, z=) on gridded data")
    frame = _frame(data, [xcol, ycol, zcol])
    gx = np.unique(frame[xcol].to_numpy(np.float64))
    gy = np.unique(frame[ycol].to_numpy(np.float64))
    if gx.size < 2 or gy.size < 2:
        raise ValueError("geom_contour() needs a grid: several x and several y values")
    z = np.full((gy.size, gx.size), np.nan)
    ix = np.searchsorted(gx, frame[xcol].to_numpy(np.float64))
    iy = np.searchsorted(gy, frame[ycol].to_numpy(np.float64))
    z[iy, ix] = frame[zcol].to_numpy(np.float64)
    out = _contour_layer(geom, [(gx, gy, z, None)], None, xcol, ycol, "#3366FF")
    out._contour_z = zcol
    return out


def _robust_t(points: np.ndarray, nu: float = 5.0, iterations: int = 100):
    """Centre and scatter under a multivariate t (MASS::cov.trob)."""
    centre = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    p = points.shape[1]
    for _ in range(iterations):
        diff = points - centre
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            break
        d2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
        w = (nu + p) / (nu + d2)
        new_centre = (w[:, None] * points).sum(axis=0) / w.sum()
        diff = points - new_centre
        new_cov = (w[:, None] * diff).T @ diff / len(points)
        done = np.allclose(new_centre, centre, rtol=1e-8) and np.allclose(new_cov, cov, rtol=1e-8)
        centre, cov = new_centre, new_cov
        if done:
            break
    return centre, cov


def ellipse(geom: _Geom, mapping: dict, data: Any):
    """stat_ellipse: a confidence ellipse per group (ggplot2's stat_ellipse).

    ``type="t"`` (default) uses a robust multivariate t fit, ``"norm"`` the
    sample covariance, and ``"euclid"`` a circle of radius ``level``.
    """
    from plot3.special import _qf1

    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    group = mapping.get("group")
    group = group if group and has_column(data, group) else None
    frame, xcol, ycol, xs, ys = _xy_numeric(geom, mapping, data, "stat_ellipse", [colour, group])
    groups, levels = _colour_groups(frame, colour or group)
    level = float(getattr(geom, "level", 0.95))
    kind = getattr(geom, "type", "t")
    segments = int(getattr(geom, "segments", 51))
    angles = np.linspace(0, 2 * np.pi, segments)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    rows, starts, tags = [], [], []
    for g, lev in enumerate(levels):
        pts = np.column_stack([xs[groups == g], ys[groups == g]])
        if len(pts) < 3:
            continue
        if kind == "euclid":
            centre, shape, radius = pts.mean(axis=0), np.eye(2), level
        else:
            centre, shape = _robust_t(pts) if kind == "t" else (pts.mean(axis=0), np.cov(pts, rowvar=False))
            radius = math.sqrt(2.0 * _qf1(level, 2.0, len(pts) - 1.0))
        try:
            chol = np.linalg.cholesky(shape)
        except np.linalg.LinAlgError:
            continue
        curve = centre + radius * unit @ chol.T
        starts.append([len(rows), len(curve)])
        rows.extend(map(tuple, curve))
        tags.extend([lev] * len(curve))
    if not rows:
        raise ValueError("stat_ellipse() needs at least 3 points per group")
    out_frame = pd.DataFrame(rows, columns=["x", "y"])
    out_map = {"x": "x", "y": "y"}
    if colour:
        out_frame[colour] = tags
        out_map["colour"] = colour
    out = _layer("line", out_frame, out_map, geom, _groups=starts,
                 linewidth=float(getattr(geom, "linewidth", 1.0) or 1.0), _ink_default=not colour)
    _title(out, "x", xcol)
    _title(out, "y", ycol)
    return out


def count_points(geom: _Geom, mapping: dict, data: Any) -> _Geom:
    """geom_count: one point per distinct (x, y), sized by how many rows
    share it (ggplot2's stat_sum, ``n``)."""
    xcol, ycol = mapping.get("x"), mapping.get("y")
    if not xcol or not ycol:
        raise ValueError("geom_count() requires aes(x=, y=)")
    colour = mapping.get("color")
    colour = colour if colour and colour not in {xcol, ycol} and has_column(data, colour) else None
    keys = [xcol, ycol] + ([colour] if colour else [])
    frame = _frame(data, keys)
    counted = frame.groupby(keys, sort=False, observed=True).size().reset_index(name="n")
    out_map = {"x": xcol, "y": ycol, "size": "n", **_colour_mapping(colour)}
    out = _layer("point", counted, out_map, geom)
    out.size = None
    out.shape = getattr(geom, "shape", None)
    _title(out, "x", xcol)
    _title(out, "y", ycol)
    return out
