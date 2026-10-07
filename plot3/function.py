"""Sample a :class:`plot3.expr.Formula` into a drawable line or surface.

``geom_function`` stays a thin parameter holder. At build time this module
evaluates it and returns a ``geom_line``, ``geom_path``, or ``geom_surface``
with ``data_override`` already filled — the same path bar and density stats use.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from plot3.expr import ExprError, Formula, evaluate
from plot3.geoms import _Geom, aes, geom_line, geom_path
from plot3.stats3d import regular_grid_mesh
from plot3.table import has_column, numeric_array

_DEFAULT_DOMAIN = (-10.0, 10.0)
_N_CURVE = 501
_N_GRID = 80
# The view fits the contour. A unit circle on the default (-10, 10) grid is
# only a few dozen chords, so refit the grid around the zero set until the
# contour fills it (or the cells are already fine).
_CONTOUR_FILL = 0.72
_CONTOUR_CELLS = 140
_CONTOUR_REFITS = 2


def data_domains(figure: Any, data: Any) -> dict[str, tuple[float, float]]:
    """Numeric ranges of x/y/z columns already mapped on non-function layers."""
    if data is None:
        return {}
    found: dict[str, tuple[float, float]] = {}
    mappings = [getattr(figure, "mapping", None) or {}]
    for layer in getattr(figure, "layers", []):
        if getattr(layer, "kind", None) == "function":
            continue
        mapping = getattr(layer, "mapping", None)
        if mapping:
            mappings.append(mapping)
    for mapping in mappings:
        getter = getattr(mapping, "get", None)
        if getter is None:
            continue
        for axis in ("x", "y", "z"):
            if axis in found:
                continue
            column = getter(axis)
            if not column or not has_column(data, column):
                continue
            try:
                values = numeric_array(data, column, dropna=True)
            except Exception:
                continue
            if values.size == 0:
                continue
            lo = float(np.min(values))
            hi = float(np.max(values))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if hi <= lo:
                lo, hi = lo - 1.0, hi + 1.0
            found[axis] = (lo, hi)
    return found


def expand_function(
    geom: _Geom,
    base_mapping: Any,
    data: Any,
    domains: dict[str, tuple[float, float]] | None = None,
) -> _Geom:
    """Turn ``geom_function`` into a line, path, or surface layer."""
    del base_mapping, data  # the formula carries its own samples
    formula: Formula = geom.formula
    domains = domains or {}
    axes = _assign_axes(formula)
    if axes.kind == "surface":
        return _expand_surface(geom, formula, axes, domains)
    if axes.kind == "implicit":
        return _expand_implicit(geom, formula, axes, domains)
    return _expand_curve(geom, formula, axes, domains)


class _Axes:
    def __init__(
        self,
        kind: str,
        x: str,
        y: str,
        z: str | None,
        computed: str | None,
    ):
        self.kind = kind
        self.x = x
        self.y = y
        self.z = z
        self.computed = computed


def _assign_axes(formula: Formula) -> _Axes:
    variables = list(formula.variables)
    if formula.mode == "callable":
        if len(formula.fn_args) >= 2:
            return _Axes("surface", formula.fn_args[0], formula.fn_args[1], "z", "z")
        name = formula.fn_args[0] if formula.fn_args else "x"
        y_name = "f" if name == "y" else "y"
        return _Axes("curve", name, y_name, None, "y")
    if formula.mode == "implicit":
        ordered = _prefer_xy(variables)
        return _Axes("implicit", ordered[0], ordered[1], None, None)
    dependent = formula.dependent or "y"
    if len(variables) >= 2:
        ordered = _prefer_xy(variables)
        z_name = dependent if dependent not in ordered else "z"
        return _Axes("surface", ordered[0], ordered[1], z_name, "z")
    if len(variables) == 1 and dependent == "x":
        return _Axes("curve", "x", variables[0], None, "x")
    if len(variables) == 0:
        return _Axes("curve", "x", dependent, None, "y")
    return _Axes("curve", variables[0], dependent, None, "y")


def _prefer_xy(names: list[str]) -> list[str]:
    """Put ``x`` then ``y`` first when those names are present."""
    rest = [name for name in names if name not in {"x", "y"}]
    ordered: list[str] = []
    for prefer in ("x", "y"):
        if prefer in names:
            ordered.append(prefer)
    ordered.extend(rest)
    return ordered


def _limit_pair(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and _both_real(value[0], value[1])
    ):
        lo, hi = float(value[0]), float(value[1])
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            hi = lo + 1.0
        return lo, hi
    raise ExprError(f"{name} must be a pair of numbers, for example (-2, 2)")


def _both_real(a: Any, b: Any) -> bool:
    try:
        return np.isfinite(float(a)) and np.isfinite(float(b))
    except (TypeError, ValueError):
        return False


def _sample_count(geom: _Geom, grid: bool) -> int:
    chosen = getattr(geom, "n", None)
    if chosen is None:
        return _N_GRID if grid else _N_CURVE
    count = int(chosen)
    if count < 2:
        raise ExprError("n must be at least 2")
    return count


def _domain_for(
    geom: _Geom,
    axis: str,
    domains: dict[str, tuple[float, float]],
) -> tuple[tuple[float, float], str]:
    """Return ``((lo, hi), source)`` where source is user, data, or default."""
    explicit = _limit_pair(getattr(geom, axis + "lim", None), axis + "lim")
    if explicit is not None:
        return explicit, "user"
    if axis in domains:
        return domains[axis], "data"
    return _DEFAULT_DOMAIN, "default"


def _linspace(lo: float, hi: float, count: int) -> np.ndarray:
    return np.linspace(float(lo), float(hi), int(count))


def _call_formula(formula: Formula, variables: dict[str, np.ndarray]) -> np.ndarray:
    values = evaluate(formula, variables)
    shapes = [np.shape(array) for array in variables.values()]
    target = shapes[0] if shapes else ()
    if values.shape != target:
        try:
            values = np.broadcast_to(values, target).astype(np.float64, copy=True)
        except ValueError as exc:
            raise ExprError(
                "formula result does not match the sampled grid"
            ) from exc
    return np.asarray(values, dtype=np.float64)


def _expand_curve(geom: _Geom, formula: Formula, axes: _Axes, domains: dict) -> _Geom:
    # Sideways ``x = f(y)`` samples the vertical axis. Everything else samples x.
    sample_axis = "y" if axes.computed == "x" else "x"
    (lo, hi), source = _domain_for(geom, sample_axis, domains)
    count = _sample_count(geom, grid=False)
    samples = _linspace(lo, hi, count)
    lo, hi, samples = _narrow_curve(
        formula, axes, samples, lo, hi, source, count
    )
    values = _curve_values(formula, axes, samples)
    view_axis = "x" if axes.computed == "x" else "y"
    view_lim = _limit_pair(getattr(geom, view_axis + "lim", None), view_axis + "lim")
    kept_s, kept_v, lock, index = _clip_series(samples, values, view_lim, view_axis)
    if axes.computed == "x":
        xs, ys = kept_v, kept_s
    else:
        xs, ys = kept_s, kept_v
    frame = pd.DataFrame({"x": np.asarray(xs, dtype=np.float64), "y": np.asarray(ys, dtype=np.float64)})
    groups = _groups_from_index(index)
    if not any(count >= 2 for _start, count in groups):
        raise ExprError("geom_function() needs at least two points on this domain")
    # Sideways ``x = f(y)`` must keep sample order. ``_groups`` tells the
    # encoder not to sort the line and where to break it.
    maker = geom_path if axes.computed == "x" else geom_line
    linewidth = getattr(geom, "linewidth", None)
    out = maker(
        aes(x="x", y="y"),
        linewidth=2.0 if linewidth is None else linewidth,
        color=geom.const_color,
        alpha=geom.alpha,
    )
    out.data_override = frame
    out._groups = groups
    out._replace_mapping = True
    out._legend_label = formula.label
    out._axis_labels = {"x": axes.x, "y": axes.y}
    if lock is not None:
        out._axis_lock = {view_axis: lock}
    return out


def _curve_values(formula: Formula, axes: _Axes, samples: np.ndarray) -> np.ndarray:
    del axes
    if formula.mode == "callable":
        name = formula.fn_args[0]
        return _call_formula(formula, {name: samples})
    if not formula.variables:
        raw = evaluate(formula, {})
        number = float(np.asarray(raw, dtype=np.float64).reshape(-1)[0])
        return np.full(samples.shape, number, dtype=np.float64)
    return _call_formula(formula, {formula.variables[0]: samples})


def _narrow_curve(
    formula: Formula,
    axes: _Axes,
    samples: np.ndarray,
    lo: float,
    hi: float,
    source: str,
    count: int,
) -> tuple[float, float, np.ndarray]:
    """Shrink the default domain when the formula is undefined on most of it."""
    if source != "default":
        return lo, hi, samples
    values = _curve_values(formula, axes, samples)
    finite = np.isfinite(values)
    fraction = float(np.mean(finite)) if finite.size else 0.0
    if fraction >= 0.55 or fraction == 0.0:
        return lo, hi, samples
    good = samples[finite]
    nlo, nhi = float(np.min(good)), float(np.max(good))
    if nhi <= nlo:
        return lo, hi, samples
    narrowed = _linspace(nlo, nhi, count)
    return nlo, nhi, narrowed


def _clip_series(
    samples: np.ndarray,
    values: np.ndarray,
    view_lim: tuple[float, float] | None,
    view_axis: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float] | None, np.ndarray]:
    finite = np.isfinite(samples) & np.isfinite(values)
    if not np.any(finite):
        raise ExprError("geom_function() is undefined everywhere on this domain")
    if view_lim is not None:
        lo, hi = view_lim
        keep = finite & (values >= lo) & (values <= hi)
        lock: tuple[float, float] | None = (lo, hi)
    else:
        lo, hi, blew_up = _robust_window(values[finite])
        if blew_up:
            keep = finite & (values >= lo) & (values <= hi)
            warnings.warn(
                "geom_function clipped extreme values to a robust window; "
                f"pass {view_axis}lim= to choose the range",
                UserWarning,
                stacklevel=4,
            )
        else:
            keep = finite
        lock = None
    if not np.any(keep):
        raise ExprError(
            f"geom_function() has no points inside {view_axis}lim="
            f"({view_lim[0]:.6g}, {view_lim[1]:.6g})"
            if view_lim is not None
            else "geom_function() is undefined everywhere on this domain"
        )
    index = np.flatnonzero(keep)
    return samples[index], values[index], lock, index


def _groups_from_index(index: np.ndarray) -> list[list[int]]:
    """Break the line wherever clipped samples left a gap."""
    if index.size == 0:
        return []
    groups: list[list[int]] = []
    start = 0
    for position in range(1, int(index.size)):
        if int(index[position]) != int(index[position - 1]) + 1:
            groups.append([start, position - start])
            start = position
    groups.append([start, int(index.size) - start])
    return groups


def _robust_window(values: np.ndarray) -> tuple[float, float, bool]:
    """Return ``(lo, hi, blew_up)`` around the bulk of ``values``."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = max(mad * 1.4826, 1e-9)
    lo = med - 8.0 * scale
    hi = med + 8.0 * scale
    full_lo = float(np.min(values))
    full_hi = float(np.max(values))
    blew_up = full_lo < lo - 1e-8 or full_hi > hi + 1e-8
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi, blew_up


def _expand_surface(
    geom: _Geom, formula: Formula, axes: _Axes, domains: dict
) -> _Geom:
    count = _sample_count(geom, grid=True)
    (xlo, xhi), x_source = _domain_for(geom, "x", domains)
    (ylo, yhi), y_source = _domain_for(geom, "y", domains)
    xs = _linspace(xlo, xhi, count)
    ys = _linspace(ylo, yhi, count)
    zz = _surface_values(formula, axes, xs, ys)
    xs, ys, zz = _narrow_surface(
        formula,
        axes,
        xs,
        ys,
        zz,
        x_source == "default" and _limit_pair(geom.xlim, "xlim") is None,
        y_source == "default" and _limit_pair(geom.ylim, "ylim") is None,
        count,
    )
    zz, lock = _clip_grid(zz, _limit_pair(geom.zlim, "zlim"))
    xx, yy = np.meshgrid(xs, ys)
    frame = pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": zz.ravel(),
        }
    )
    vertices, indices, nx, ny = regular_grid_mesh(frame, "x", "y", "z")
    out = _Geom(
        aes(x="x", y="y", z="z"),
        color=geom.const_color,
        alpha=geom.alpha if geom.alpha is not None else 0.95,
    )
    out.kind = "surface"
    out.data_override = vertices
    out.const_color = geom.const_color
    out.alpha = geom.alpha if geom.alpha is not None else 0.95
    out.wireframe = bool(getattr(geom, "wireframe", False))
    out._indices = indices
    out._nx = nx
    out._ny = ny
    out._replace_mapping = True
    out._legend_label = formula.label
    out._axis_labels = {"x": axes.x, "y": axes.y, "z": axes.z or "z"}
    if lock is not None:
        out._axis_lock = {"z": lock}
    return out


def _surface_values(
    formula: Formula, axes: _Axes, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    xx, yy = np.meshgrid(xs, ys)
    if formula.mode == "callable":
        return _call_formula(
            formula,
            {formula.fn_args[0]: xx, formula.fn_args[1]: yy},
        )
    return _call_formula(formula, {axes.x: xx, axes.y: yy})


def _narrow_surface(
    formula: Formula,
    axes: _Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    zz: np.ndarray,
    narrow_x: bool,
    narrow_y: bool,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(zz)
    fraction = float(np.mean(finite)) if finite.size else 0.0
    if fraction >= 0.55 or fraction == 0.0 or not (narrow_x or narrow_y):
        return xs, ys, zz
    rows = np.any(finite, axis=1)
    cols = np.any(finite, axis=0)
    if narrow_y and np.any(rows):
        ys = _linspace(float(ys[rows][0]), float(ys[rows][-1]), count)
    if narrow_x and np.any(cols):
        xs = _linspace(float(xs[cols][0]), float(xs[cols][-1]), count)
    return xs, ys, _surface_values(formula, axes, xs, ys)


def _clip_grid(
    zz: np.ndarray, zlim: tuple[float, float] | None
) -> tuple[np.ndarray, tuple[float, float] | None]:
    finite = zz[np.isfinite(zz)]
    if finite.size == 0:
        raise ExprError("geom_function() is undefined everywhere on this domain")
    if zlim is not None:
        lo, hi = zlim
        clipped = np.clip(np.where(np.isfinite(zz), zz, lo), lo, hi)
        return clipped, (lo, hi)
    lo, hi, blew_up = _robust_window(finite)
    if not blew_up:
        filled = np.where(np.isfinite(zz), zz, float(np.median(finite)))
        return filled, None
    warnings.warn(
        "geom_function clipped extreme values to a robust window; "
        "pass zlim= to choose the range",
        UserWarning,
        stacklevel=4,
    )
    filled = np.where(np.isfinite(zz), zz, lo)
    filled = np.clip(filled, lo, hi)
    return filled, (lo, hi)


def _expand_implicit(
    geom: _Geom, formula: Formula, axes: _Axes, domains: dict
) -> _Geom:
    count = _sample_count(geom, grid=True)
    (xlo, xhi), _x_source = _domain_for(geom, "x", domains)
    (ylo, yhi), _y_source = _domain_for(geom, "y", domains)
    polylines: list[list[tuple[float, float]]] = []
    # Up to two refits. Stop once the contour fills the grid, and keep the
    # last curve found if a tighter window misses it.
    for _attempt in range(_CONTOUR_REFITS + 1):
        xs = _linspace(xlo, xhi, count)
        ys = _linspace(ylo, yhi, count)
        xx, yy = np.meshgrid(xs, ys)
        field = _call_formula(formula, {axes.x: xx, axes.y: yy})
        found = _marching_squares(xs, ys, field, level=0.0)
        if not found:
            break
        polylines = found
        window = _tighter_contour_window(xs, ys, polylines, xlo, xhi, ylo, yhi)
        if window is None:
            break
        xlo, xhi, ylo, yhi = window
    if not polylines:
        raise ExprError(
            "geom_function() found no curve where the equation is zero "
            "on this domain. Try a wider xlim= and ylim="
        )
    rows_x: list[float] = []
    rows_y: list[float] = []
    groups: list[list[int]] = []
    for poly in polylines:
        if len(poly) < 2:
            continue
        start = len(rows_x)
        for x_val, y_val in poly:
            rows_x.append(x_val)
            rows_y.append(y_val)
        groups.append([start, len(rows_x) - start])
    if not groups:
        raise ExprError(
            "geom_function() found no curve where the equation is zero "
            "on this domain. Try a wider xlim= and ylim="
        )
    frame = pd.DataFrame({"x": rows_x, "y": rows_y})
    linewidth = getattr(geom, "linewidth", None)
    out = geom_path(
        aes(x="x", y="y"),
        linewidth=2.0 if linewidth is None else linewidth,
        color=geom.const_color,
        alpha=geom.alpha,
    )
    out.data_override = frame
    out.sort_x = False
    out._groups = groups
    out._replace_mapping = True
    out._implicit = True
    out._legend_label = formula.label
    out._axis_labels = {"x": axes.x, "y": axes.y}
    return out


def _tighter_contour_window(
    xs: np.ndarray,
    ys: np.ndarray,
    polylines: list[list[tuple[float, float]]],
    xlo: float,
    xhi: float,
    ylo: float,
    yhi: float,
) -> tuple[float, float, float, float] | None:
    """Return a smaller sample window around the contour, or None to keep it.

    Padding stays inside the current window and is at least one cell, so the
    next pass still has grid cells on both sides of the curve.
    """
    points = [pt for poly in polylines if len(poly) >= 2 for pt in poly]
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    xmin, ymin = (float(v) for v in arr.min(axis=0))
    xmax, ymax = (float(v) for v in arr.max(axis=0))
    dx = float(xs[-1] - xs[0]) / max(len(xs) - 1, 1)
    dy = float(ys[-1] - ys[0]) / max(len(ys) - 1, 1)
    if dx <= 0.0 or dy <= 0.0:
        return None
    span_x = max(xmax - xmin, dx)
    span_y = max(ymax - ymin, dy)
    win_x = float(xhi - xlo)
    win_y = float(yhi - ylo)
    if win_x <= 0.0 or win_y <= 0.0:
        return None
    fills = span_x >= _CONTOUR_FILL * win_x and span_y >= _CONTOUR_FILL * win_y
    cells_across = min(span_x / dx, span_y / dy)
    if fills or cells_across >= _CONTOUR_CELLS:
        return None
    pad_x = max(dx, 0.12 * span_x)
    pad_y = max(dy, 0.12 * span_y)
    nlo = max(float(xlo), xmin - pad_x)
    nhi = min(float(xhi), xmax + pad_x)
    mlo = max(float(ylo), ymin - pad_y)
    mhi = min(float(yhi), ymax + pad_y)
    if nhi - nlo <= dx or mhi - mlo <= dy:
        return None
    shrunk = (nhi - nlo) < 0.85 * win_x or (mhi - mlo) < 0.85 * win_y
    if not shrunk:
        return None
    return nlo, nhi, mlo, mhi


def _marching_squares(
    xs: np.ndarray,
    ys: np.ndarray,
    field: np.ndarray,
    level: float,
) -> list[list[tuple[float, float]]]:
    """Zero contour of ``field`` (shape ny × nx) as polylines."""
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    ny, nx = field.shape
    for j in range(ny - 1):
        for i in range(nx - 1):
            corners = (
                (float(xs[i]), float(ys[j]), float(field[j, i])),
                (float(xs[i + 1]), float(ys[j]), float(field[j, i + 1])),
                (float(xs[i + 1]), float(ys[j + 1]), float(field[j + 1, i + 1])),
                (float(xs[i]), float(ys[j + 1]), float(field[j + 1, i])),
            )
            if not all(np.isfinite(corner[2]) for corner in corners):
                continue
            segments.extend(_cell_segments(corners, level))
    return _chain_segments(segments)


def _cell_segments(
    corners: tuple[tuple[float, float, float], ...],
    level: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    inside = [corner[2] >= level for corner in corners]
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    hits = [
        edge
        for edge, (a, b) in enumerate(pairs)
        if inside[a] != inside[b]
    ]
    if len(hits) == 2:
        return [(_cross(corners, pairs[hits[0]], level), _cross(corners, pairs[hits[1]], level))]
    if len(hits) != 4:
        return []
    # Saddle: two crossings. Pair edges so the cell centre agrees with the field.
    center = sum(corner[2] for corner in corners) / 4.0
    # hits are 0,1,2,3 (bottom, right, top, left) when all four edges cross.
    if center >= level:
        pairing = ((0, 1), (2, 3))
    else:
        pairing = ((0, 3), (1, 2))
    out = []
    for a, b in pairing:
        out.append(
            (
                _cross(corners, pairs[a], level),
                _cross(corners, pairs[b], level),
            )
        )
    return out


def _cross(
    corners: tuple[tuple[float, float, float], ...],
    edge: tuple[int, int],
    level: float,
) -> tuple[float, float]:
    a = corners[edge[0]]
    b = corners[edge[1]]
    span = b[2] - a[2]
    if span == 0.0:
        t = 0.5
    else:
        t = (level - a[2]) / span
    t = min(1.0, max(0.0, t))
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _chain_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    """Greedily join segment endpoints into polylines."""
    if not segments:
        return []

    def key(point: tuple[float, float]) -> tuple[float, float]:
        return (round(point[0], 8), round(point[1], 8))

    unused = set(range(len(segments)))
    by_point: dict[tuple[float, float], list[tuple[int, int]]] = {}
    for index, (start, end) in enumerate(segments):
        by_point.setdefault(key(start), []).append((index, 0))
        by_point.setdefault(key(end), []).append((index, 1))

    polylines: list[list[tuple[float, float]]] = []
    while unused:
        index = unused.pop()
        start, end = segments[index]
        poly = [start, end]
        _extend(poly, unused, segments, by_point, key, forward=True)
        _extend(poly, unused, segments, by_point, key, forward=False)
        if key(poly[0]) == key(poly[-1]) and len(poly) > 2:
            poly = poly[:-1]
            poly.append(poly[0])
        elif _close_enough(poly[0], poly[-1]):
            poly.append(poly[0])
        polylines.append(poly)
    return polylines


def _extend(
    poly: list[tuple[float, float]],
    unused: set[int],
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    by_point: dict[tuple[float, float], list[tuple[int, int]]],
    key,
    forward: bool,
) -> None:
    while True:
        tip = poly[-1] if forward else poly[0]
        nxt = None
        for index, _end in by_point.get(key(tip), []):
            if index in unused:
                nxt = index
                break
        if nxt is None:
            return
        unused.remove(nxt)
        start, end = segments[nxt]
        other = end if key(start) == key(tip) else start
        if forward:
            poly.append(other)
        else:
            poly.insert(0, other)


def _close_enough(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) <= 1e-8 and abs(a[1] - b[1]) <= 1e-8
