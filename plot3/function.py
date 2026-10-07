"""Sample a :class:`plot3.expr.Formula` into a drawable line or surface.

``geom_function`` stays a thin parameter holder. At build time this module
evaluates it and returns a ``geom_line``, ``geom_path``, or ``geom_surface``
with ``data_override`` already filled — the same path bar and density stats use.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from plot3.contour import _contour_lines, _refine_active_cells
from plot3.expr import ExprError, Formula, evaluate
from plot3.mathtext import split_math
from plot3.geoms import _Geom, aes, geom_line, geom_path
from plot3.stats3d import regular_grid_mesh
from plot3.table import has_column, numeric_array

_DEFAULT_DOMAIN = (-10.0, 10.0)
_N_CURVE = 501
_N_GRID = 80


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
    view_name = axes.x if view_axis == "x" else axes.y
    kept_s, kept_v, lock, index, note = _clip_series(
        samples, values, view_lim, view_axis, view_name
    )
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
    _stamp_formula(out, geom, formula)
    out._axis_labels = {"x": axes.x, "y": axes.y}
    if lock is not None:
        out._axis_lock = {view_axis: lock}
    if note:
        out._notes = [note]
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
    note_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float] | None, np.ndarray, str | None]:
    finite = np.isfinite(samples) & np.isfinite(values)
    if not np.any(finite):
        raise ExprError("geom_function() is undefined everywhere on this domain")
    note: str | None = None
    if view_lim is not None:
        lo, hi = view_lim
        keep = finite & (values >= lo) & (values <= hi)
        lock: tuple[float, float] | None = (lo, hi)
    else:
        lo, hi, blew_up = _robust_window(values[finite])
        if blew_up:
            keep = finite & (values >= lo) & (values <= hi)
            note = _clip_note(note_name or view_axis, lo, hi, param=view_axis)
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
    return samples[index], values[index], lock, index, note


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


def _bound(value: float) -> str:
    """``-2.3`` with a Unicode minus, so the caption matches the axis ticks."""
    return f"{value:.6g}".replace("-", "−")


def _clip_note(name: str, lo: float, hi: float, *, param: str | None = None) -> str:
    """Caption for a pole that was clipped to the bulk of the samples.

    ``name`` is the variable the reader sees (``t`` on a surface of ``t``).
    ``param`` is the keyword that changes the window (``zlim``).
    """
    flag = param or name
    return (
        f"{name} clipped to [{_bound(lo)}, {_bound(hi)}]; "
        f"pass {flag}lim= to change"
    )


def _stamp_formula(out, geom, formula: Formula) -> None:
    """Legend text, and the symbolic form the tooltip shows above the values."""
    custom = getattr(geom, "label", None)
    if custom:
        plain, segments = split_math(str(custom))
        out._legend_label = plain
        out._legend_math = segments
        out._legend_latex = None
        if formula.mode == "callable":
            if segments and len(segments) == 1 and segments[0].get("latex"):
                out._tip_latex = segments[0]["latex"]
                out._tip_pretty = segments[0]["text"]
            else:
                out._tip_latex = ""
                out._tip_pretty = plain
        else:
            out._tip_latex = formula.caption_latex or formula.latex
            out._tip_pretty = formula.caption_pretty or formula.pretty
    else:
        out._legend_label = formula.legend_pretty or formula.pretty or formula.label
        out._legend_latex = formula.legend_latex or formula.latex or None
        out._legend_math = None
        out._tip_latex = formula.caption_latex or formula.latex or ""
        out._tip_pretty = formula.caption_pretty or formula.pretty or out._legend_label
    out._is_formula = True


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
    zz, lock, note = _clip_grid(
        zz, _limit_pair(geom.zlim, "zlim"), axes.z or "z"
    )
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
    _stamp_formula(out, geom, formula)
    out._axis_labels = {"x": axes.x, "y": axes.y, "z": axes.z or "z"}
    out._function_surface = True
    if lock is not None:
        out._axis_lock = {"z": lock}
    if note:
        out._notes = [note]
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
    zz: np.ndarray,
    zlim: tuple[float, float] | None,
    note_name: str = "z",
) -> tuple[np.ndarray, tuple[float, float] | None, str | None]:
    finite = zz[np.isfinite(zz)]
    if finite.size == 0:
        raise ExprError("geom_function() is undefined everywhere on this domain")
    if zlim is not None:
        lo, hi = zlim
        clipped = np.clip(np.where(np.isfinite(zz), zz, lo), lo, hi)
        return clipped, (lo, hi), None
    lo, hi, blew_up = _robust_window(finite)
    if not blew_up:
        filled = np.where(np.isfinite(zz), zz, float(np.median(finite)))
        return filled, None, None
    filled = np.where(np.isfinite(zz), zz, lo)
    filled = np.clip(filled, lo, hi)
    return filled, (lo, hi), _clip_note(note_name, lo, hi, param="z")


def _expand_implicit(
    geom: _Geom, formula: Formula, axes: _Axes, domains: dict
) -> _Geom:
    count = _sample_count(geom, grid=True)
    (xlo, xhi), _x_source = _domain_for(geom, "x", domains)
    (ylo, yhi), _y_source = _domain_for(geom, "y", domains)
    xs = _linspace(xlo, xhi, count)
    ys = _linspace(ylo, yhi, count)
    xx, yy = np.meshgrid(xs, ys)
    field = _call_formula(formula, {axes.x: xx, axes.y: yy})

    def sample(xx_fine: np.ndarray, yy_fine: np.ndarray) -> np.ndarray:
        return _call_formula(formula, {axes.x: xx_fine, axes.y: yy_fine})

    # The base grid stays at ``n`` (default 80). Only cells the contour
    # crosses are subdivided, so a small loop stays smooth beside a long
    # curve. A full-window refit cannot separate those two pieces. If the
    # finer grid would be too large, keep the coarse contour.
    polylines = _refine_active_cells(xs, ys, field, 0.0, sample)
    if polylines is None:
        polylines = _contour_lines(xs, ys, field, 0.0)
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
    _stamp_formula(out, geom, formula)
    out._axis_labels = {"x": axes.x, "y": axes.y}
    return out
