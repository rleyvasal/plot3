"""Shaded integrals, marks, tangents, parametric curves, polar plots, fields.

``geom_function`` stays the formula. This module turns the extra math
(``area``, ``tangent``, ``derivative``, ``mark``, polar coordinates, and
``geom_vector_field``) into line, area, polygon, and point layers the
viewer already draws.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import math

import numpy as np
import pandas as pd

from plot3.expr import (
    ExprError,
    Formula,
    _compile_tree,
    _missing_param_build_message,
    differentiate,
    evaluate,
)
from plot3.function import (
    _DEFAULT_DOMAIN,
    _assign_axes,
    _call_formula,
    _curve_values,
    _domain_for,
    _linspace,
    _sample_count,
    _stamp_formula,
)
from plot3.geoms import _Geom, aes, geom_line, geom_path, geom_point
from plot3.mathtext import formula_texts

_POLAR_DOMAIN = (0.0, float(2.0 * np.pi))


def animation_blocked(formula: Formula, geom, coord, addons) -> str | None:
    """Why a slider or parameter sweep cannot drive this layer, or None."""
    names: list[str] = []
    for addon in addons or []:
        names.append(type(addon).__name__ + "()")
    if tuple(getattr(geom, "marks", ()) or ()):
        names.append("mark=")
    if formula.mode == "parametric":
        names.append("a parametric curve")
    elif formula.mode == "inequality":
        names.append("an inequality")
    elif formula.mode == "field":
        names.append("a vector field")
    if coord is not None and type(coord).__name__ == "coord_polar":
        names.append("coord_polar()")
    if not names:
        return None
    head = names[0]
    return (
        f"{head} cannot follow slider() or transition_time() yet. "
        "Draw it at one value of the coefficient."
    )


def expand_special(geom, formula: Formula, domains, coord) -> list | None:
    """Parametric, inequality, and polar layers. None for an ordinary formula."""
    if formula.mode == "parametric":
        return [_expand_parametric(geom, formula)]
    if formula.mode == "inequality":
        return _expand_inequality(geom, formula, domains)
    if coord is not None and type(coord).__name__ == "coord_polar":
        return [_expand_polar(geom, formula)]
    return None


def attach_calculus(primary, geom, formula: Formula, domains, addons) -> list:
    """Fills and marks that belong to ``primary``. The curve stays on top of a fill."""
    axes = _assign_axes(formula)
    before: list = []
    after: list = []
    for addon in addons or []:
        kind = type(addon).__name__
        if kind == "area":
            before.append(_integral_layer(geom, formula, axes, addon, primary))
        elif kind == "tangent":
            after.extend(_tangent_layers(geom, formula, axes, primary, addon))
        elif kind == "derivative":
            after.append(_derivative_layer(geom, formula, axes, primary))
        else:
            raise TypeError(f"cannot draw {kind}()")
    marks = tuple(getattr(geom, "marks", ()) or ())
    if "roots" in marks or "extrema" in marks:
        after.extend(_mark_layers(geom, formula, axes, primary, marks))
    if "intersections" in marks:
        primary._want_intersections = True
    return before + [primary] + after


def mark_intersections(layers: list) -> list:
    """Add a point where two function curves cross, when one of them asked."""
    if not any(getattr(layer, "_want_intersections", False) for layer in layers):
        return layers
    curves = [layer for layer in layers if _is_function_curve(layer)]
    found: list[tuple[float, float]] = []
    for index, left in enumerate(curves):
        for right in curves[index + 1 :]:
            found.extend(_crossings(left, right))
    if not found:
        return layers
    frame = pd.DataFrame(
        {
            "x": [point[0] for point in found],
            "y": [point[1] for point in found],
        }
    )
    out = geom_point(aes(x="x", y="y"), size=8, alpha=1)
    out.data_override = frame
    out._replace_mapping = True
    out._legend_label = "intersections"
    out._is_formula = True
    return [*layers, out]


def expand_vector_field(geom, transition, slider) -> list:
    """Arrows or streamlines for ``dx, dy`` on a grid."""
    formula: Formula = geom.formula
    if formula.mode != "field":
        raise ExprError(
            'geom_vector_field() needs dx and dy, for example "dx = -y, dy = x"'
        )
    sweep = slider if slider is not None else transition
    ranges = getattr(sweep, "ranges", None) or {}
    pending = tuple(getattr(formula, "pending", ()) or ())
    if pending and any(name not in ranges for name in pending):
        missing = next(name for name in pending if name not in ranges)
        raise ExprError(_missing_param_build_message(missing))
    if ranges:
        raise ValueError(
            "a vector field cannot follow slider() or transition_time() yet. "
            "Draw it at one value of the coefficient."
        )
    xlo, xhi = _pair(geom.xlim, "xlim", (-2.0, 2.0))
    ylo, yhi = _pair(geom.ylim, "ylim", (-2.0, 2.0))
    count = max(2, int(geom.n))
    xs = _linspace(xlo, xhi, count)
    ys = _linspace(ylo, yhi, count)
    xx, yy = np.meshgrid(xs, ys)
    dx = _component(formula, "dx", {"x": xx, "y": yy})
    dy = _component(formula, "dy", {"x": xx, "y": yy})
    cell = min((xhi - xlo) / (count - 1), (yhi - ylo) / (count - 1))
    if geom.stream:
        rows, groups = _streamlines(
            formula, xs, ys, cell, (xlo, xhi), (ylo, yhi)
        )
    else:
        rows, groups = _arrows(xx, yy, dx, dy, 0.72 * cell)
    if not rows:
        raise ExprError(
            "geom_vector_field() has no arrows on this domain. "
            "Try a wider xlim= and ylim="
        )
    frame = pd.DataFrame(
        {"x": [row[0] for row in rows], "y": [row[1] for row in rows]}
    )
    out = geom_path(
        aes(x="x", y="y"),
        linewidth=1.5 if geom.linewidth is None else geom.linewidth,
        color=geom.const_color,
        alpha=geom.alpha if geom.alpha is not None else 0.95,
    )
    out.data_override = frame
    out._groups = groups
    out._replace_mapping = True
    out.sort_x = False
    _stamp_formula(out, geom, formula)
    out._axis_labels = {"x": "x", "y": "y"}
    return [out]


def _expand_parametric(geom, formula: Formula):
    if formula.parameter == "":
        raise ExprError(
            'a parametric curve needs one parameter, for example '
            '"x = cos(t), y = sin(t)"'
        )
    lo, hi = _parameter_domain(geom)
    count = _sample_count(geom, grid=False)
    samples = _linspace(lo, hi, count)
    env = {formula.parameter: samples}
    columns = {
        name: _component(formula, name, env) for name, _code in formula.components
    }
    has_z = "z" in columns
    xs = np.asarray(columns["x"], dtype=np.float64).reshape(-1)
    ys = np.asarray(columns["y"], dtype=np.float64).reshape(-1)
    zs = np.asarray(columns["z"], dtype=np.float64).reshape(-1) if has_z else None
    finite = np.isfinite(xs) & np.isfinite(ys)
    if zs is not None:
        finite = finite & np.isfinite(zs)
    if int(np.count_nonzero(finite)) < 2:
        raise ExprError("geom_function() needs at least two points on this domain")
    keep = np.flatnonzero(finite)
    data = {"x": xs[keep], "y": ys[keep]}
    mapping = {"x": "x", "y": "y"}
    if zs is not None:
        data["z"] = zs[keep]
        mapping["z"] = "z"
    out = geom_path(
        aes(**mapping),
        linewidth=2.0 if geom.linewidth is None else geom.linewidth,
        color=geom.const_color,
        alpha=geom.alpha,
    )
    out.data_override = pd.DataFrame(data)
    out._groups = _runs(keep)
    out._replace_mapping = True
    out.sort_x = False
    _stamp_formula(out, geom, formula)
    labels = {"x": "x", "y": "y"}
    if has_z:
        labels["z"] = "z"
    out._axis_labels = labels
    return out


def _parameter_domain(geom) -> tuple[float, float]:
    chosen = getattr(geom, "tlim", None)
    if chosen is None:
        chosen = getattr(geom, "xlim", None)
    if chosen is None:
        return _DEFAULT_DOMAIN
    return _pair(chosen, "tlim", _DEFAULT_DOMAIN)


def _expand_polar(geom, formula: Formula):
    if formula.mode != "explicit" or formula.dependent != "r":
        raise ExprError(
            'coord_polar() plots r = f(theta). For example '
            'geom_function("r = 1 + cos(theta)") + coord_polar()'
        )
    if len(formula.variables) > 1:
        raise ExprError(
            "coord_polar() needs one angle, for example r = 1 + cos(theta)"
        )
    lo, hi = _POLAR_DOMAIN
    if getattr(geom, "tlim", None) is not None:
        lo, hi = _pair(geom.tlim, "tlim", _POLAR_DOMAIN)
    count = _sample_count(geom, grid=False)
    theta = _linspace(lo, hi, count)
    name = formula.variables[0] if formula.variables else "theta"
    radius = _call_formula(formula, {name: theta})
    finite = np.isfinite(theta) & np.isfinite(radius)
    if int(np.count_nonzero(finite)) < 2:
        raise ExprError("geom_function() needs at least two points on this domain")
    keep = np.flatnonzero(finite)
    angle = theta[keep]
    radial = radius[keep]
    frame = pd.DataFrame(
        {
            "x": radial * np.cos(angle),
            "y": radial * np.sin(angle),
        }
    )
    out = geom_path(
        aes(x="x", y="y"),
        linewidth=2.0 if geom.linewidth is None else geom.linewidth,
        color=geom.const_color,
        alpha=geom.alpha,
    )
    out.data_override = frame
    out._groups = _runs(keep)
    out._replace_mapping = True
    out.sort_x = False
    _stamp_formula(out, geom, formula)
    out._axis_labels = {"x": "x", "y": "y"}
    return out


def _expand_inequality(geom, formula: Formula, domains) -> list:
    if formula.dependent in {"x", "y"}:
        return _curve_inequality(geom, formula, domains)
    return _region_inequality(geom, formula, domains)


def _curve_inequality(geom, formula: Formula, domains) -> list:
    dependent = formula.dependent or "y"
    if dependent == "y":
        (lo, hi), _source = _domain_for(geom, "x", domains)
    else:
        (lo, hi), _source = _domain_for(geom, "y", domains)
    count = _sample_count(geom, grid=False)
    samples = _linspace(lo, hi, count)
    if formula.variables:
        boundary = _call_formula(formula, {formula.variables[0]: samples})
    else:
        raw = evaluate(formula, {})
        number = float(np.asarray(raw, dtype=np.float64).reshape(-1)[0])
        boundary = np.full(samples.shape, number, dtype=np.float64)
    finite = np.isfinite(samples) & np.isfinite(boundary)
    if int(np.count_nonzero(finite)) < 2:
        raise ExprError("geom_function() is undefined everywhere on this domain")
    keep = np.flatnonzero(finite)
    samples = samples[keep]
    boundary = boundary[keep]
    above = formula.relation in {">", ">="}
    groups = _runs(keep)
    if dependent == "y":
        view = _limit_or_none(getattr(geom, "ylim", None), "ylim")
        baseline, lock = _open_baseline(boundary, above, view)
        shade = _area_layer(
            samples,
            boundary,
            baseline,
            groups,
            color=geom.const_color,
            alpha=0.35,
        )
        shade._axis_lock = {"y": lock}
        curve_x, curve_y = samples, boundary
    else:
        view = _limit_or_none(getattr(geom, "xlim", None), "xlim")
        baseline, lock = _open_baseline(boundary, above, view)
        shade = _vertical_shade(boundary, samples, baseline, color=geom.const_color)
        shade._axis_lock = {"x": lock}
        curve_x, curve_y = boundary, samples
    line = geom_path(
        aes(x="x", y="y"),
        linewidth=2.0 if geom.linewidth is None else geom.linewidth,
        color=geom.const_color,
        alpha=geom.alpha,
    )
    line.data_override = pd.DataFrame({"x": curve_x, "y": curve_y})
    line._groups = groups if dependent == "y" else [[0, int(samples.size)]]
    line._replace_mapping = True
    line.sort_x = False
    _stamp_formula(line, geom, formula)
    line._axis_labels = {"x": "x", "y": "y"}
    if dependent == "y":
        line._axis_lock = {"y": lock}
    else:
        line._axis_lock = {"x": lock}
    return [shade, line]


def _open_baseline(values, above: bool, view):
    finite = values[np.isfinite(values)]
    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    if view is not None:
        lo, hi = view
        baseline = hi if above else lo
        return baseline, (min(lo, ymin), max(hi, ymax))
    span = max(ymax - ymin, 1.0)
    pad = 0.22 * span
    if above:
        baseline = ymax + pad
        return baseline, (ymin - 0.08 * span, baseline)
    baseline = ymin - pad
    return baseline, (baseline, ymax + 0.08 * span)


def _vertical_shade(xs, ys, baseline, *, color):
    """Shade from a sideways boundary to a vertical baseline."""
    forward = list(zip(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)))
    back = [(float(baseline), float(y)) for _x, y in reversed(forward)]
    points = forward + back
    frame = pd.DataFrame(
        {"x": [point[0] for point in points], "y": [point[1] for point in points]}
    )
    out = _Geom(aes(x="x", y="y"), color=color, alpha=0.35)
    out.kind = "poly"
    out.data_override = frame
    out._groups = [[0, len(points)]]
    out._replace_mapping = True
    out._inherit_color = True
    out._is_formula = True
    out.linewidth = 0.0
    out.const_color = color
    out.alpha = 0.35
    return out


def _region_inequality(geom, formula: Formula, domains) -> list:
    from plot3.contour import _contour_lines

    count = _sample_count(geom, grid=True)
    (xlo, xhi), _xs = _domain_for(geom, "x", domains)
    (ylo, yhi), _ys = _domain_for(geom, "y", domains)
    xs = _linspace(xlo, xhi, count)
    ys = _linspace(ylo, yhi, count)
    # Cell centers decide the fill. The contour uses the same corner grid.
    xx, yy = np.meshgrid(xs, ys)
    names = list(formula.variables)
    field = _call_formula(formula, {names[0]: xx, names[1]: yy})
    # evaluate() follows variable order, but _call_formula names them.
    # Meshgrid is (y, x). variables may be (x, y) or (y, x). Name them.
    named = {}
    if set(names) >= {"x", "y"}:
        named = {"x": xx, "y": yy}
        field = _call_formula(formula, named)
    cx = 0.5 * (xs[:-1] + xs[1:])
    cy = 0.5 * (ys[:-1] + ys[1:])
    cxx, cyy = np.meshgrid(cx, cy)
    centers = _call_formula(formula, {"x": cxx, "y": cyy} if "x" in names else {
        names[0]: cxx, names[1]: cyy
    })
    points: list[tuple[float, float]] = []
    groups: list[list[int]] = []
    inside = np.isfinite(centers) & (centers >= 0)
    for j in range(cy.size):
        for i in range(cx.size):
            if not inside[j, i]:
                continue
            quad = (
                (float(xs[i]), float(ys[j])),
                (float(xs[i + 1]), float(ys[j])),
                (float(xs[i + 1]), float(ys[j + 1])),
                (float(xs[i]), float(ys[j + 1])),
            )
            # Paired strip: left edge bottom→top, right edge top→bottom.
            ordered = (quad[0], quad[3], quad[2], quad[1])
            start = len(points)
            points.extend(ordered)
            groups.append([start, 4])
    if not points:
        raise ExprError(
            "geom_function() found no region where the inequality holds. "
            "Try a wider xlim= and ylim="
        )
    shade = _Geom(aes(x="x", y="y"), color=geom.const_color, alpha=0.35)
    shade.kind = "poly"
    shade.data_override = pd.DataFrame(
        {"x": [p[0] for p in points], "y": [p[1] for p in points]}
    )
    shade._groups = groups
    shade._replace_mapping = True
    shade._inherit_color = True
    shade._is_formula = True
    shade.linewidth = 0.0
    shade.const_color = geom.const_color
    shade.alpha = 0.35
    polylines = _contour_lines(xs, ys, field, 0.0)
    rows_x: list[float] = []
    rows_y: list[float] = []
    line_groups: list[list[int]] = []
    for poly in polylines or []:
        if len(poly) < 2:
            continue
        start = len(rows_x)
        for x_val, y_val in poly:
            rows_x.append(float(x_val))
            rows_y.append(float(y_val))
        line_groups.append([start, len(rows_x) - start])
    layers = [shade]
    if line_groups:
        line = geom_path(
            aes(x="x", y="y"),
            linewidth=2.0 if geom.linewidth is None else geom.linewidth,
            color=geom.const_color,
            alpha=geom.alpha,
        )
        line.data_override = pd.DataFrame({"x": rows_x, "y": rows_y})
        line._groups = line_groups
        line._replace_mapping = True
        line.sort_x = False
        _stamp_formula(line, geom, formula)
        line._axis_labels = {"x": "x", "y": "y"}
        layers.append(line)
    else:
        _stamp_formula(shade, geom, formula)
        shade._inherit_color = False
        shade._axis_labels = {"x": "x", "y": "y"}
    return layers


def _integral_layer(geom, formula: Formula, axes, addon, primary=None):
    if formula.mode == "inequality":
        raise ExprError('area() integrates a curve y = f(x), not an inequality')
    if axes.kind != "curve" or axes.computed == "x":
        raise ExprError(
            'area() integrates a curve y = f(x). For example '
            'geom_function("y = x^2") + area(0, 2)'
        )
    asked_lo, asked_hi = float(addon.lo), float(addon.hi)
    baseline = float(addon.baseline)
    count = _sample_count(geom, grid=False)
    if (count - 1) % 2 == 1:
        count += 1
    edge_lo, edge_hi = _curve_edges(primary)
    lo, hi = asked_lo, asked_hi
    if not math.isfinite(lo) or not math.isfinite(hi):
        # area(-inf, -1.96) runs to the edge of the drawn curve.
        if edge_lo is None:
            raise ExprError("area() with an infinite limit needs the curve it shades")
        lo = max(lo, edge_lo)
        hi = min(hi, edge_hi)
        if hi <= lo:
            raise ExprError(
                f"area({_num(asked_lo)}, {_num(asked_hi)}) misses the curve, "
                f"which is drawn on ({_num(edge_lo)}, {_num(edge_hi)})"
            )
    xs = _linspace(lo, hi, count)
    ys = _curve_values(formula, axes, xs)
    if ys.shape != xs.shape or not np.all(np.isfinite(ys)):
        raise ExprError(
            f"area() is undefined on ({_num(lo)}, {_num(hi)})"
        )
    signed = _simpson(xs, ys) - baseline * (float(xs[-1]) - float(xs[0]))
    if baseline == 0.0 and _is_density(formula, axes, edge_lo, edge_hi, count):
        name = formula.variables[0] if formula.variables else "x"
        # The shading stops at the drawn edge; the probability does not.
        # Student's t keeps 1.5% of its mass beyond +-5.
        width = edge_hi - edge_lo
        if not math.isfinite(asked_lo):
            signed += _tail_mass(formula, axes, edge_lo, -1.0, width)
        if not math.isfinite(asked_hi):
            signed += _tail_mass(formula, axes, edge_hi, 1.0, width)
        text = _probability_text(name, asked_lo, asked_hi, signed)
    elif math.isfinite(asked_lo) and math.isfinite(asked_hi):
        text = _integral_text(signed)
    else:
        # Not a density: the infinite limit stopped at the edge of the view.
        text = _integral_text(signed).replace("∫ = ", "∫ ≈ ", 1)
    xc, yc = _centroid(xs, ys, baseline)
    # A thin area (a 2.5% tail) has its centroid on the axis, under the
    # label's backing. Put the label just above the shading instead.
    peak = _curve_peak(primary)
    if peak is not None and baseline == 0.0:
        top = float(np.max(ys))
        if 0.0 <= top < 0.25 * peak:
            yc = top + 0.12 * peak
    shade = _area_layer(
        xs, ys, baseline, [[0, int(xs.size)]], color=geom.const_color, alpha=0.35
    )
    shade._annotations = [{"x": float(xc), "y": float(yc), "text": text}]
    return shade


def _area_layer(xs, ys, baseline, groups, *, color, alpha):
    out = _Geom(aes(x="x", y="y"), color=color, alpha=alpha)
    out.kind = "area"
    out.data_override = pd.DataFrame(
        {"x": np.asarray(xs, dtype=np.float64), "y": np.asarray(ys, dtype=np.float64)}
    )
    out._groups = groups
    out._replace_mapping = True
    out._baseline = float(baseline)
    out._inherit_color = True
    out._is_formula = True
    out.linewidth = 1.5
    out.const_color = color
    out.alpha = alpha
    out.sort_x = False
    return out


def _tangent_layers(geom, formula, axes, primary, addon) -> list:
    if axes.kind != "curve" or axes.computed == "x":
        raise ExprError(
            'tangent() is for a curve y = f(x). For example '
            'geom_function("y = x^2") + tangent(at=1)'
        )
    x0 = float(addon.at)
    y0 = _scalar(formula, axes, x0)
    if not np.isfinite(y0):
        raise ExprError(f"tangent() is undefined at x={_num(x0)}")
    slope = _slope(formula, axes, x0)
    if not np.isfinite(slope):
        raise ExprError(f"tangent() is undefined at x={_num(x0)}")
    frame = primary.data_override
    xs = np.asarray(frame["x"], dtype=np.float64)
    finite = xs[np.isfinite(xs)]
    xlo = float(np.min(finite))
    xhi = float(np.max(finite))
    line_x = np.array([xlo, xhi], dtype=np.float64)
    line_y = y0 + slope * (line_x - x0)
    line = geom_line(
        aes(x="x", y="y"),
        linewidth=1.75,
        alpha=0.95,
    )
    line.data_override = pd.DataFrame({"x": line_x, "y": line_y})
    line._groups = [[0, 2]]
    line._replace_mapping = True
    line._legend_label = "tangent"
    line._is_formula = True
    token = f"tangent-{id(line)}"
    line._color_key = token
    point = geom_point(aes(x="x", y="y"), size=7, alpha=1)
    point.data_override = pd.DataFrame({"x": [x0], "y": [y0]})
    point._replace_mapping = True
    point._inherit_color = True
    point._is_formula = True
    point._inherit_from = token
    return [line, point]


def _derivative_layer(geom, formula, axes, primary):
    if axes.kind != "curve" or axes.computed == "x":
        raise ExprError(
            "derivative() is for a curve y = f(x). "
            'For example geom_function("y = x^2") + derivative()'
        )
    frame = primary.data_override
    xs = np.asarray(frame["x"], dtype=np.float64)
    var = formula.variables[0] if formula.variables else "x"
    tree = differentiate(formula, var)
    if tree is not None:
        values = _eval_tree(formula, tree, {var: xs})
        legend = _prime_texts(formula, tree)
    else:
        values = _numeric_derivative(formula, axes, xs)
        legend = None
    out = geom_line(
        aes(x="x", y="y"),
        linewidth=2.0 if geom.linewidth is None else geom.linewidth,
        alpha=geom.alpha if geom.alpha is not None else 0.95,
    )
    out.data_override = pd.DataFrame({"x": xs, "y": np.asarray(values, dtype=np.float64)})
    out._groups = [[0, int(xs.size)]]
    out._replace_mapping = True
    out._is_formula = True
    if legend is not None:
        out._legend_label = legend["pretty"]
        out._legend_latex = legend["latex"]
        out._tip_pretty = legend["pretty"]
        out._tip_latex = legend["latex"]
    else:
        out._legend_label = "y'"
        out._tip_pretty = "y'"
        out._tip_latex = "y'"
    return out


def _mark_layers(geom, formula, axes, primary, marks) -> list:
    if axes.kind != "curve":
        raise ExprError(
            "mark='roots' and mark='extrema' are for a curve y = f(x)"
        )
    frame = primary.data_override
    if axes.computed == "x":
        samples = np.asarray(frame["y"], dtype=np.float64)
        values = np.asarray(frame["x"], dtype=np.float64)
        sample_at = lambda value: _scalar_swapped(formula, axes, value)
    else:
        samples = np.asarray(frame["x"], dtype=np.float64)
        values = np.asarray(frame["y"], dtype=np.float64)
        sample_at = lambda value: _scalar(formula, axes, value)
    layers = []
    if "roots" in marks:
        roots = _roots(samples, values, sample_at)
        if roots:
            layers.append(_point_mark(
                samples_at(samples, values, roots, axes),
                "roots",
            ))
    if "extrema" in marks:
        spots = _extrema(samples, values, sample_at)
        if spots:
            layers.append(_point_mark(
                samples_at(samples, values, spots, axes),
                "extrema",
            ))
    return layers


def samples_at(samples, values, places, axes) -> pd.DataFrame:
    xs = []
    ys = []
    for place in places:
        if axes.computed == "x":
            xs.append(float(_interp(samples, values, place)))
            ys.append(place)
        else:
            xs.append(place)
            ys.append(float(_interp(samples, values, place)))
    return pd.DataFrame({"x": xs, "y": ys})


def _point_mark(frame: pd.DataFrame, label: str):
    out = geom_point(aes(x="x", y="y"), size=8, alpha=1)
    out.data_override = frame
    out._replace_mapping = True
    out._legend_label = label
    out._is_formula = True
    return out


def _roots(samples, values, f) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    scale = max(float(np.max(np.abs(finite))), 1.0)
    tol = 1e-8 * scale
    found: list[float] = []
    for index, value in enumerate(values):
        if np.isfinite(value) and abs(float(value)) <= tol:
            found.append(float(samples[index]))
    for index in range(values.size - 1):
        left, right = float(values[index]), float(values[index + 1])
        if not np.isfinite(left) or not np.isfinite(right) or left * right >= 0:
            continue
        found.append(_bisect(f, float(samples[index]), float(samples[index + 1])))
    return _unique(found)


def _extrema(samples, values, f) -> list[float]:
    found: list[float] = []

    def slope(x: float) -> float:
        step = 1e-5 * max(1.0, abs(x))
        return f(x + step) - f(x - step)

    slopes = np.array(
        [slope(float(x)) if np.isfinite(x) else np.nan for x in samples],
        dtype=np.float64,
    )
    finite_s = slopes[np.isfinite(slopes)]
    scale = float(np.max(np.abs(finite_s))) if finite_s.size else 1.0
    tol = 1e-6 * max(scale, 1e-12)
    # A sample that lands on the vertex has slope zero, so the brackets
    # on either side do not change sign. Keep it when it is a local min or max.
    for index in range(1, int(samples.size) - 1):
        if not np.isfinite(values[index]) or not np.isfinite(slopes[index]):
            continue
        if abs(float(slopes[index])) > tol:
            continue
        left = float(values[index - 1])
        mid = float(values[index])
        right = float(values[index + 1])
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        low = (mid < left and mid <= right) or (mid <= left and mid < right)
        high = (mid > left and mid >= right) or (mid >= left and mid > right)
        if low or high:
            found.append(float(samples[index]))
    for index in range(int(samples.size) - 1):
        a = float(samples[index])
        b = float(samples[index + 1])
        if not np.isfinite(values[index]) or not np.isfinite(values[index + 1]):
            continue
        da, db = float(slopes[index]), float(slopes[index + 1])
        if not np.isfinite(da) or not np.isfinite(db) or da * db >= 0:
            continue
        if abs(da) <= tol or abs(db) <= tol:
            continue
        found.append(_bisect(slope, a, b))
    return _unique(found)


def _bisect(f, lo: float, hi: float) -> float:
    a, b = lo, hi
    fa, fb = f(a), f(b)
    if not np.isfinite(fa) or not np.isfinite(fb):
        return 0.5 * (lo + hi)
    for _ in range(60):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if not np.isfinite(fm) or abs(b - a) < 1e-12 * max(1.0, abs(mid)):
            return mid
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def _unique(values: list[float], tol: float = 1e-6) -> list[float]:
    ordered = sorted(values)
    out: list[float] = []
    for value in ordered:
        if not out or abs(value - out[-1]) > tol * max(1.0, abs(value)):
            out.append(value)
    return out


def _interp(samples, values, at: float) -> float:
    """Value of the sampled series at ``at`` along the sample axis."""
    if at <= float(samples[0]):
        return float(values[0])
    if at >= float(samples[-1]):
        return float(values[-1])
    index = int(np.searchsorted(samples, at))
    left, right = float(samples[index - 1]), float(samples[index])
    span = right - left
    if span == 0:
        return float(values[index])
    weight = (at - left) / span
    return float(values[index - 1] + weight * (values[index] - values[index - 1]))


def _crossings(left, right) -> list[tuple[float, float]]:
    ax = np.asarray(left.data_override["x"], dtype=np.float64)
    ay = np.asarray(left.data_override["y"], dtype=np.float64)
    bx = np.asarray(right.data_override["x"], dtype=np.float64)
    by = np.asarray(right.data_override["y"], dtype=np.float64)
    if not _increasing(ax) or not _increasing(bx):
        return []
    lo = max(float(ax[0]), float(bx[0]))
    hi = min(float(ax[-1]), float(bx[-1]))
    if hi <= lo:
        return []
    grid = _linspace(lo, hi, 401)
    diff = np.interp(grid, ax, ay) - np.interp(grid, bx, by)
    found: list[tuple[float, float]] = []
    scale = max(float(np.nanmax(np.abs(diff))), 1.0)
    for index, value in enumerate(diff):
        if np.isfinite(value) and abs(float(value)) <= 1e-8 * scale:
            x = float(grid[index])
            y = float(np.interp(x, ax, ay))
            found.append((x, y))
    for index in range(diff.size - 1):
        a, b = float(diff[index]), float(diff[index + 1])
        if not np.isfinite(a) or not np.isfinite(b) or a * b >= 0:
            continue
        if abs(a) <= 1e-8 * scale or abs(b) <= 1e-8 * scale:
            continue
        weight = abs(a) / (abs(a) + abs(b))
        x = float(grid[index] + weight * (grid[index + 1] - grid[index]))
        y = float(np.interp(x, ax, ay))
        found.append((x, y))
    unique: list[tuple[float, float]] = []
    for point in found:
        if any(abs(point[0] - kept[0]) <= 1e-5 * max(1.0, abs(point[0])) for kept in unique):
            continue
        unique.append(point)
    return unique


def _increasing(values: np.ndarray) -> bool:
    if values.size < 2:
        return False
    delta = np.diff(values)
    good = delta[np.isfinite(delta)]
    return bool(good.size) and bool(np.all(good > 0))


def _is_function_curve(layer) -> bool:
    if not getattr(layer, "_formula_primary", False):
        return False
    if getattr(layer, "kind", "") not in {"line", ""}:
        return False
    frame = getattr(layer, "data_override", None)
    if frame is None or "x" not in getattr(frame, "columns", []):
        return False
    if "z" in frame.columns:
        return False
    return True


def _arrows(xx, yy, dx, dy, length: float):
    rows: list[tuple[float, float]] = []
    groups: list[list[int]] = []
    flat_x = np.asarray(xx, dtype=np.float64).ravel()
    flat_y = np.asarray(yy, dtype=np.float64).ravel()
    flat_dx = np.asarray(dx, dtype=np.float64).ravel()
    flat_dy = np.asarray(dy, dtype=np.float64).ravel()
    for x, y, vx, vy in zip(flat_x, flat_y, flat_dx, flat_dy):
        arrow = _arrow(float(x), float(y), float(vx), float(vy), length)
        if arrow is None:
            continue
        start = len(rows)
        rows.extend(arrow)
        groups.append([start, len(arrow)])
    return rows, groups


def _arrow(x, y, dx, dy, length: float):
    mag = float(np.hypot(dx, dy))
    if not np.isfinite(mag) or mag == 0.0 or not np.isfinite(x) or not np.isfinite(y):
        return None
    ux, uy = dx / mag, dy / mag
    tail = (x - 0.5 * length * ux, y - 0.5 * length * uy)
    head = (x + 0.5 * length * ux, y + 0.5 * length * uy)
    back = 0.28 * length
    wing = 0.16 * length
    bx = head[0] - back * ux
    by = head[1] - back * uy
    barb1 = (bx - wing * uy, by + wing * ux)
    barb2 = (bx + wing * uy, by - wing * ux)
    return [tail, head, barb1, head, barb2]


def _streamlines(formula, xs, ys, cell, xlim, ylim):
    stride = max(1, int(len(xs) / 6))
    seeds = [
        (float(xs[i]), float(ys[j]))
        for j in range(0, len(ys), stride)
        for i in range(0, len(xs), stride)
    ]
    step = 0.45 * cell
    rows: list[tuple[float, float]] = []
    groups: list[list[int]] = []
    for sx, sy in seeds:
        for direction in (1.0, -1.0):
            points = [(sx, sy)]
            x, y = sx, sy
            for _ in range(48):
                vx, vy = _field_at(formula, x, y)
                mag = float(np.hypot(vx, vy))
                if not np.isfinite(mag) or mag < 1e-12:
                    break
                x += direction * step * vx / mag
                y += direction * step * vy / mag
                if x < xlim[0] or x > xlim[1] or y < ylim[0] or y > ylim[1]:
                    break
                points.append((x, y))
            if len(points) < 2:
                continue
            start = len(rows)
            rows.extend(points)
            groups.append([start, len(points)])
    return rows, groups


def _field_at(formula, x: float, y: float) -> tuple[float, float]:
    env = {"x": np.array([x]), "y": np.array([y])}
    dx = float(np.asarray(_component(formula, "dx", env)).reshape(-1)[0])
    dy = float(np.asarray(_component(formula, "dy", env)).reshape(-1)[0])
    return dx, dy


def _component(formula: Formula, name: str, variables: dict[str, np.ndarray]):
    code = dict(formula.components)[name]
    env: dict[str, Any] = {"__builtins__": {}}
    env.update(formula.namespace)
    env.update(variables)
    with np.errstate(all="ignore"):
        try:
            value = eval(code, env)  # noqa: S307
        except Exception as exc:
            raise ExprError(f"could not evaluate formula: {exc}") from exc
    return np.asarray(value, dtype=np.float64)


def _eval_tree(formula: Formula, tree, variables: dict[str, np.ndarray]):
    code = _compile_tree(tree)
    env: dict[str, Any] = {"__builtins__": {}}
    env.update(formula.namespace)
    env.update(variables)
    with np.errstate(all="ignore"):
        value = eval(code, env)  # noqa: S307
    return np.asarray(value, dtype=np.float64)


def _scalar(formula, axes, x: float) -> float:
    values = _curve_values(formula, axes, np.array([x], dtype=np.float64))
    return float(np.asarray(values, dtype=np.float64).reshape(-1)[0])


def _scalar_swapped(formula, axes, y: float) -> float:
    """Value of a sideways ``x = f(y)`` at one y."""
    del axes
    if formula.variables:
        values = _call_formula(formula, {formula.variables[0]: np.array([y])})
    else:
        values = evaluate(formula, {})
    return float(np.asarray(values, dtype=np.float64).reshape(-1)[0])


def _slope(formula, axes, x0: float) -> float:
    var = formula.variables[0] if formula.variables else "x"
    tree = differentiate(formula, var)
    if tree is not None:
        value = _eval_tree(formula, tree, {var: np.array([x0])})
        return float(np.asarray(value).reshape(-1)[0])
    step = 1e-5 * max(1.0, abs(x0))
    return (_scalar(formula, axes, x0 + step) - _scalar(formula, axes, x0 - step)) / (2 * step)


def _numeric_derivative(formula, axes, xs: np.ndarray) -> np.ndarray:
    span = float(xs[-1] - xs[0]) if xs.size else 1.0
    step = max(span / max(xs.size, 2) * 0.25, 1e-5)
    above = _curve_values(formula, axes, xs + step)
    below = _curve_values(formula, axes, xs - step)
    return (above - below) / (2 * step)


def _prime_texts(formula: Formula, tree) -> dict[str, str]:
    dependent = formula.dependent or "y"
    raw = formula_texts(
        __import__("ast").Name(id=dependent, ctx=__import__("ast").Load()),
        tree,
        mode="explicit",
        dependent=dependent,
    )
    primed = {}
    for key, text in raw.items():
        primed[key] = text.replace(f"{dependent} =", f"{dependent}' =", 1)
    return primed


def _simpson(xs: np.ndarray, ys: np.ndarray) -> float:
    intervals = int(ys.size - 1)
    if intervals < 2 or intervals % 2 == 1:
        raise ExprError("area() needs an even number of steps")
    step = (float(xs[-1]) - float(xs[0])) / intervals
    total = (
        float(ys[0])
        + float(ys[-1])
        + 4.0 * float(np.sum(ys[1:-1:2]))
        + 2.0 * float(np.sum(ys[2:-1:2]))
    )
    return step / 3.0 * total


def _trap(values: np.ndarray, step: float) -> float:
    if values.size == 1:
        return float(values[0])
    return float(step * (0.5 * values[0] + 0.5 * values[-1] + np.sum(values[1:-1])))


def _centroid(xs, ys, baseline: float) -> tuple[float, float]:
    step = float(xs[1] - xs[0])
    height = ys - baseline
    weight = np.abs(height)
    mass = _trap(weight, step)
    if mass < 1e-14:
        return float(0.5 * (xs[0] + xs[-1])), float(baseline)
    xc = _trap(xs * weight, step) / mass
    yc = _trap((baseline + 0.5 * height) * weight, step) / mass
    return float(xc), float(yc)


def _curve_peak(primary) -> float | None:
    frame = getattr(primary, "data_override", None)
    if frame is None or "y" not in frame:
        return None
    ys = np.asarray(frame["y"], dtype=np.float64)
    ys = ys[np.isfinite(ys)]
    if ys.size == 0 or float(ys.max()) <= 0.0:
        return None
    return float(ys.max())


def _curve_edges(primary) -> tuple[float | None, float | None]:
    frame = getattr(primary, "data_override", None)
    if frame is None or "x" not in frame:
        return None, None
    xs = np.asarray(frame["x"], dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return None, None
    return float(xs.min()), float(xs.max())


def _is_density(formula, axes, lo, hi, count) -> bool:
    """Non-negative and integrates to 1, counting the tails past the view."""
    if lo is None or hi is None or hi <= lo:
        return False
    xs = _linspace(lo, hi, count)
    ys = _curve_values(formula, axes, xs)
    if ys.shape != xs.shape or not np.all(np.isfinite(ys)):
        return False
    if float(np.min(ys)) < -1e-12:
        return False
    width = hi - lo
    total = (
        _simpson(xs, ys)
        + _tail_mass(formula, axes, lo, -1.0, width)
        + _tail_mass(formula, axes, hi, 1.0, width)
    )
    return abs(total - 1.0) <= 0.005


def _tail_mass(formula, axes, edge: float, direction: float, width: float) -> float:
    """Integral from ``edge`` outward, in doubling steps, until it stops adding.

    Returns 0 for a curve that is not finite and non-negative out there,
    so an ordinary function never gains a spurious tail.
    """
    total = 0.0
    start = float(edge)
    span = max(float(width), 1e-9)
    for _step in range(40):
        stop = start + direction * span
        xs = _linspace(min(start, stop), max(start, stop), 201)
        ys = _curve_values(formula, axes, xs)
        if ys.shape != xs.shape or not np.all(np.isfinite(ys)) or float(np.min(ys)) < 0.0:
            return total
        piece = _simpson(xs, ys)
        total += piece
        if piece <= 1e-9 * max(total, 1e-12) or piece < 1e-12:
            break
        start = stop
        span *= 2.0
    return total


def _prob_number(value: float) -> str:
    value = min(max(value, 0.0), 1.0)
    if value == 0.0 or value >= 1e-3:
        text = f"{value:.3g}"
    else:
        mantissa, power = f"{value:.2e}".split("e")
        text = f"{mantissa} × 10^{int(power)}"
        return _pretty_minus(text.replace("^", "", 1).replace(
            str(int(power)), _superscript(int(power)), 1))
    return text


def _superscript(power: int) -> str:
    table = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return str(power).translate(table)


def _pretty_minus(text: str) -> str:
    return text.replace("-", "−")


def _probability_text(name: str, lo: float, hi: float, value: float) -> str:
    """P(0.2 ≤ X ≤ 0.5) = 0.546, or a one-sided P(X ≤ −1.96) = 0.025."""
    var = name.upper() if len(name) == 1 else name
    if not math.isfinite(lo) and not math.isfinite(hi):
        event = f"−∞ < {var} < ∞"
    elif not math.isfinite(lo):
        event = f"{var} ≤ {_pretty_minus(f'{hi:.4g}')}"
    elif not math.isfinite(hi):
        event = f"{var} ≥ {_pretty_minus(f'{lo:.4g}')}"
    else:
        event = f"{_pretty_minus(f'{lo:.4g}')} ≤ {var} ≤ {_pretty_minus(f'{hi:.4g}')}"
    return f"P({event}) = {_prob_number(value)}"


def _integral_text(value: float) -> str:
    """A short fraction such as 8/3, or a decimal when the fraction is not simple.

    Simpson's error on a Beta density is within 1e-6 of 1965/3599, which is
    not an exact result. Denominators up to 12 keep 1/2, 1/3, and 8/3.
    """
    frac = Fraction(value).limit_denominator(12)
    close = abs(float(frac) - value) <= 1e-6 * max(1.0, abs(value))
    if close:
        if frac.denominator == 1:
            body = str(frac.numerator)
        else:
            body = f"{frac.numerator}/{frac.denominator}"
    else:
        body = f"{value:.4g}"
    return f"∫ = {body.replace('-', '−')}"


def _runs(index: np.ndarray) -> list[list[int]]:
    if index.size == 0:
        return []
    groups = []
    start = 0
    for position in range(1, int(index.size)):
        if int(index[position]) != int(index[position - 1]) + 1:
            groups.append([start, position - start])
            start = position
    groups.append([start, int(index.size) - start])
    return groups


def _pair(value, name: str, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ExprError(f"{name} must be a pair of numbers, for example (-2, 2)") from exc
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        hi = lo + 1.0
    return lo, hi


def _limit_or_none(value, name: str):
    if value is None:
        return None
    return _pair(value, name, _DEFAULT_DOMAIN)


def _num(value: float) -> str:
    return f"{value:.6g}"
