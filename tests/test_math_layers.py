"""Integrals, marks, tangents, parametric curves, polar, fields, inequalities."""

from __future__ import annotations

import numpy as np
import pytest

from plot3 import (
    area,
    coord_polar,
    derivative,
    geom_function,
    geom_vector_field,
    ggplot,
    ggsave,
    slider,
    tangent,
)
from plot3.build import build_spec
from plot3.calculus import expand_vector_field, mark_intersections
from plot3.expr import ExprError
from plot3.function import expand_function
from plot3.masking import apply_masking, default_known_names
from plot3.static import _glyph_for


def _layers(fig):
    """Expanded layers, including intersection marks, before encoding."""
    grouped: dict[int, list] = {}
    for index, addon in getattr(fig, "_addons", ()) or ():
        grouped.setdefault(index, []).append(addon)
    layers = []
    for index, geom in enumerate(fig.layers):
        if getattr(geom, "kind", None) == "vector":
            layers.extend(expand_vector_field(geom, fig.transition, fig.slider))
            continue
        layers.extend(
            expand_function(
                geom,
                fig.mapping,
                None,
                {},
                fig.transition,
                fig.slider,
                fig.coord,
                grouped.get(index),
            )
        )
    return mark_intersections(layers)


def _frame(layer):
    return layer.data_override


def test_area_labels_the_integral_of_x_squared():
    fig = ggplot() + geom_function("y = x^2", xlim=(0, 2), n=21) + area(0, 2)
    spec, _payloads = build_spec(fig)
    assert spec["ann"][0]["text"] == "∫ = 8/3"
    kinds = [layer["kind"] for layer in spec["layers"]]
    assert "area" in kinds and "line" in kinds
    # The shade sits under the parabola, on the baseline y = 0.
    area_layer = next(layer for layer in spec["layers"] if layer["kind"] == "area")
    assert area_layer["y0"] == pytest.approx(0.0, abs=1e-6)


def test_area_swaps_reversed_limits_and_needs_a_curve():
    shade = area(2, 0)
    assert (shade.lo, shade.hi) == (0.0, 2.0)
    with pytest.raises(TypeError, match="follows geom_function"):
        _ = ggplot() + area(0, 2)


def test_area_rejects_a_slider():
    fig = (
        ggplot()
        + geom_function("y = a x^2")
        + area(0, 2)
        + slider(a=(0, 3))
    )
    with pytest.raises(ValueError, match=r"area\(\) cannot follow slider"):
        build_spec(fig)


def test_roots_of_a_parabola_and_of_a_square():
    roots = _layers(ggplot() + geom_function("y = x^2 - 1", mark="roots", n=101))
    marks = next(layer for layer in roots if getattr(layer, "_legend_label", None) == "roots")
    found = sorted(float(value) for value in _frame(marks)["x"])
    assert found == pytest.approx([-1.0, 1.0], abs=1e-6)

    # x^2 does not change sign, so the root at zero has to be an exact sample.
    square = _layers(ggplot() + geom_function("y = x^2", mark="roots", n=101))
    zero = next(layer for layer in square if getattr(layer, "_legend_label", None) == "roots")
    assert list(_frame(zero)["x"]) == pytest.approx([0.0], abs=1e-8)


def test_extremum_of_a_parabola_is_the_vertex():
    layers = _layers(ggplot() + geom_function("y = x^2", mark="extrema", xlim=(-2, 2), n=41))
    marks = next(layer for layer in layers if getattr(layer, "_legend_label", None) == "extrema")
    assert list(_frame(marks)["x"]) == pytest.approx([0.0], abs=1e-4)
    assert list(_frame(marks)["y"]) == pytest.approx([0.0], abs=1e-4)


def test_intersections_of_a_line_and_a_parabola():
    fig = (
        ggplot()
        + geom_function("y = x", xlim=(-2, 2), n=81)
        + geom_function("y = x^2", xlim=(-2, 2), n=81, mark="intersections")
    )
    layers = _layers(fig)
    marks = next(layer for layer in layers if getattr(layer, "_legend_label", None) == "intersections")
    found = sorted(float(value) for value in _frame(marks)["x"])
    assert found == pytest.approx([0.0, 1.0], abs=1e-4)
    spec, _payloads = build_spec(fig)
    assert any(layer["kind"] == "point" for layer in spec["layers"])
    labels = [entry["label"] for entry in spec["legend"]]
    assert "intersections" in labels


def test_tangent_at_one_has_slope_two():
    layers = _layers(ggplot() + geom_function("y = x^2", xlim=(-2, 4), n=21) + tangent(at=1))
    line = next(layer for layer in layers if getattr(layer, "_legend_label", None) == "tangent")
    frame = _frame(line)
    slope = (float(frame["y"].iloc[-1]) - float(frame["y"].iloc[0])) / (
        float(frame["x"].iloc[-1]) - float(frame["x"].iloc[0])
    )
    assert slope == pytest.approx(2.0)
    # The line is y = 2x - 1, so it passes through the contact (1, 1).
    at_contact = np.interp(1.0, frame["x"], frame["y"])
    assert at_contact == pytest.approx(1.0)
    spec, _payloads = build_spec(
        ggplot() + geom_function("y = x^2", xlim=(-2, 4), n=21) + tangent(at=1)
    )
    assert any(entry["label"] == "tangent" for entry in spec["legend"])


def test_derivative_is_the_slope_curve():
    layers = _layers(ggplot() + geom_function("y = x^2", xlim=(-2, 2), n=11) + derivative())
    curve = next(
        layer for layer in layers
        if getattr(layer, "_legend_label", "") not in {"", None}
        and layer.kind == "line"
        and not getattr(layer, "_formula_primary", False)
    )
    frame = _frame(curve)
    assert np.allclose(frame["y"], 2 * frame["x"])
    assert "y'" in curve._legend_label or "y′" in curve._legend_label


def test_parametric_circle_and_keyword_form():
    layer = _layers(ggplot() + geom_function("x = cos(t), y = sin(t)", n=64))[0]
    frame = _frame(layer)
    assert float(frame["x"].max()) == pytest.approx(1.0, abs=1e-2)
    assert float(frame["x"].min()) == pytest.approx(-1.0, abs=1e-2)
    assert float(frame["y"].max()) == pytest.approx(1.0, abs=1e-2)
    assert float(frame["y"].min()) == pytest.approx(-1.0, abs=1e-2)
    assert not np.all(np.diff(frame["x"]) >= -1e-9)
    assert layer.sort_x is False

    keyword = _layers(ggplot() + geom_function(x="cos(t)", y="sin(t)", n=64))[0]
    assert np.allclose(_frame(keyword)[["x", "y"]], frame[["x", "y"]])


def test_helix_is_a_3d_curve():
    layer = _layers(ggplot() + geom_function("x = cos(t), y = sin(t), z = t", n=32))[0]
    assert "z" in _frame(layer).columns
    spec, _payloads = build_spec(
        ggplot() + geom_function(x="cos(t)", y="sin(t)", z="t", tlim=(0, 4), n=32)
    )
    assert spec["is3d"] is True


def test_parametric_rejects_root_marks():
    with pytest.raises(ExprError, match="curve y = f\\(x\\)"):
        _layers(ggplot() + geom_function("x = cos(t), y = sin(t)", mark="roots", n=16))


def test_cardioid_passes_through_the_origin_and_two():
    fig = ggplot() + geom_function("r = 1 + cos(theta)", n=5) + coord_polar()
    spec, _payloads = build_spec(fig)
    assert spec["coord"]["aspect"] == "equal"
    assert spec["coord"]["ratio"] == 1.0
    frame = _frame(_layers(fig)[0])
    assert np.min(np.hypot(frame["x"] - 2, frame["y"])) < 1e-8
    assert np.min(np.hypot(frame["x"], frame["y"])) < 1e-8
    # The angle domain is a full turn, not the cartesian default (−10, 10).
    assert float(frame["x"].max()) == pytest.approx(2.0, abs=1e-8)


def test_vector_field_at_one_zero_points_up():
    fig = ggplot() + geom_vector_field("dx = -y, dy = x", n=5)
    spec, _payloads = build_spec(fig)
    assert spec["is3d"] is False
    assert spec["layers"][0]["kind"] == "line"
    layer = _layers(fig)[0]
    frame = _frame(layer)
    found = False
    for start, count in layer._groups:
        block = frame.iloc[start : start + count]
        tail = block.iloc[0]
        head = block.iloc[1]
        cx = 0.5 * (float(tail["x"]) + float(head["x"]))
        cy = 0.5 * (float(tail["y"]) + float(head["y"]))
        if abs(cx - 1.0) < 1e-8 and abs(cy) < 1e-8:
            assert float(head["y"]) > float(tail["y"])
            assert float(head["x"]) == pytest.approx(float(tail["x"]))
            found = True
    assert found
    # (0, 1) points along −x.
    found = False
    for start, count in layer._groups:
        block = frame.iloc[start : start + count]
        tail = block.iloc[0]
        head = block.iloc[1]
        cx = 0.5 * (float(tail["x"]) + float(head["x"]))
        cy = 0.5 * (float(tail["y"]) + float(head["y"]))
        if abs(cx) < 1e-8 and abs(cy - 1.0) < 1e-8:
            assert float(head["x"]) < float(tail["x"])
            found = True
    assert found


def test_vector_components_are_not_a_curve():
    with pytest.raises(ExprError, match="geom_vector_field"):
        geom_function("dx = -y, dy = x")


def test_inequality_shades_above_the_parabola():
    layers = _layers(ggplot() + geom_function("y > x^2", xlim=(-1, 1), n=21))
    shade = next(layer for layer in layers if layer.kind == "area")
    frame = _frame(shade)
    assert np.allclose(frame["y"], frame["x"] ** 2)
    assert shade._baseline > float(frame["y"].max())


def test_region_inequality_fills_the_unit_disk():
    layers = _layers(
        ggplot() + geom_function("x^2 + y^2 < 1", xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), n=12)
    )
    shade = next(layer for layer in layers if layer.kind == "poly")
    frame = _frame(shade)
    # Cell centers inside the disk are filled; corners of those cells stay near it.
    assert float(np.min(frame["x"] ** 2 + frame["y"] ** 2)) < 1.0
    assert float(np.max(frame["x"] ** 2 + frame["y"] ** 2)) < 2.5


def test_where_and_latex_cases_sample_the_same_piecewise_curve():
    where_layer = expand_function(
        geom_function("y = where(x < 0, 0, x^2)", xlim=(-2, 2), n=5),
        None,
        None,
    )[0]
    cases = (
        r"y = \begin{cases} 0 & x < 0 \\ x^{2} & x \ge 0 \end{cases}"
    )
    cases_layer = expand_function(
        geom_function(cases, xlim=(-2, 2), n=5),
        None,
        None,
    )[0]
    expect = [0.0, 0.0, 0.0, 1.0, 4.0]
    assert list(_frame(where_layer)["y"]) == pytest.approx(expect)
    assert list(_frame(cases_layer)["y"]) == pytest.approx(expect)


def test_ggsave_draws_the_integral_label(tmp_path):
    fig = ggplot() + geom_function("y = x^2", xlim=(0, 2), n=21) + area(0, 2)
    svg = tmp_path / "area.svg"
    png = tmp_path / "area.png"
    ggsave(svg, fig)
    ggsave(png, fig)
    text = svg.read_text(encoding="utf-8")
    assert "<polygon" in text
    assert "∫ = 8/3" in text
    assert png.stat().st_size > 1000
    glyph, _kind = _glyph_for("∫")
    question, _qkind = _glyph_for("?")
    assert glyph != question


def test_mark_name_and_notebook_quoting():
    with pytest.raises(ValueError, match="mark must be"):
        geom_function("y = x", mark="peaks")
    known = default_known_names()
    parametric = apply_masking(
        "geom_function(x=cos(t), y=sin(t))", known=known
    )
    assert "cos(t)" in parametric.replace(" ", "")
    assert "sin(t)" in parametric.replace(" ", "")
    bound = apply_masking("geom_function(y=x**2, z=1)", known=known)
    assert "z = 1" in bound or "z=1" in bound
    assert "**" in bound
    field = apply_masking("geom_vector_field(dx=-y^2, dy=x)", known=known)
    assert "**" in field
    assert "^" not in field
    marked = apply_masking("geom_function(y=x**2, mark=roots)", known=known)
    assert "roots" in marked
    added = apply_masking(
        "geom_function(y=x**2) + area(0, 2) + tangent(at=1)", known=known
    )
    assert "area" in added and "tangent" in added
    assert "at" in added


def test_streamlines_draw():
    layer = _layers(
        ggplot() + geom_vector_field("dx = -y, dy = x", n=7, stream=True)
    )[0]
    assert layer.kind == "line"
    assert len(_frame(layer)) > 4


def test_streamlines_stay_apart_and_close_orbits():
    layers = _layers(ggplot() + geom_vector_field("dx = -y, dy = x", stream=True))
    frame = _frame(layers[0])
    groups = layers[0]._groups
    pts = frame[["x", "y"]].to_numpy()
    lines = [pts[start : start + count] for start, count in groups if count > 3]
    assert 3 <= len(lines) <= 30
    # Rings around the centre close on themselves instead of wrapping again.
    closing = [np.hypot(*(line[-1] - line[0])) for line in lines]
    assert sum(gap < 0.1 for gap in closing) >= 3
    # No two lines run on top of each other.
    cell = 4.0 / 10
    for a in range(len(lines)):
        for b in range(a + 1, len(lines)):
            d = np.hypot(*(lines[a][:, None, :] - lines[b][None, :, :]).transpose(2, 0, 1))
            assert d.min() > 0.35 * cell
