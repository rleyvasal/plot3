"""geom_function curves, surfaces, and implicit contours."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from plot3 import aes, coord_equal, geom_function, geom_point, ggplot
from plot3.build import build_spec, expand_stat_geom


def _layer(expr, **kw):
    return expand_stat_geom(geom_function(expr, **kw), aes(), None)


def test_line_samples_match_formula():
    out = _layer("y = 2x", xlim=(0, 4), n=5)
    frame = out.data_override
    assert np.allclose(frame["y"], 2 * frame["x"])
    assert list(frame["x"]) == pytest.approx([0, 1, 2, 3, 4])


def test_constant_and_bare_expression():
    flat = _layer("y = 4", xlim=(-1, 1), n=4)
    assert np.allclose(flat.data_override["y"], 4)
    bare = _layer("x^2 + 1", xlim=(0, 2), n=3)
    assert np.allclose(bare.data_override["y"], bare.data_override["x"] ** 2 + 1)
    assert bare._axis_labels["y"] == "y"


def test_parameters_and_user_function():
    out = _layer("y = a x^2 + b", a=2, b=-1, xlim=(0, 2), n=3)
    xs = out.data_override["x"].to_numpy()
    assert np.allclose(out.data_override["y"], 2 * xs**2 - 1)

    def damp(t):
        return math.exp(-float(t) / 5)

    with pytest.warns(UserWarning, match="vectorize"):
        waved = _layer("y = damp(x)", damp=damp, xlim=(0, 1), n=4)
    assert np.allclose(
        waved.data_override["y"],
        np.exp(-waved.data_override["x"] / 5),
    )


def test_lambda_and_variable_names():
    out = _layer(lambda x: x**2, xlim=(-2, 2), n=5)
    xs = out.data_override["x"].to_numpy()
    assert np.allclose(out.data_override["y"], xs**2)

    named = _layer("v = 9.8 t", xlim=(0, 1), n=3)
    assert named._axis_labels == {"x": "t", "y": "v"}
    spec, _ = build_spec(ggplot() + geom_function("v = 9.8 t", xlim=(0, 1), n=11))
    assert spec["labs"]["x"] == "t"
    assert spec["labs"]["y"] == "v"
    assert spec["is3d"] is False
    assert spec["legend"][0]["label"].startswith("v =")


def test_ggplot_without_data_and_overlay_domain():
    spec, _ = build_spec(ggplot() + geom_function("y = x", xlim=(-2, 2), n=11))
    assert spec["layers"][0]["kind"] == "line"
    assert spec["scales"]["x"]["lo"] == pytest.approx(-2)
    assert spec["scales"]["x"]["hi"] == pytest.approx(2)
    html = (ggplot() + geom_function("y = sin(x)", n=21)).html()
    assert "three" in html.lower() or "WebGL" in html

    df = pd.DataFrame({"wt": [1.0, 5.0], "mpg": [10.0, 30.0]})
    over = (
        ggplot(df, aes(x="wt", y="mpg"))
        + geom_point()
        + geom_function("y = 2x", n=5)
    )
    spec, _ = build_spec(over)
    assert spec["layers"][1]["kind"] == "line"
    assert spec["scales"]["x"]["lo"] <= 1
    assert spec["scales"]["x"]["hi"] >= 5
    assert spec["labs"]["x"] == "wt"


def test_log_domain_starts_where_defined():
    spec, _ = build_spec(ggplot() + geom_function("y = log(x)", n=101))
    assert spec["scales"]["x"]["lo"] > 0


def test_clipping_breaks_the_line():
    with pytest.warns(UserWarning, match="clipped"):
        out = _layer("y = tan(x)", xlim=(-3, 3), n=401)
    assert len(out._groups) > 1
    assert float(np.max(np.abs(out.data_override["y"]))) < 500

    spec, _ = build_spec(
        ggplot() + geom_function("y = tan(x)", xlim=(-3, 3), ylim=(-10, 10), n=201)
    )
    assert spec["scales"]["y"]["lo"] == pytest.approx(-10)
    assert spec["scales"]["y"]["hi"] == pytest.approx(10)


def test_surface_and_sideways_parabola():
    out = _layer("z = x + 2y", xlim=(0, 1), ylim=(0, 1), n=3)
    assert out.kind == "surface"
    frame = out.data_override
    assert np.allclose(frame["z"], frame["x"] + 2 * frame["y"])
    spec, _ = build_spec(
        ggplot() + geom_function("z = sin(x) cos(y)", xlim=(-1, 1), ylim=(-1, 1), n=6)
    )
    assert spec["is3d"] is True
    assert spec["layers"][0]["kind"] == "surface"

    side = _layer("x = y^2", ylim=(-2, 2), n=5)
    ys = side.data_override["y"].to_numpy()
    xs = side.data_override["x"].to_numpy()
    assert np.allclose(xs, ys**2)
    assert list(ys) == pytest.approx([-2, -1, 0, 1, 2])


def test_implicit_circle_and_rearranged_line():
    circle = _layer("x^2 + y^2 = 1", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), n=48)
    pts = circle.data_override
    radius = np.sqrt(pts["x"] ** 2 + pts["y"] ** 2)
    assert len(pts) > 20
    assert np.median(np.abs(radius - 1)) < 0.05

    line = _layer("y = 2x + 2y", xlim=(-2, 2), ylim=(-4, 4), n=31)
    pts = line.data_override
    assert np.median(np.abs(pts["y"] + 2 * pts["x"])) < 0.15


def _equal_spans(width, height, x_span, y_span, ratio=1.0, pad=0.03):
    """Camera spans (normalized) that the 2D viewer uses for coord_equal."""
    target = (width / height) * (y_span / x_span) / ratio
    need = 1 + 2 * pad
    nx = ny = need
    if nx / ny < target:
        nx = ny * target
    else:
        ny = nx / target
    return nx, ny


def test_coord_equal_locks_units_and_implicit_figures_use_it():
    circle, _ = build_spec(ggplot() + geom_function("x^2 + y^2 = 1"))
    assert circle["coord"] == {"aspect": "equal", "ratio": 1.0}
    # Wide panel, equal data spans: x shows more range so the circle stays round.
    nx, ny = _equal_spans(900, 400, 2.0, 2.0)
    assert nx / ny == pytest.approx(900 / 400)
    assert 900 / (nx * 2.0) == pytest.approx(400 / (ny * 2.0))
    # ratio=2: one x unit is as long on screen as two y units.
    nx, ny = _equal_spans(800, 800, 4.0, 2.0, ratio=2.0)
    px_x = 800 / (nx * 4.0)
    px_y = 800 / (ny * 2.0)
    assert px_x == pytest.approx(2 * px_y)

    free, _ = build_spec(ggplot() + geom_function("y = x", xlim=(0, 1), n=4))
    assert free["coord"] is None

    forced, _ = build_spec(
        ggplot() + geom_function("y = x", xlim=(0, 1), n=4) + coord_equal(ratio=2)
    )
    assert forced["coord"] == {"aspect": "equal", "ratio": 2.0}

    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    mixed, _ = build_spec(
        ggplot(df, aes(x="x", y="y"))
        + geom_point()
        + geom_function("x^2 + y^2 = 1")
    )
    assert mixed["coord"] is None

    with pytest.raises(ValueError, match="positive number"):
        coord_equal(0)
    cloud = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]})
    with pytest.raises(ValueError, match="coord_equal"):
        build_spec(
            ggplot(cloud, aes(x="x", y="y", z="z")) + geom_point() + coord_equal()
        )


def test_default_circle_is_a_smooth_loop():
    # No limits: the sample window starts at (-10, 10) and must tighten,
    # otherwise the unit circle is a few dozen straight chords.
    circle = _layer("x^2 + y^2 = 1")
    pts = circle.data_override
    x = pts["x"].to_numpy()
    y = pts["y"].to_numpy()
    assert len(circle._groups) == 1
    start, count = circle._groups[0]
    assert count == len(pts)
    assert np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-6
    radius = np.hypot(x, y)
    assert np.median(np.abs(radius - 1)) < 0.01
    steps = np.abs(np.diff(np.unwrap(np.arctan2(y[start:start + count], x[start:start + count]))))
    assert len(pts) > 120
    assert float(np.max(steps)) < np.radians(4.0)

    line = _layer("y = 2x + 2y")
    pts = line.data_override
    assert np.median(np.abs(pts["y"] + 2 * pts["x"])) < 0.05


def test_keyword_formula_uses_that_name_as_the_output():
    # What the notebook transformer produces for geom_function(z = x + y).
    out = expand_stat_geom(
        geom_function(z="x + y", xlim=(0, 1), ylim=(0, 1), n=3),
        aes(),
        None,
    )
    assert out.kind == "surface"
    assert out._axis_labels["z"] == "z"
    assert np.allclose(
        out.data_override["z"],
        out.data_override["x"] + out.data_override["y"],
    )


def test_function_does_not_inherit_colour_mapping():
    df = pd.DataFrame({"wt": [1.0, 2.0], "mpg": [3.0, 4.0], "cyl": ["a", "b"]})
    fig = (
        ggplot(df, aes(x="wt", y="mpg", colour="cyl"))
        + geom_point()
        + geom_function("y = x", n=4)
    )
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["kind"] == "point"
    assert spec["layers"][1]["kind"] == "line"
    assert "color" not in spec["layers"][1]


def test_color_and_missing_data_still_errors():
    spec, _ = build_spec(
        ggplot() + geom_function("y = x", color="firebrick", xlim=(0, 1), n=4)
    )
    assert spec["layers"][0]["constColor"] == "firebrick"
    with pytest.raises(ValueError, match="no data"):
        build_spec(ggplot() + geom_point())


def test_sympy_duck_type():
    sympy = pytest.importorskip("sympy")
    x = sympy.symbols("x")
    out = _layer(sympy.sin(x), xlim=(0, np.pi), n=5)
    xs = out.data_override["x"].to_numpy()
    assert np.allclose(out.data_override["y"], np.sin(xs), atol=1e-6)
