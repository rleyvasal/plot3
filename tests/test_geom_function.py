"""geom_function curves, surfaces, and implicit contours."""

from __future__ import annotations

import ast
import math

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes,
    coord_3d,
    coord_equal,
    geom_function,
    geom_point,
    ggplot,
    labs,
    transition_time,
)
from plot3.build import build_spec, expand_stat_geom
from plot3.masking import (
    Plot3MaskTransformer,
    apply_masking,
    default_known_names,
    plot3_backtick_transform,
)


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
    # One function: the formula is the title, not a one-row legend.
    assert spec["legend"] is None
    assert spec["labs"]["title"].startswith("v =")
    assert "^{" not in spec["labs"]["title"]
    assert spec["math"] is True


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
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = _layer("y = tan(x)", xlim=(-3, 3), n=401)
    assert len(out._groups) > 1
    assert float(np.max(np.abs(out.data_override["y"]))) < 500
    assert out._notes[0].startswith("y clipped to [")
    assert "pass ylim= to change" in out._notes[0]

    spec, _ = build_spec(
        ggplot() + geom_function("y = tan(x)", xlim=(-3, 3), n=201)
    )
    assert spec["notes"][0].startswith("y clipped to [")
    assert "ylim=" in spec["notes"][0]

    locked, _ = build_spec(
        ggplot() + geom_function("y = tan(x)", xlim=(-3, 3), ylim=(-10, 10), n=201)
    )
    assert locked["scales"]["y"]["lo"] == pytest.approx(-10)
    assert locked["scales"]["y"]["hi"] == pytest.approx(10)
    assert locked["notes"] == []


def test_clip_note_is_in_the_figure_not_on_stderr(capsys):
    html = (ggplot() + geom_function("y = 1/x", n=40)).html()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert "y clipped to [" in html
    assert "pass ylim= to change" in html
    assert 'id="note"' in html


def test_status_line_is_opt_in_and_save_still_prints(capsys, monkeypatch, tmp_path):
    fig = ggplot() + geom_function("y = x", xlim=(0, 1), n=4)
    monkeypatch.delenv("PLOT3_VERBOSE", raising=False)
    fig.save(tmp_path / "fig.html")
    saved = capsys.readouterr().out
    assert "saved" in saved
    assert "layer(s)" not in saved

    monkeypatch.setenv("PLOT3_VERBOSE", "1")
    fig.html()
    logged = capsys.readouterr().out
    assert logged.count("layer(s)") == 1


def test_show_builds_once_and_hints_once(monkeypatch, tmp_path):
    import importlib
    import sys
    import types

    gg = importlib.import_module("plot3.ggplot")

    gg._BLANK_HINT_SHOWN = False
    fig = ggplot() + geom_function("y = x", xlim=(0, 1), n=4)
    calls = {"n": 0}
    real_html = fig.html

    def counting():
        calls["n"] += 1
        return real_html()

    monkeypatch.setattr(fig, "html", counting)
    shown: list[str] = []

    class HTML:
        def __init__(self, data):
            self.data = data

    def display(obj):
        shown.append(obj.data)

    html_mod = types.ModuleType("IPython.display")
    html_mod.HTML = HTML
    html_mod.display = display
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", html_mod)
    monkeypatch.setattr(gg, "_in_solveit", lambda: False)
    monkeypatch.setattr(gg.webbrowser, "open", lambda *_a, **_k: None)
    monkeypatch.delenv("PLOT3_DISPLAY", raising=False)

    fig.show(browser=False, path=tmp_path / "a.html")
    assert calls["n"] == 1
    assert shown[0].startswith('<div class="plot3-fig"><iframe')
    assert sum("blank" in item for item in shown) == 1

    shown.clear()
    fig.show(browser=False, path=tmp_path / "b.html")
    assert calls["n"] == 2
    assert len(shown) == 1
    assert "blank" not in shown[0]

    gg._BLANK_HINT_SHOWN = False
    monkeypatch.setenv("PLOT3_DISPLAY", "iframe")
    shown.clear()
    fig.show(browser=False, path=tmp_path / "c.html")
    assert calls["n"] == 3
    assert len(shown) == 1
    assert shown[0].startswith('<div class="plot3-fig"><iframe')
    assert "blank" not in shown[0]
    gg._BLANK_HINT_SHOWN = False


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


def test_implicit_circle_and_rearranged_line(contour_backend):
    del contour_backend
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


def test_coord_equal_locks_units_and_implicit_figures_use_it(contour_backend):
    del contour_backend
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


def test_default_circle_is_a_smooth_loop(contour_backend):
    del contour_backend
    # No limits: the sample window is (-10, 10). Cells the circle crosses
    # are subdivided, otherwise it would be a few dozen straight chords.
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


def test_small_loop_next_to_a_long_line_stays_smooth(contour_backend):
    del contour_backend
    # The line spans the whole window, so the circle cannot be isolated by
    # shrinking the sample domain. Refining the cells the curve crosses is
    # what keeps the loop smooth.
    out = _layer("(x^2 + y^2 - 1) * (y - x) = 0")
    x = out.data_override["x"].to_numpy()
    y = out.data_override["y"].to_numpy()
    assert float(x.min()) < -8.0
    assert float(x.max()) > 8.0
    steps = []
    for start, count in out._groups:
        seg_x = x[start : start + count]
        seg_y = y[start : start + count]
        radius = np.hypot(seg_x, seg_y)
        on_circle = np.abs(radius - 1.0) < 0.02
        angle = np.unwrap(np.arctan2(seg_y, seg_x))
        for index in range(count - 1):
            if on_circle[index] and on_circle[index + 1]:
                steps.append(abs(float(angle[index + 1] - angle[index])))
    steps = np.asarray(steps)
    assert len(steps) > 80
    assert float(steps.max()) < np.radians(4.0)


def test_notebook_caret_is_power_with_math_precedence():
    # apply_masking quotes the formula. The caret is rewritten before parse,
    # so 2*x^3 stays 2 * x**3 and (2*x)^3 stays a power of the product.
    cases = {
        "geom_function(y=2*x^3 + 3*x^3)": "2 * x ** 3 + 3 * x ** 3",
        "geom_function(y=(2*x)^3)": "(2 * x) ** 3",
        "geom_function(y=-x^2)": "-x ** 2",
    }
    for src, formula in cases.items():
        out = apply_masking(src, known=default_known_names())
        assert formula in out
        assert "^" not in out

    limited = apply_masking(
        "geom_function(z=sin(x)^2 + cos(y)^2, xlim=(0, 3))",
        known=default_known_names(),
    )
    assert "sin(x) ** 2 + cos(y) ** 2" in limited
    assert "xlim=(0, 3)" in limited
    assert "^" not in limited

    quoted = apply_masking("geom_function('y = x^2')", known=default_known_names())
    assert "x^2" in quoted
    assert "**" not in quoted

    outside = apply_masking(
        "flags = a ^ b\ngeom_function(y = x^2)",
        known=default_known_names(),
    )
    assert "a ^ b" in outside
    assert "x ** 2" in outside

    # The notebook hook used to return early when the cell had no backtick.
    lines = plot3_backtick_transform(["geom_function(y=2*x^3)\n"])
    assert "2*x**3" in "".join(lines)


def test_unrewritten_caret_is_not_guessed():
    tree = ast.parse("geom_function(y=2*x^3)")
    with pytest.raises(ValueError, match="quote the formula"):
        Plot3MaskTransformer(known=default_known_names()).visit(tree)


def test_function_surface_defaults_to_a_cube():
    spec, _ = build_spec(
        ggplot()
        + geom_function("t = 2x^3 + 3y^3", xlim=(0, 1), ylim=(0, 1), n=4)
    )
    assert spec["is3d"] is True
    assert spec["coord"]["aspect"] == "equal"

    cloud = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 2.0]})
    data_spec, _ = build_spec(
        ggplot(cloud, aes(x="x", y="y", z="z")) + geom_point()
    )
    assert data_spec["coord"]["aspect"] == "data"

    forced, _ = build_spec(
        ggplot()
        + geom_function("z = x + y", xlim=(0, 1), ylim=(0, 1), n=3)
        + coord_3d(aspect="data")
    )
    assert forced["coord"]["aspect"] == "data"


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
    assert spec["layers"][0]["constColor"] == "#b22222"  # firebrick, as hex for the viewer
    with pytest.raises(ValueError, match="no data"):
        build_spec(ggplot() + geom_point())


def test_sympy_duck_type():
    sympy = pytest.importorskip("sympy")
    x = sympy.symbols("x")
    out = _layer(sympy.sin(x), xlim=(0, np.pi), n=5)
    xs = out.data_override["x"].to_numpy()
    assert np.allclose(out.data_override["y"], np.sin(xs), atol=1e-6)


def _range(figure, axis):
    spec, _ = build_spec(figure)
    return spec["notes"], spec["scales"][axis]["lo"], spec["scales"][axis]["hi"]


@pytest.mark.parametrize(
    "formula, kwargs, top",
    [
        # Beta(5, 1) = 5x^4: steep at x = 1 but its maximum is 5.
        ("y = 5 x^4", {"xlim": (0, 1)}, 5.0),
        ("y = x^4", {}, 1e4),
        ("y = exp(x)", {}, math.exp(10)),
    ],
)
def test_steep_but_finite_curves_are_not_clipped(formula, kwargs, top):
    notes, _lo, hi = _range(ggplot() + geom_function(formula, **kwargs), "y")
    assert notes == []
    assert hi == pytest.approx(top, rel=1e-6)


@pytest.mark.parametrize(
    "formula, kwargs",
    [
        ("y = 1/x", {}),
        ("y = 1/x^2", {}),
        ("y = tan(x)", {}),
        # Beta(0.5, 0.5) is infinite at both edges.
        ("y = 1/(pi sqrt(x (1 - x)))", {"xlim": (0, 1)}),
    ],
)
def test_poles_are_still_clipped(formula, kwargs):
    notes, _lo, _hi = _range(ggplot() + geom_function(formula, **kwargs), "y")
    assert notes and notes[0].startswith("y clipped to [")


def test_surface_rounding_spike_beside_a_singularity_is_clipped():
    # x^2 - y^2 rounds to ~1e-16 near the diagonal and gives 3.6e15.
    notes, lo, hi = _range(
        ggplot()
        + geom_function("t = x*y/(x^2 - y^2)", n=21, xlim=(-3, 3), ylim=(-3, 3)),
        "z",
    )
    assert notes
    assert hi < 100 and lo > -100


def test_surface_pole_at_a_grid_point_is_clipped():
    notes, _lo, hi = _range(
        ggplot()
        + geom_function("t = 1/(x^2 + y^2)", n=21, xlim=(-3, 3), ylim=(-3, 3)),
        "z",
    )
    assert notes
    assert hi < 11


def test_animated_steep_curve_is_not_clipped():
    spec, _ = build_spec(
        ggplot() + geom_function("y = a x^4") + transition_time(a=(0.5, 2))
    )
    assert spec["notes"] == []
