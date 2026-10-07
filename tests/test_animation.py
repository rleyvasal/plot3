"""Bubble size, log scales, and Gapminder-style transitions."""

from __future__ import annotations

import base64
import gzip
import math

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_x_log10,
    scale_y_log10,
    slider,
    transition_states,
    transition_time,
)
from plot3.build import build_doc, build_spec
from plot3.masking import apply_masking, default_known_names


def _u16(b64: str) -> np.ndarray:
    raw = gzip.decompress(base64.b64decode(b64))
    planes = np.frombuffer(raw, dtype=np.uint8)
    half = planes.size // 2
    delta = planes[:half].astype(np.int32) | (planes[half:].astype(np.int32) << 8)
    out = np.empty(half, dtype=np.int32)
    acc = 0
    for i, step in enumerate(delta):
        acc = (acc + int(step)) % 65536
        out[i] = acc
    return out


def _u8(b64: str) -> np.ndarray:
    raw = gzip.decompress(base64.b64decode(b64))
    return np.frombuffer(raw, dtype=np.uint8).copy()


def _norm(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + (q.astype(np.float64) / 65535.0) * (hi - lo)


def _blob(payloads, pid: str) -> str:
    return dict(payloads)[pid]


def test_aes_size_is_area_and_draws_large_first():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 1.0, 1.0], "pop": [1.0, 4.0, 9.0]})
    spec, payloads = build_spec(
        ggplot(df, aes(x="x", y="y", size="pop")) + geom_point()
    )
    layer = spec["layers"][0]
    assert layer["size"]["scale"] == "area"
    assert layer["size"]["vmax"] == pytest.approx(9.0)
    # sqrt(value / 9), largest bubble first: 1, 2/3, 1/3
    got = _norm(_u16(_blob(payloads, layer["size"]["id"])), 0.0, 1.0)
    assert got == pytest.approx([1.0, 2.0 / 3.0, 1.0 / 3.0], abs=1e-3)
    # x follows the same order: pop 9, 4, 1 -> x 3, 2, 1
    xs = _norm(_u16(_blob(payloads, layer["x"]["id"])), spec["scales"]["x"]["lo"], spec["scales"]["x"]["hi"])
    assert xs == pytest.approx([3.0, 2.0, 1.0], abs=1e-2)
    labels = [b["label"] for b in spec["sizeLegend"]["breaks"]]
    assert labels
    assert spec["sizeLegend"]["label"] == "pop"
    # A constant size= is still one number for the layer.
    plain, _ = build_spec(ggplot(df, aes(x="x", y="y")) + geom_point(size=5))
    assert plain["layers"][0]["size"] == 5.0
    assert plain["sizeLegend"] is None


def test_size_legend_uses_labs():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "pop": [10.0, 1000.0]})
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y", size="pop"))
        + geom_point()
        + labs(size="population")
    )
    assert spec["sizeLegend"]["label"] == "population"
    assert spec["sizeLegend"]["breaks"][-1]["t"] == pytest.approx(1.0, abs=1e-6)


def test_scale_log10_domain_and_dropped_rows():
    df = pd.DataFrame({"x": [1.0, 10.0, 100.0, -5.0], "y": [1.0, 2.0, 3.0, 4.0]})
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y")) + geom_point() + scale_x_log10()
    )
    sx = spec["scales"]["x"]
    assert sx["trans"] == "log10"
    assert sx["lo"] == pytest.approx(0.0)
    assert sx["hi"] == pytest.approx(2.0)
    assert spec["layers"][0]["n"] == 3
    assert any("non-positive" in note for note in spec["notes"])
    # Tick positions live in log space; labels are the original numbers.
    tick_pos = [t[0] for t in sx["ticks"]]
    tick_lab = [t[1] for t in sx["ticks"]]
    assert tick_pos[0] == pytest.approx(0.0)
    assert "1" in tick_lab and "100" in tick_lab

    spec_y, _ = build_spec(
        ggplot(df.iloc[:3], aes(x="y", y="x")) + geom_point() + scale_y_log10()
    )
    assert spec_y["scales"]["y"]["trans"] == "log10"
    assert spec_y["scales"]["y"]["hi"] == pytest.approx(2.0)


def test_log_scale_rejects_categories_and_non_positive_limits():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="numeric"):
        build_spec(ggplot(df, aes(x="x", y="y")) + geom_point() + scale_x_log10())


def test_transition_time_fixed_scales_and_masks():
    df = pd.DataFrame({
        "country": ["A", "A", "B", "B", "B"],
        "year": [2000, 2001, 2000, 2001, 2000],
        "gdp": [1.0, 10.0, 100.0, 10.0, 100.0],
        "life": [50.0, 60.0, 40.0, 70.0, 40.0],
        "pop": [1.0e6, 2.0e6, 4.0e6, 4.0e6, 4.0e6],
        "continent": ["N", "N", "S", "S", "S"],
    })
    # B is duplicated in 2000 — rejected.
    with pytest.raises(ValueError, match="more than one row"):
        build_spec(
            ggplot(df, aes(x="gdp", y="life", size="pop", group="country"))
            + geom_point()
            + transition_time("year")
        )
    df = df.drop(index=4)
    fig = (
        ggplot(
            df,
            aes(x="gdp", y="life", size="pop", color="continent", group="country"),
        )
        + geom_point()
        + scale_x_log10()
        + transition_time("year")
        + labs(title="Life expectancy, {frame_time}")
    )
    spec, payloads = build_spec(fig)
    tr = spec["transition"]
    assert tr["type"] == "time"
    assert tr["nFrames"] == 2
    assert tr["times"] == [2000, 2001]
    assert tr["integer"] is True
    assert tr["ease"] == "linear"
    assert spec["labs"]["title"] == "Life expectancy, {frame_time}"
    layer = spec["layers"][0]
    assert layer["n"] == 2
    # B's population is larger, so B is drawn first.
    assert layer["ids"] == ["B", "A"]
    assert layer["frames"]["nFrames"] == 2
    # Domain covers every frame, not just the last one (gdp 1 .. 100).
    assert spec["scales"]["x"]["trans"] == "log10"
    assert spec["scales"]["x"]["lo"] == pytest.approx(0.0)
    assert spec["scales"]["x"]["hi"] == pytest.approx(2.0)
    assert spec["scales"]["y"]["lo"] == pytest.approx(40.0)
    assert spec["scales"]["y"]["hi"] == pytest.approx(70.0)
    # Last frame is the static fallback: B life 70, A life 60.
    y = _norm(
        _u16(_blob(payloads, layer["y"]["id"])),
        spec["scales"]["y"]["lo"],
        spec["scales"]["y"]["hi"],
    )
    assert y == pytest.approx([70.0, 60.0], abs=0.05)

    fx = layer["frames"]["x"]
    mat = _norm(
        _u16(_blob(payloads, fx["id"])),
        spec["scales"]["x"]["lo"],
        spec["scales"]["x"]["hi"],
    ).reshape(2, 2)
    # log10(gdp): B is 2 then 1, A is 0 then 1.
    assert mat[0] == pytest.approx([2.0, 1.0], abs=0.02)
    assert mat[1] == pytest.approx([0.0, 1.0], abs=0.02)
    mask = _u8(_blob(payloads, fx["mask"])).reshape(2, 2)
    assert mask.tolist() == [[1, 1], [1, 1]]
    # Area fractions use the global max (4e6): B is 1, A is sqrt(1/4) then sqrt(1/2).
    sm = _norm(_u16(_blob(payloads, layer["frames"]["size"]["id"])), 0.0, 1.0).reshape(2, 2)
    assert sm[0] == pytest.approx([1.0, 1.0], abs=1e-3)
    assert sm[1, 0] == pytest.approx(math.sqrt(0.25), abs=1e-3)
    assert sm[1, 1] == pytest.approx(math.sqrt(0.5), abs=1e-3)


def test_transition_requires_group_and_points():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "year": [2000, 2001]})
    with pytest.raises(ValueError, match="group"):
        build_spec(
            ggplot(df, aes(x="x", y="y")) + geom_point() + transition_time("year")
        )


def test_missing_frame_is_masked():
    df = pd.DataFrame({
        "country": ["A", "A", "B"],
        "year": [2000, 2001, 2000],
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0],
    })
    spec, payloads = build_spec(
        ggplot(df, aes(x="x", y="y", group="country"))
        + geom_point()
        + transition_time("year")
    )
    layer = spec["layers"][0]
    # Alphabetical ids, no size column: A then B.
    assert layer["ids"] == ["A", "B"]
    mask = _u8(_blob(payloads, layer["frames"]["x"]["mask"])).reshape(2, 2)
    # A present both years; B missing in 2001.
    assert mask.tolist() == [[1, 1], [1, 0]]


def test_transition_states_keep_first_seen_order():
    df = pd.DataFrame({
        "id": ["p", "p", "q", "q"],
        "when": ["before", "after", "before", "after"],
        "x": [0.0, 1.0, 0.0, 1.0],
        "y": [0.0, 1.0, 1.0, 0.0],
    })
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y", group="id"))
        + geom_point()
        + transition_states("when")
    )
    tr = spec["transition"]
    assert tr["type"] == "states"
    assert tr["times"] == ["before", "after"]
    assert tr["ease"] == "smooth"
    assert spec["layers"][0]["ids"] == ["p", "q"]


def test_frame_time_token_and_player_markup():
    df = pd.DataFrame({
        "country": ["A", "A"],
        "year": [2000, 2001],
        "x": [1.0, 2.0],
        "y": [3.0, 4.0],
    })
    fig = (
        ggplot(df, aes(x="x", y="y", group="country"))
        + geom_point()
        + transition_time("year")
        + labs(title="{frame_time}")
    )
    html = build_doc(fig)
    assert 'id="play-btn"' in html
    assert 'id="year"' in html
    assert "prefers-reduced-motion" in html
    assert "captureStream" in html
    assert "{frame_time}" in html
    # The viewer template is a normal Python string. Shader injections must
    # stay JS escapes, or the newline lands inside a single-quoted string.
    assert "\\nattribute float aSize" in html
    assert "\\nvAlpha = aAlpha" in html


def test_frame_color_uses_sentinel_for_missing():
    df = pd.DataFrame({
        "id": ["A", "A", "B"],
        "year": [2000, 2001, 2000],
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0],
        "region": ["East", "West", "East"],
    })
    spec, payloads = build_spec(
        ggplot(df, aes(x="x", y="y", color="region", group="id"))
        + geom_point()
        + transition_time("year")
    )
    layer = spec["layers"][0]
    assert spec["color"]["cats"] == ["East", "West"]
    codes = _u16(_blob(payloads, layer["frames"]["color"]["id"])).reshape(2, 2)
    # A is East then West. B is East, then absent (65535, not a real category).
    assert codes.tolist() == [[0, 1], [0, 65535]]


def test_transition_rejects_non_point_layers():
    df = pd.DataFrame({
        "id": ["A", "A"],
        "year": [2000, 2001],
        "x": [1.0, 2.0],
        "y": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="geom_point"):
        build_spec(
            ggplot(df, aes(x="x", y="y", group="id"))
            + geom_line()
            + transition_time("year")
        )


def test_transition_time_spans_the_column_range():
    df = pd.DataFrame({
        "id": ["A", "A", "A"],
        "year": [2010, 2000, 2001],
        "x": [3.0, 1.0, 2.0],
        "y": [3.0, 1.0, 2.0],
    })
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y", group="id"))
        + geom_point()
        + transition_time("year")
    )
    assert spec["transition"]["times"] == [2000, 2001, 2010]
    assert spec["transition"]["integer"] is True


def test_masking_size_and_transition_column():
    import ast

    src = "transition_time(year) + aes(size=pop, group=country)"
    out = ast.unparse(ast.parse(apply_masking(src, known=default_known_names())))
    assert "year" in out and ("'year'" in out or '"year"' in out)
    assert "'pop'" in out or '"pop"' in out
    assert "'country'" in out or '"country"' in out


def test_unbound_coefficient_errors_at_build_time():
    from plot3 import geom_function
    from plot3.expr import ExprError

    layer = geom_function("y = a x^2")
    assert layer.formula.pending == ("a",)
    assert layer.formula.variables == ("x",)
    with pytest.raises(ExprError, match=r"transition_time\(a=\(0, 3\)\)") as err:
        build_spec(ggplot() + layer)
    assert "slider(a=(0, 3))" in str(err.value)


def test_function_parameter_sweep_fixes_axes():
    import json

    from plot3 import geom_function

    spec, payloads = build_spec(
        ggplot() + geom_function("y = a x^2") + transition_time(a=(0, 3))
    )
    json.dumps(spec)
    tr = spec["transition"]
    assert tr["nFrames"] == 60
    assert tr["type"] == "time"
    assert tr["params"] == [{"name": "a", "lo": 0.0, "hi": 3.0}]
    assert spec["scales"]["x"]["lo"] == pytest.approx(-10)
    assert spec["scales"]["x"]["hi"] == pytest.approx(10)
    assert spec["scales"]["y"]["lo"] == pytest.approx(0)
    assert spec["scales"]["y"]["hi"] == pytest.approx(300)
    layer = spec["layers"][0]
    assert layer["frames"]["nFrames"] == 60
    assert spec["slider"] is None
    y = _u16(_blob(payloads, layer["frames"]["y"]["id"]))
    mat = y.reshape(-1, 60)
    assert int(mat[:, 0].max()) == 0
    assert int(mat[:, -1].max()) == 65535
    # No-JS fallback stays the last frame. Sliders use the first one.
    static_y = _u16(_blob(payloads, layer["y"]["id"]))
    assert np.array_equal(static_y, mat[:, -1])


def test_function_sweep_uses_every_frame_for_the_scale():
    # The last frame is y = 0. The middle frame is y = x on [-10, 10].
    from plot3 import geom_function

    spec, _payloads = build_spec(
        ggplot()
        + geom_function("y = sin(a) x")
        + transition_time(a=(0, math.pi), frames=5)
    )
    assert spec["transition"]["nFrames"] == 5
    assert spec["scales"]["y"]["lo"] == pytest.approx(-10)
    assert spec["scales"]["y"]["hi"] == pytest.approx(10)


def test_transition_time_column_xor_ranges():
    with pytest.raises(TypeError, match="takes a column"):
        transition_time()
    with pytest.raises(TypeError, match="takes a column"):
        transition_time("year", a=(0, 3))
    swept = transition_time(a=(3, 0))
    assert swept.column is None
    assert swept.ranges["a"] == (0.0, 3.0)
    assert swept.frames == 60
    with pytest.raises(ValueError, match="at least 2"):
        transition_time(a=(0, 1), frames=1)


def test_masking_keeps_transition_ranges_as_python():
    import ast

    src = "transition_time(a=(0, 2*pi), frames=30)"
    out = ast.unparse(ast.parse(apply_masking(src, known=default_known_names())))
    assert "2 * pi" in out or "2*pi" in out
    assert "'pi'" not in out and '"pi"' not in out
    named = ast.unparse(
        ast.parse(
            apply_masking("transition_time(column=year)", known=default_known_names())
        )
    )
    assert "'year'" in named or '"year"' in named


def test_parameter_ranges_reject_point_layers():
    df = pd.DataFrame({"x": [1.0], "y": [1.0]})
    with pytest.raises(ValueError, match="geom_function"):
        build_spec(
            ggplot(df, aes(x="x", y="y"))
            + geom_point()
            + transition_time(a=(0, 1))
        )


def test_implicit_parameter_sweep_steps():
    from plot3 import geom_function

    spec, _payloads = build_spec(
        ggplot()
        + geom_function("x^2 + y^2 = a", n=20)
        + transition_time(a=(1, 4), frames=3)
    )
    frames = spec["layers"][0]["frames"]
    assert frames["mode"] == "step"
    assert frames["nFrames"] == 3
    counts = [span[1] for span in frames["spans"]]
    assert all(count >= 2 for count in counts)
    assert len(set(counts)) > 1


def test_surface_parameter_sweep_uses_middle_frames():
    from plot3 import geom_function

    spec, _payloads = build_spec(
        ggplot()
        + geom_function("z = sin(a) sin(x) cos(y)", n=8)
        + transition_time(a=(0, math.pi), frames=5)
    )
    assert spec["is3d"] is True
    assert spec["layers"][0]["kind"] == "surface"
    assert spec["layers"][0]["frames"]["nFrames"] == 5
    # Quiet frames must not pull the robust window in and clip the wave.
    assert not spec.get("notes")
    assert spec["scales"]["z"]["lo"] == pytest.approx(-0.83, abs=0.05)
    assert spec["scales"]["z"]["hi"] == pytest.approx(0.83, abs=0.05)


def test_parameter_sweep_still_clips_poles():
    from plot3 import geom_function

    spec, _payloads = build_spec(
        ggplot()
        + geom_function("y = a tan(x)", xlim=(-3, 3), n=201)
        + transition_time(a=(0, 1), frames=5)
    )
    assert spec["notes"][0].startswith("y clipped to [")
    assert spec["scales"]["y"]["hi"] == pytest.approx(12.46, abs=0.1)
    assert spec["layers"][0]["frames"]["nFrames"] == 5


def test_slider_requires_ranges_and_steps():
    with pytest.raises(TypeError, match="parameter range"):
        slider()
    with pytest.raises(ValueError, match="at least 2"):
        slider(a=(0, 1), steps=1)
    with pytest.raises(TypeError, match=r"slider\(\) range"):
        slider(a=3)
    with pytest.raises(TypeError, match="steps="):
        slider(a=(0, 1), steps=(0, 1))


def test_slider_cannot_combine_with_a_transition():
    with pytest.raises(ValueError, match="cannot be combined"):
        ggplot() + slider(a=(0, 1)) + transition_time(a=(0, 1))
    with pytest.raises(ValueError, match="cannot be combined"):
        ggplot() + transition_time(a=(0, 1)) + slider(a=(0, 1))
    with pytest.raises(ValueError, match="cannot be combined"):
        ggplot() + slider(a=(0, 1)) + transition_states("year")
    with pytest.raises(ValueError, match="cannot be combined"):
        ggplot() + transition_states("year") + slider(a=(0, 1))


def test_masking_keeps_slider_ranges_as_python():
    import ast

    src = "slider(a=(0, 2*pi), steps=10)"
    out = ast.unparse(ast.parse(apply_masking(src, known=default_known_names())))
    assert "2 * pi" in out or "2*pi" in out
    assert "'pi'" not in out and '"pi"' not in out
    assert "slider" in default_known_names()


def test_slider_one_parameter_opens_at_the_low_end():
    import json

    from plot3 import geom_function
    from plot3.payload import build_payload, validate_payload

    fig = ggplot() + geom_function("y = a x^2") + slider(a=(0, 3))
    spec, payloads = build_spec(fig)
    json.dumps(spec)
    assert spec["transition"] is None
    knob = spec["slider"]
    assert knob["nFrames"] == 25
    assert knob["params"] == [
        {"name": "a", "lo": 0.0, "hi": 3.0, "n": 25, "stride": 1}
    ]
    assert spec["scales"]["x"]["lo"] == pytest.approx(-10)
    assert spec["scales"]["x"]["hi"] == pytest.approx(10)
    assert spec["scales"]["y"]["lo"] == pytest.approx(0)
    assert spec["scales"]["y"]["hi"] == pytest.approx(300)
    layer = spec["layers"][0]
    assert layer["frames"]["nFrames"] == 25
    assert "mode" not in layer["frames"]
    y = _u16(_blob(payloads, layer["frames"]["y"]["id"])).reshape(-1, 25)
    assert int(y[:, 0].max()) == 0
    assert int(y[:, -1].max()) == 65535
    static_y = _u16(_blob(payloads, layer["y"]["id"]))
    assert np.array_equal(static_y, y[:, 0])
    payload = validate_payload(build_payload(fig))
    assert payload["spec"]["slider"]["nFrames"] == 25
    assert payload["spec"]["transition"] is None


def test_slider_grid_is_independent_per_parameter():
    import json

    from plot3 import geom_function

    spec, payloads = build_spec(
        ggplot()
        + geom_function("y = a sin(k x)")
        + slider(a=(0, 3), k=(1, 5), steps=5)
    )
    json.dumps(spec)
    knob = spec["slider"]
    assert knob["nFrames"] == 25
    assert [(p["name"], p["n"], p["stride"]) for p in knob["params"]] == [
        ("a", 5, 5),
        ("k", 5, 1),
    ]
    assert knob["params"][0]["lo"] == pytest.approx(0)
    assert knob["params"][0]["hi"] == pytest.approx(3)
    assert knob["params"][1]["lo"] == pytest.approx(1)
    assert knob["params"][1]["hi"] == pytest.approx(5)
    assert spec["scales"]["y"]["lo"] == pytest.approx(-3, abs=0.05)
    assert spec["scales"]["y"]["hi"] == pytest.approx(3, abs=0.05)
    layer = spec["layers"][0]
    y = _u16(_blob(payloads, layer["frames"]["y"]["id"])).reshape(-1, 25)
    # a = 0 along the whole k axis: a flat line.
    assert int(y[:, 0].max()) - int(y[:, 0].min()) <= 1
    assert int(y[:, 4].max()) - int(y[:, 4].min()) <= 1
    slow = y[:, 20].astype(np.float64)
    fast = y[:, 24].astype(np.float64)
    assert int(slow.max()) > 60000 and int(slow.min()) < 5000
    assert int(fast.max()) > 60000 and int(fast.min()) < 5000
    assert np.corrcoef(slow, fast)[0, 1] < 0.95
    x = _u16(_blob(payloads, layer["frames"]["x"]["id"])).reshape(-1, 25)
    assert np.all(x == x[:, :1])


def test_slider_overrides_a_bound_coefficient():
    from plot3 import geom_function

    spec, payloads = build_spec(
        ggplot()
        + geom_function("y = a x^2", a=2)
        + slider(a=(0, 3), steps=3)
    )
    assert spec["slider"]["nFrames"] == 3
    y = _u16(_blob(payloads, spec["layers"][0]["frames"]["y"]["id"]))
    y = y.reshape(-1, 3)
    # Frame 0 is a = 0, not the bound value 2.
    assert int(y[:, 0].max()) == 0
    assert int(y[:, -1].max()) == 65535
    static_y = _u16(_blob(payloads, spec["layers"][0]["y"]["id"]))
    assert np.array_equal(static_y, y[:, 0])


def test_slider_rejects_parameters_that_are_not_coefficients():
    from plot3 import geom_function

    with pytest.raises(ValueError, match="not coefficients"):
        build_spec(ggplot() + geom_function("y = x^2") + slider(a=(0, 1)))


def test_slider_rejects_point_layers():
    df = pd.DataFrame({"x": [1.0], "y": [1.0]})
    with pytest.raises(ValueError, match="geom_function"):
        build_spec(
            ggplot(df, aes(x="x", y="y")) + geom_point() + slider(a=(0, 1))
        )


def test_implicit_slider_snaps_and_opens_on_the_first_frame():
    from plot3 import geom_function

    spec, payloads = build_spec(
        ggplot()
        + geom_function("x^2 + y^2 = a", n=20)
        + slider(a=(1, 16), steps=3)
    )
    frames = spec["layers"][0]["frames"]
    assert frames["mode"] == "step"
    assert frames["nFrames"] == 3
    counts = [span[1] for span in frames["spans"]]
    assert all(count >= 2 for count in counts)
    assert counts[0] < counts[-1]
    fx = _u16(_blob(payloads, frames["x"]["id"]))
    sx = _u16(_blob(payloads, spec["layers"][0]["x"]["id"]))
    start, count = frames["spans"][0]
    assert np.array_equal(sx, fx[start:start + count])


def test_slider_surface_keeps_a_finite_wave():
    from plot3 import geom_function

    spec, _payloads = build_spec(
        ggplot()
        + geom_function("z = sin(a) sin(x) cos(y)", n=8)
        + slider(a=(0, math.pi), steps=3)
    )
    assert spec["is3d"] is True
    assert spec["layers"][0]["kind"] == "surface"
    assert spec["layers"][0]["frames"]["nFrames"] == 3
    assert not spec.get("notes")
    assert spec["scales"]["z"]["lo"] == pytest.approx(-0.83, abs=0.05)
    assert spec["scales"]["z"]["hi"] == pytest.approx(0.83, abs=0.05)


def test_slider_refuses_a_grid_that_is_too_large():
    from plot3 import geom_function

    with pytest.raises(ValueError, match="smaller steps"):
        build_spec(
            ggplot()
            + geom_function("z = a sin(x) cos(y)")
            + slider(a=(0, 1), k=(1, 2), steps=25)
        )
