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
