"""Step 8: aes() expressions, geom_freqpoly, point positions, crossbars,
horizontal error bars, polygons, and dodged boxplots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes, geom_boxplot, geom_col, geom_crossbar, geom_errorbar, geom_errorbarh,
    geom_freqpoly, geom_point, geom_polygon, geom_text, ggplot, ggsave,
    position_jitter, position_jitterdodge, position_nudge, theme_bw,
)
from plot3.aesexpr import parse
from plot3.build import build_spec, expand_stat_geom
from plot3.table import ColumnNotFound


def _summary():
    return pd.DataFrame({"arm": ["a", "b", "c"], "mean": [5.6, 4.1, 4.0], "se": [0.2, 0.3, 0.25],
                         "cyl": [4, 8, 6], "First Name": [1.0, 2.0, 3.0]})


def test_expressions_compute_columns_named_by_the_expression():
    d = _summary()
    spec, _ = build_spec(ggplot(d, aes("arm", "mean")) + geom_col()
                         + geom_errorbar(aes(ymin="mean - se", ymax="mean + se")))
    assert spec["scales"]["y"]["hi"] == pytest.approx(5.8)
    spec, _ = build_spec(ggplot(d, aes("mean", "se", colour="factor(cyl)")) + geom_point())
    assert spec["color"]["cats"] == ["4", "6", "8"]
    assert spec["labs"]["color"] == "factor(cyl)"
    spec, _ = build_spec(ggplot(d, aes("mean", "log10(se * 100)")) + geom_point())
    assert spec["labs"]["y"] == "log10(se * 100)"
    spec, _ = build_spec(ggplot(d, aes("mean", "se", colour="mean > 4.5")) + geom_point())
    assert spec["color"]["cats"] == ["False", "True"]
    spec, _ = build_spec(ggplot(d, aes("mean", "`First Name`^2")) + geom_point())
    assert spec["scales"]["y"]["hi"] >= 9.0


def test_expressions_work_on_polars_and_layer_data():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 4.0, 5.0]})
    spec, _ = build_spec(ggplot(frame, aes("x", "y - mean(y)")) + geom_point())
    assert spec["scales"]["y"]["lo"] == pytest.approx(-1.0)
    other = pd.DataFrame({"x": [1.0, 2.0], "lo": [0.0, 1.0]})
    spec, _ = build_spec(ggplot(pd.DataFrame({"x": [1.0], "y": [1.0]}), aes("x", "y")) + geom_point()
                         + geom_point(aes(y="lo + 10"), data=other))
    assert spec["scales"]["y"]["hi"] >= 11.0


def test_expressions_run_no_code():
    for text in ['__import__("os")', "x.real", "open(x)", "(lambda: 1)()", "x[0]"]:
        assert parse(text) is None
    with pytest.raises(ColumnNotFound):
        build_spec(ggplot(_summary(), aes("mean", 'open("f")')) + geom_point())


def test_freqpoly_pads_zero_bins_per_group():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.normal(size=200), "g": rng.choice(["a", "b"], 200)})
    spec, _ = build_spec(ggplot(d, aes("x", colour="g")) + geom_freqpoly(bins=10))
    layer = spec["layers"][0]
    assert layer["kind"] == "line" and len(layer["groups"]) == 2
    assert [count for _start, count in layer["groups"]] == [12, 12]  # 10 bins + 2 zeros
    dens, _ = build_spec(ggplot(d, aes("x", y="after_stat(density)")) + geom_freqpoly(bins=10))
    assert dens["labs"]["y"] == "density"


def _points(geom, d, mapping):
    out = expand_stat_geom(geom, mapping, d)
    return out.data_override


def test_position_jitter_nudge_and_jitterdodge():
    d = pd.DataFrame({"g": ["a"] * 40 + ["b"] * 40, "s": ["F", "M"] * 40, "y": np.arange(80.0)})
    jit = _points(geom_point(position=position_jitter(width=0.2, height=0)), d, aes("g", "y"))
    assert np.allclose(jit["y"], d["y"])  # height=0 keeps y
    nudged = _points(geom_point(position=position_nudge(y=1.5)), d, aes("g", "y"))
    assert np.allclose(nudged["y"], d["y"] + 1.5)
    dodged = _points(geom_point(position=position_jitterdodge()), d, aes("g", "y", colour="s"))
    left = dodged.loc[dodged["s"] == "F", "x"].to_numpy()
    right = dodged.loc[dodged["s"] == "M", "x"].to_numpy()
    # Within category a (position 0): F to the left, M to the right.
    assert left[:20].max() < 0 < right[:20].min()
    again = _points(geom_point(position=position_jitterdodge()), d, aes("g", "y", colour="s"))
    assert np.allclose(again["x"], dodged["x"])  # same seed, same figure


def test_text_position_nudge_makes_room():
    d = _summary()
    spec, _ = build_spec(ggplot(d, aes("arm", "mean", label="round(mean, 1)")) + geom_col()
                         + geom_text(position=position_nudge(y=0.5)))
    assert spec["scales"]["y"]["hi"] >= 6.1
    assert {a["text"] for a in spec["ann"]} == {"5.6", "4.1", "4"}


def test_crossbar_box_and_fattened_middle():
    d = _summary()
    spec, _ = build_spec(ggplot(d, aes("arm", "mean", fill="arm"))
                         + geom_crossbar(aes(ymin="mean - se", ymax="mean + se")))
    kinds = [layer["kind"] for layer in spec["layers"]]
    assert kinds == ["poly", "line", "line"]
    assert spec["layers"][0]["polygon"] is True
    assert spec["layers"][2]["linewidth"] == pytest.approx(2.5 * spec["layers"][1]["linewidth"])


def test_errorbarh_on_a_category_axis():
    d = _summary()
    spec, _ = build_spec(ggplot(d, aes(y="arm", x="mean")) + geom_point()
                         + geom_errorbarh(aes(xmin="mean - se", xmax="mean + se")))
    bars = spec["layers"][1]
    assert bars["n"] == 18 and len(bars["groups"]) == 3
    assert spec["scales"]["y"]["kind"] == "cat"


def test_polygon_groups_fill_and_concave_shapes(tmp_path):
    t = np.linspace(0, 2 * np.pi, 11)[:-1]
    r = np.where(np.arange(10) % 2 == 0, 1.0, 0.45)
    star = pd.DataFrame({"x": np.r_[r * np.cos(t), 3 + r * np.cos(t)],
                         "y": np.r_[r * np.sin(t), r * np.sin(t)], "id": ["A"] * 10 + ["B"] * 10})
    spec, _ = build_spec(ggplot(star, aes("x", "y", fill="id")) + geom_polygon())
    layer = spec["layers"][0]
    assert layer["polygon"] is True and len(layer["groups"]) == 2
    assert spec["color"]["cats"] == ["A", "B"]
    path = tmp_path / "stars.svg"
    ggsave(str(path), ggplot(star, aes("x", "y", fill="id")) + geom_polygon(), width=400, height=300)
    assert path.read_text(encoding="utf-8").count("<polygon") >= 2


def test_boxplots_dodge_by_a_second_group():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"arm": rng.choice(["a", "b"], 80), "sex": rng.choice(["F", "M"], 80),
                      "y": rng.normal(size=80)})
    spec, _ = build_spec(ggplot(d, aes("arm", "y", colour="sex")) + geom_boxplot() + theme_bw())
    assert spec["scales"]["x"]["kind"] == "cat"
    assert spec["layers"][0]["n"] == 4  # a/F, a/M, b/F, b/M side by side
