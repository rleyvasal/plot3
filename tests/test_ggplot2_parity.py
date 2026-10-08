"""Step 7: ggplot2's defaults and messages (ticks, black marks, warnings,
labs(None), facet grids, missing columns, non-finite values)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes, facet_wrap, geom_boxplot, geom_col, geom_histogram, geom_point, geom_violin,
    ggplot, labs, theme_bw, theme_dark,
)
from plot3.build import _panel_grid, build_doc, build_spec
from plot3.scales import fmt_ticks, nice_ticks
from plot3.table import ColumnNotFound


@pytest.mark.parametrize("lo,hi,expected", [
    (-0.3, 9.1, [0, 2, 4, 6, 8]),
    (0, 16, [0, 5, 10, 15]),
    (-2.4, 2.6, [-2, -1, 0, 1, 2]),
    (0, 1, [0, 0.25, 0.5, 0.75, 1]),
    (1990, 2024, [1990, 2000, 2010, 2020]),
])
def test_about_five_round_ticks_like_ggplot2(lo, hi, expected):
    assert nice_ticks(lo, hi) == expected


def test_tick_labels_stay_apart_for_large_numbers():
    ticks = nice_ticks(1e12, 1e12 + 5)
    labels = fmt_ticks(ticks)
    assert len(set(labels)) == len(labels) == len(ticks)
    assert labels[1] == "1000000000001"


def test_unmapped_layers_are_black_and_bars_grey35_on_light_themes():
    d = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0]})
    spec, _ = build_spec(ggplot(d, aes("x", "y")) + geom_col() + geom_point() + theme_bw())
    assert [layer["constColor"] for layer in spec["layers"]] == ["#595959", "#000000"]
    dark, _ = build_spec(ggplot(d, aes("x", "y")) + geom_point() + theme_dark())
    assert dark["layers"][0]["constColor"] != "#000000"  # black would not show


def test_unmapped_boxes_and_violins_are_white_inside():
    d = pd.DataFrame({"g": list("ab") * 10, "y": np.arange(20.0)})
    for geom in (geom_boxplot(), geom_violin()):
        spec, _ = build_spec(ggplot(d, aes("g", "y")) + geom + theme_bw())
        assert spec["layers"][0]["plainFill"] is True
    mapped, _ = build_spec(ggplot(d, aes("g", "y", fill="g")) + geom_boxplot() + theme_bw())
    assert "plainFill" not in mapped["layers"][0]


def test_unknown_parameters_warn_with_a_guess():
    with pytest.warns(UserWarning, match=r"geom_point\(\): colr \(did you mean colo"):
        geom_point(colr="red")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        geom_point(colour="red", size=2, alpha=0.5)  # all known


def test_boxplot_outliers_can_be_hidden():
    d = pd.DataFrame({"g": ["a"] * 21, "y": list(range(20)) + [100.0]})
    shown, _ = build_spec(ggplot(d, aes("g", "y")) + geom_boxplot())
    hidden, _ = build_spec(ggplot(d, aes("g", "y")) + geom_boxplot(outliers=False))
    na_shape, _ = build_spec(ggplot(d, aes("g", "y")) + geom_boxplot(outlier_shape=None))
    assert shown["layers"][0]["nOut"] == 1
    assert hidden["layers"][0]["nOut"] == 0 == na_shape["layers"][0]["nOut"]


def test_histogram_on_the_density_scale():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.normal(size=500), "g": rng.choice(["a", "b"], 500)})
    spec, _ = build_spec(ggplot(d, aes("x", y="after_stat(density)")) + geom_histogram(bins=20))
    assert spec["labs"]["y"] == "density"
    grouped, _ = build_spec(ggplot(d, aes("x", y="..density..", fill="g")) + geom_histogram(bins=20))
    assert grouped["labs"]["y"] == "density"
    with pytest.raises(ValueError, match="after_stat"):
        build_spec(ggplot(d, aes("x", y="after_stat(ncount)")) + geom_histogram())


def test_density_bars_integrate_to_one():
    from plot3.build import _density_scale

    counts = np.array([2.0, 6.0, 2.0])
    edges = np.array([0.0, 0.5, 1.0, 1.5])
    assert np.isclose(np.sum(_density_scale(counts, edges) * np.diff(edges)), 1.0)


def test_labs_none_removes_a_title():
    d = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    spec, _ = build_spec(ggplot(d, aes("x", "y")) + geom_point() + labs(x=None, y="Y"))
    assert spec["labs"]["x"] == "" and spec["labs"]["y"] == "Y"
    # A label left out is not a label removed.
    kept, _ = build_spec(ggplot(d, aes("x", "y")) + geom_point() + labs(title="T"))
    assert kept["labs"]["x"] == "x"


def test_facet_wrap_grid_follows_ggplot2():
    assert [_panel_grid(n, None, None) for n in (2, 3, 4, 5, 7, 10)] == [
        (2, 1), (3, 1), (2, 2), (3, 2), (3, 3), (4, 3)]


def test_facet_legend_shows_shapes_and_colour_bars():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30),
                      "g": rng.choice(list("ab"), 30), "f": rng.choice(["F1", "F2"], 30)})
    doc = build_doc(ggplot(d, aes("x", "y", colour="g", shape="g")) + geom_point() + facet_wrap("f"))
    assert "▲" in doc
    doc = build_doc(ggplot(d, aes("x", "y", colour="y")) + geom_point() + facet_wrap("f"))
    assert "linear-gradient" in doc


def test_missing_column_names_the_closest_match():
    d = pd.DataFrame({"x": [1.0], "weight": [2.0]})
    with pytest.raises(ColumnNotFound, match="Did you mean 'weight'"):
        build_spec(ggplot(d, aes("x", "wieght")) + geom_point())
    with pytest.raises(KeyError, match="add it as a column first"):
        build_spec(ggplot(d, aes("x", "mean - se")) + geom_point())


def test_infinite_values_are_dropped_and_reported():
    d = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, np.inf, 3.0]})
    spec, _ = build_spec(ggplot(d, aes("x", "y")) + geom_point())
    assert spec["layers"][0]["n"] == 2
    assert spec["notes"] == ["Removed 1 row containing non-finite values (geom_point)"]


def test_building_html_prints_nothing(capsys, monkeypatch):
    monkeypatch.delenv("PLOT3_VERBOSE", raising=False)
    d = pd.DataFrame({"x": range(10), "y": range(10), "f": ["a", "b"] * 5})
    (ggplot(d, aes("x", "y")) + geom_point() + facet_wrap("f")).html()
    assert capsys.readouterr().out == ""
