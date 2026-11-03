"""Scale transformations run before statistics, coord_transform after (ggplot2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from plot3 import (
    aes,
    coord_flip,
    coord_transform,
    geom_boxplot,
    geom_histogram,
    geom_point,
    ggplot,
    pseudo_log_trans,
    scale_x_log10,
    scale_y_continuous,
    scale_y_log10,
)
from plot3.build import build_spec
from plot3.transforms import PseudoLog, pseudo_log_breaks


def _costs():
    rng = np.random.default_rng(1)
    cost = np.round(rng.lognormal(3, 1.2, 200), 2)
    cost[0] = 0.0
    return pd.DataFrame({"g": ["a"] * 200, "cost": cost})


def _n_outliers(values):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())


def _box(spec):
    return next(layer for layer in spec["layers"] if layer["kind"] == "box")


def test_log_scale_boxplot_statistics_use_log_values():
    df = _costs()
    positive = df.cost[df.cost > 0].to_numpy()
    spec, _ = build_spec(ggplot(df, aes(x="g", y="cost")) + geom_boxplot() + scale_y_log10())
    assert _box(spec)["nOut"] == _n_outliers(np.log10(positive))
    assert any("non-positive" in note for note in spec["notes"])
    assert not any("missing values" in note for note in spec["notes"])


def test_coord_transform_boxplot_statistics_use_original_values():
    df = _costs()
    positive = df.cost[df.cost > 0].to_numpy()
    spec, _ = build_spec(ggplot(df, aes(x="g", y="cost")) + geom_boxplot() + coord_transform(y="log10"))
    assert _box(spec)["nOut"] == _n_outliers(positive)
    assert spec["scales"]["y"]["trans"] == "log10"


def test_scale_and_coord_differ_and_flip_keeps_the_scale_order():
    df = _costs()
    base = ggplot(df, aes(x="g", y="cost")) + geom_boxplot()
    scaled = _box(build_spec(base + scale_y_log10())[0])["nOut"]
    coord = _box(build_spec(base + coord_transform(y="log10"))[0])["nOut"]
    upright, _ = build_spec(base + scale_y_log10())
    flipped, _ = build_spec(base + scale_y_log10() + coord_flip())
    assert scaled != coord
    # A flipped box is drawn as polygons, whisker lines, and outlier points.
    points = next(layer for layer in flipped["layers"] if layer["kind"] == "point")
    assert points["n"] == scaled
    assert flipped["scales"]["x"]["trans"] == "log10"
    assert flipped["scales"]["x"]["lo"] == pytest.approx(upright["scales"]["y"]["lo"])
    assert flipped["scales"]["x"]["hi"] == pytest.approx(upright["scales"]["y"]["hi"])


def test_polars_data_is_transformed_before_statistics():
    df = _costs()
    spec, _ = build_spec(ggplot(pl.DataFrame({"g": df.g.tolist(), "cost": df.cost.to_numpy()}), aes(x="g", y="cost")) + geom_boxplot() + scale_y_log10())
    positive = df.cost[df.cost > 0].to_numpy()
    assert _box(spec)["nOut"] == _n_outliers(np.log10(positive))


def test_log_x_histogram_bins_in_log_space():
    x = 10.0 ** np.linspace(0, 4, 400)
    df = pd.DataFrame({"x": x})
    spec, _ = build_spec(ggplot(df, aes(x="x")) + geom_histogram(bins=4) + scale_x_log10())
    sc = spec["scales"]["x"]
    # Bins over log10 values reach from about 0 to 4, not from 1 to 10000.
    assert sc["lo"] == pytest.approx(0.0, abs=0.7)
    assert sc["hi"] == pytest.approx(4.0, abs=0.7)


def test_same_column_on_log_axis_and_colour_keeps_raw_colour():
    df = pd.DataFrame({"x": [1.0, 10.0, 100.0], "y": [1.0, 2.0, 3.0]})
    spec, _ = build_spec(ggplot(df, aes(x="x", y="y", colour="x")) + geom_point() + scale_x_log10())
    assert spec["scales"]["x"]["lo"] <= 0.0 + 1e-9
    # Colour reads the original values (robust limits: near 100, not 2).
    assert spec["color"]["hi"] > 50


def test_scale_and_coord_on_one_axis_is_an_error():
    df = _costs()
    with pytest.raises(ValueError, match="coord_transform"):
        build_spec(ggplot(df, aes(x="g", y="cost")) + geom_boxplot() + scale_y_log10() + coord_transform(y="log10"))


def test_pseudo_log_keeps_zeros_and_negatives():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [-500.0, 0.0, 3.0, 20000.0]})
    spec, _ = build_spec(ggplot(df, aes(x="x", y="y")) + geom_point() + scale_y_continuous(trans="pseudo_log"))
    sc = spec["scales"]["y"]
    assert sc["trans"] == "pseudo_log" and sc["sigma"] == 1.0
    assert not spec["notes"]
    labels = [label for _pos, label in sc["ticks"]]
    assert "0" in labels and "-100" in labels and "10000" in labels
    t = PseudoLog()
    assert sc["lo"] <= t.value(-500.0) and sc["hi"] >= t.value(20000.0)


def test_pseudo_log_transform_round_trips_and_is_linear_near_zero():
    t = pseudo_log_trans(sigma=2, base=10)
    v = np.array([-1e5, -3.0, 0.0, 0.5, 7.0, 1e6])
    assert np.allclose(t.inverse(t.transform(v)), v)
    assert t.value(0.0) == 0.0
    assert t.value(0.01) == pytest.approx(0.01 / 4 / math.log(10), rel=1e-4)
    assert scale_y_continuous(transform="pseudo_log").transform == PseudoLog()


def test_pseudo_log_breaks():
    assert pseudo_log_breaks(0, 5) == [0, 1, 2, 3, 4, 5]
    assert pseudo_log_breaks(-1000, 1000) == [-1000, -100, -10, -1, 0, 1, 10, 100, 1000]
    assert len(pseudo_log_breaks(-1e9, 1e9)) <= 11


def test_bad_trans_is_rejected():
    with pytest.raises(ValueError, match="pseudo_log"):
        scale_y_continuous(trans="log2")
