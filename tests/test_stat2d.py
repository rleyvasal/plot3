"""Positioned bars, jitter, error bars, ribbons, smoothers, and summaries."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes,
    geom_bar,
    geom_col,
    geom_errorbar,
    geom_jitter,
    geom_line,
    geom_point,
    geom_pointrange,
    geom_ribbon,
    geom_smooth,
    ggplot,
    position_dodge,
    stat_summary,
)
from plot3 import special
from plot3.build import build_spec, expand_stat_geom
from plot3.static import _ticks
from plot3.scales import dt_ladder


@pytest.fixture
def counts():
    return pd.DataFrame({
        "day": ["Thu", "Thu", "Thu", "Fri", "Fri", "Sat"],
        "sex": ["F", "M", "M", "F", "F", "M"],
    })


def _rects(layer):
    frame = layer.data_override
    out = []
    for start, n in layer._groups:
        piece = frame.iloc[start:start + n]
        out.append((
            float(piece.x.min()), float(piece.x.max()),
            float(piece.y.min()), float(piece.y.max()),
            piece.iloc[0].get("sex"),
        ))
    return out


def test_stacked_bar_counts_split_by_group_first_level_on_top(counts):
    layer = expand_stat_geom(geom_bar(), aes(x="day", colour="sex"), counts)
    assert layer.kind == "poly"
    # ggplot2 level order: alphabetical, not first appearance.
    assert layer._violin_levels == ["Fri", "Sat", "Thu"]
    at = layer._violin_levels.index("Thu")
    rects = _rects(layer)
    thu = sorted([r for r in rects if abs((r[0] + r[1]) / 2 - at) < 1e-9], key=lambda r: r[2])
    # Thu: 1 F, 2 M. F is the first level, so it sits on top.
    assert [(r[4], r[2], r[3]) for r in thu] == [("M", 0.0, 2.0), ("F", 2.0, 3.0)]


def test_fill_position_sums_to_one(counts):
    layer = expand_stat_geom(
        geom_bar(position="fill"), aes(x="day", colour="sex"), counts
    )
    tops = {}
    for x0, x1, _y0, y1, _sex in _rects(layer):
        key = round((x0 + x1) / 2, 6)
        tops[key] = max(tops.get(key, 0.0), y1)
    assert all(top == pytest.approx(1.0) for top in tops.values())


def test_dodged_bars_sit_side_by_side(counts):
    layer = expand_stat_geom(
        geom_bar(position="dodge"), aes(x="day", colour="sex"), counts
    )
    at = layer._violin_levels.index("Thu")
    thu = sorted(
        [r for r in _rects(layer) if r[1] <= at + 0.5 and r[0] >= at - 0.5], key=lambda r: r[0]
    )
    assert [r[4] for r in thu] == ["F", "M"]
    assert thu[0][1] == pytest.approx(at)  # F's right edge meets M's left edge
    assert thu[1][0] == pytest.approx(at)
    assert thu[0][1] - thu[0][0] == pytest.approx(0.45)


def test_plain_columns_keep_the_ordinary_path():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    layer = expand_stat_geom(geom_col(), aes(x="x", y="y"), df)
    assert layer.kind == "col"


def test_error_bars_line_up_with_dodged_bars_by_name():
    df = pd.DataFrame({
        "day": ["Sat", "Sat", "Fri", "Fri"],
        "sex": ["F", "M", "F", "M"],
        "m": [3.0, 4.0, 1.0, 2.0],
    })
    df["lo"], df["hi"] = df.m - 0.5, df.m + 0.5
    spec, pairs = build_spec(
        ggplot(df, aes(x="day", y="m", colour="sex"))
        + geom_col(position="dodge")
        + geom_errorbar(aes(ymin="lo", ymax="hi"), position=position_dodge(0.9))
    )
    assert spec["scales"]["x"]["cats"] == ["Fri", "Sat"]
    bars = expand_stat_geom(geom_col(position="dodge"), aes(x="day", y="m", colour="sex"), df)
    bar_centres = sorted(round((r[0] + r[1]) / 2, 6) for r in _rects(bars))
    bars_err = expand_stat_geom(
        geom_errorbar(aes(ymin="lo", ymax="hi"), position="dodge"),
        aes(x="day", colour="sex"), df,
    )
    frame = bars_err.data_override
    stems = sorted(
        round(float(frame.x.iloc[start + 2]), 6) for start, _n in bars_err._groups
    )
    assert stems == bar_centres


def test_errorbar_path_has_caps():
    df = pd.DataFrame({"x": [1.0, 2.0], "lo": [0.0, 1.0], "hi": [2.0, 3.0]})
    layer = expand_stat_geom(geom_errorbar(width=0.5), aes(x="x", ymin="lo", ymax="hi"), df)
    assert layer.kind == "line"
    assert [n for _s, n in layer._groups] == [6, 6]
    first = layer.data_override.iloc[0:6]
    assert first.y.tolist() == [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
    assert first.x.max() - first.x.min() == pytest.approx(0.5)


def test_pointrange_is_a_line_and_a_point():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0], "lo": [0.5, 1.5], "hi": [1.5, 2.5]})
    layers = expand_stat_geom(geom_pointrange(), aes(x="x", y="y", ymin="lo", ymax="hi"), df)
    assert [layer.kind for layer in layers] == ["line", "point"]


def test_jitter_is_repeatable_and_keeps_y_with_height_zero():
    df = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50, "v": np.arange(100.0)})
    one = expand_stat_geom(geom_jitter(width=0.2, height=0), aes(x="g", y="v"), df)
    two = expand_stat_geom(geom_jitter(width=0.2, height=0), aes(x="g", y="v"), df)
    assert np.array_equal(one.data_override.x, two.data_override.x)
    assert np.array_equal(one.data_override.y, df.v)
    offsets = one.data_override.x.to_numpy() - np.repeat([0.0, 1.0], 50)
    assert np.all(np.abs(offsets) <= 0.2)
    assert one._violin_levels == ["a", "b"]


def test_ribbon_is_lower_then_upper_reversed():
    df = pd.DataFrame({"x": [2.0, 0.0, 1.0], "lo": [2.0, 0.0, 1.0], "hi": [3.0, 1.0, 2.0]})
    layer = expand_stat_geom(geom_ribbon(), aes(x="x", ymin="lo", ymax="hi"), df)
    assert layer.kind == "poly"
    assert layer.data_override.x.tolist() == [0.0, 1.0, 2.0, 2.0, 1.0, 0.0]
    assert layer.data_override.y.tolist() == [0.0, 1.0, 2.0, 3.0, 2.0, 1.0]


def test_lm_smoother_matches_least_squares_and_t_interval():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 40)
    y = 2.0 + 0.5 * x + rng.normal(0, 1, 40)
    df = pd.DataFrame({"x": x, "y": y})
    band, line = expand_stat_geom(geom_smooth(method="lm", n=11), aes(x="x", y="y"), df)
    slope, intercept = np.polyfit(x, y, 1)
    assert line.data_override.y.to_numpy() == pytest.approx(intercept + slope * np.linspace(0, 10, 11))
    # Half-width at the mean of x is t * sigma / sqrt(n).
    resid = y - (intercept + slope * x)
    sigma = math.sqrt(float(resid @ resid) / 38)
    half = float(special.qt(0.975, 38)) * sigma / math.sqrt(40)
    upper = band.data_override.y.to_numpy()[11:][::-1]
    lower = band.data_override.y.to_numpy()[:11]
    assert (upper[5] - lower[5]) / 2 == pytest.approx(half, rel=1e-6)


def test_loess_recovers_a_straight_line():
    x = np.linspace(0, 10, 60)
    df = pd.DataFrame({"x": x, "y": 3.0 - 2.0 * x})
    layers = expand_stat_geom(geom_smooth(n=7), aes(x="x", y="y"), df)
    line = layers[-1]
    assert line.data_override.y.to_numpy() == pytest.approx(3.0 - 2.0 * np.linspace(0, 10, 7), abs=1e-9)


def test_mean_cl_normal_matches_the_t_interval():
    df = pd.DataFrame({"g": ["a"] * 5, "v": [1.0, 2.0, 3.0, 4.0, 10.0]})
    layers = expand_stat_geom(
        stat_summary(fun_data="mean_cl_normal"), aes(x="g", y="v"), df
    )
    line, dot = layers
    v = df.v.to_numpy()
    half = float(special.qt(0.975, 4)) * v.std(ddof=1) / math.sqrt(5)
    assert dot.data_override.y.iloc[0] == pytest.approx(4.0)
    ys = sorted(line.data_override.y.tolist())
    assert ys[0] == pytest.approx(4.0 - half)
    assert ys[-1] == pytest.approx(4.0 + half)


@pytest.mark.parametrize(
    "fn, args, expected",
    [
        (special.qt, (0.975, 3), 3.182446305284263),
        (special.qt, (0.975, 10), 2.228138851986274),
        (special.pt, (2.0, 5), 0.9490302605850709),
        (special.pbeta, (0.3, 2, 5), 0.579825),
    ],
)
def test_t_and_beta_cdfs_match_r(fn, args, expected):
    assert float(fn(*args)) == pytest.approx(expected, rel=1e-9)


def test_layer_aes_names_the_axis_before_a_ribbon():
    ts = pd.DataFrame({"t": np.arange(5.0), "v": np.arange(5.0)})
    ts["lo"], ts["hi"] = ts.v - 1, ts.v + 1
    spec, _ = build_spec(
        ggplot(ts, aes(x="t"))
        + geom_ribbon(aes(ymin="lo", ymax="hi"))
        + geom_line(aes(y="v"))
    )
    assert spec["labs"]["y"] == "v"


def test_static_date_ticks_are_thinned_for_print():
    lo = pd.Timestamp("2024-01-01").timestamp()
    hi = lo + 59 * 86400
    ticks = _ticks({"kind": "dt", "lo": lo, "hi": hi, "ladder": dt_ladder(lo, hi)})
    assert 3 <= len(ticks) <= 8


def test_raw_points_and_summaries_share_a_categorical_axis():
    df = pd.DataFrame({"g": ["b", "a", "b", "a"], "v": [1.0, 2.0, 3.0, 4.0]})
    spec, _ = build_spec(
        ggplot(df, aes(x="g", y="v")) + geom_point() + stat_summary()
    )
    assert spec["scales"]["x"]["cats"] == ["a", "b"]
