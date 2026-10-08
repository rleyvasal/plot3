"""Regressions for the second publication review (bugs 1-8, print theme)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes,
    facet_wrap,
    geom_bar,
    geom_col,
    geom_density,
    geom_histogram,
    geom_line,
    geom_point,
    geom_violin,
    ggplot,
    ggsave,
    labs,
    stat_summary,
    theme_dark,
)
from plot3.build import build_spec, facet_cells
from plot3.scales import col_values, ordered_levels
from plot3.static import _fit_x_labels, _needs_wide_font


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "val": rng.normal(10, 2, 120),
        "sex": rng.choice(["M", "F"], 120),
        "grp": rng.choice(["Low", "High", "Control"], 120),
        "cyl": rng.choice([10, 4, 8, 6], 120),
    })


def test_1_histogram_with_fill_stacks_groups(df):
    spec, _ = build_spec(ggplot(df, aes(x="val", fill="sex")) + geom_histogram(bins=10))
    assert [layer["kind"] for layer in spec["layers"]] == ["poly"]
    assert [e["label"] for e in spec["legend"]] == ["F", "M"]
    assert spec["labs"]["y"] == "count"


def test_1_histogram_with_fill_in_facets(df):
    layout = facet_cells(ggplot(df, aes(x="val", fill="sex")) + geom_histogram(bins=8) + facet_wrap("grp"))
    for cell in layout["cells"]:
        build_spec(cell["fig"])  # no KeyError


def test_2_summary_bars_dodge_by_fill(df):
    spec, _ = build_spec(
        ggplot(df, aes(x="grp", y="val", fill="sex")) + stat_summary(geom="col")
    )
    layer = spec["layers"][0]
    assert layer["kind"] == "poly" and layer["color"]["kind"] == "cat"


def test_3_booleans_are_two_groups():
    kind, codes, cats = col_values(pd.Series([True, False, True]))
    assert kind == "cat" and cats == ["False", "True"] and codes.tolist() == [1.0, 0.0, 1.0]


def test_4_one_level_order_everywhere(df):
    assert ordered_levels([10, 4, 8, 6]) == [4, 6, 8, 10]
    assert ordered_levels(["b", None, "a"]) == ["a", "b", None]
    assert ordered_levels(["lo", "hi"], keep_order=True) == ["lo", "hi"]
    bar, _ = build_spec(ggplot(df, aes(x="grp")) + geom_bar())
    violin, _ = build_spec(ggplot(df, aes(x="grp", y="val")) + geom_violin())
    numeric, _ = build_spec(ggplot(df, aes(x="cyl")) + geom_bar())
    assert bar["scales"]["x"]["cats"] == violin["scales"]["x"]["cats"] == ["Control", "High", "Low"]
    assert numeric["scales"]["x"]["cats"] == ["4", "6", "8", "10"]
    layout = facet_cells(ggplot(df, aes(x="val", y="val")) + geom_point() + facet_wrap("grp"))
    assert [c["strip"] for c in layout["cells"]] == ["Control", "High", "Low"]


@pytest.mark.parametrize(
    "geom, x, y",
    [(geom_density(), "val", "density"), (geom_histogram(bins=5), "val", "count"), (geom_bar(), "grp", "count")],
)
def test_5_computed_axes_are_named(df, geom, x, y):
    spec, _ = build_spec(ggplot(df, aes(x=x)) + geom)
    assert spec["labs"]["y"] == y


def test_5_labs_still_wins(df):
    spec, _ = build_spec(ggplot(df, aes(x="val")) + geom_density() + labs(y="Density"))
    assert spec["labs"]["y"] == "Density"


@pytest.mark.parametrize(
    "labels, angle, thin",
    [
        (list("abcd"), None, None),
        ([f"category {i:02d}" for i in range(12)], 45.0, None),
        ([f"category {i:02d}" for i in range(60)], 90.0, 4),
    ],
)
def test_6_crowded_category_labels_turn_or_thin(labels, angle, thin):
    from plot3 import static

    frame = pd.DataFrame({"c": labels, "v": range(len(labels))})
    spec, _ = build_spec(ggplot(frame, aes(x="c", y="v")) + geom_col())
    token = static._REAL_FONT.set(True)  # measured as ggsave measures
    try:
        opts = _fit_x_labels(spec, 336, 11).get("themeOpts") or {}
    finally:
        static._REAL_FONT.reset(token)
    assert opts.get("xAngle") == angle
    assert opts.get("xThin") == thin


def test_7_dropped_rows_are_reported(capsys, tmp_path):
    frame = pd.DataFrame({"x": [1, 2, np.nan, 4], "y": [1, np.nan, 3, 4]})
    spec, _ = build_spec(ggplot(frame, aes(x="x", y="y")) + geom_point() + geom_line())
    assert spec["notes"][:2] == [
        "Removed 2 rows containing missing values (geom_point)",
        "Removed 2 rows containing missing values (geom_line)",
    ]
    ggsave(str(tmp_path / "na.svg"), ggplot(frame, aes(x="x", y="y")) + geom_point())
    assert "Removed 2 rows" in capsys.readouterr().out


def test_8_non_latin_text_gets_a_wide_font(tmp_path):
    assert _needs_wide_font("東京") and _needs_wide_font("✓") and _needs_wide_font("서울")
    assert not _needs_wide_font("Zürich — μ ± 2°")
    frame = pd.DataFrame({"x": ["東京", "Zürich"], "y": [1, 2]})
    path = tmp_path / "uni.svg"
    ggsave(str(path), ggplot(frame, aes(x="x", y="y")) + geom_col())
    svg = path.read_text()
    tokyo = re.search(r'<text[^>]*font-family="([^"]*)"[^>]*>東京<', svg)
    assert tokyo and "Hiragino" in tokyo.group(1)


def test_saved_files_default_to_theme_bw(tmp_path):
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    plain = tmp_path / "plain.svg"
    dark = tmp_path / "dark.svg"
    ggsave(str(plain), ggplot(frame, aes(x="x", y="y")) + geom_point())
    ggsave(str(dark), ggplot(frame, aes(x="x", y="y")) + geom_point() + theme_dark())
    assert 'fill="#ffffff"' in plain.read_text()
    assert 'fill="#ffffff"' not in dark.read_text().split("<circle")[0]
    # The interactive viewer keeps its own default.
    spec, _ = build_spec(ggplot(frame, aes(x="x", y="y")) + geom_point())
    assert spec["theme"]["surface"] != "#ffffff"
