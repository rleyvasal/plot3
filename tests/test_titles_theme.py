"""labs(subtitle, caption, tag) and theme() elements."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from plot3 import aes, geom_col, geom_point, ggplot, ggsave, labs, theme, theme_bw
from plot3.build import build_spec


@pytest.fixture
def df():
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [2.0, 4.0, 3.0],
        "g": ["a", "b", "a"],
        "country": ["United States", "Germany", "France"],
    })


def test_subtitle_caption_and_tag_reach_the_spec(df):
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y"))
        + geom_point()
        + labs(title="T", subtitle="S $\\mu$", caption="Source: X", tag="A")
    )
    assert spec["labs"]["subtitle"] == "S μ"
    assert spec["labs"]["caption"] == "Source: X"
    assert spec["labs"]["tag"] == "A"


def test_theme_options_accumulate(df):
    fig = (
        ggplot(df, aes(x="x", y="y"))
        + geom_point()
        + theme(panel_grid=False)
        + theme(axis_text_x_angle=45, plot_title_hjust=0.5)
    )
    spec, _ = build_spec(fig)
    assert spec["themeOpts"] == {"panelGrid": False, "xAngle": 45.0, "titleHjust": 0.5}


def test_theme_base_size_and_family_reach_saved_files(df):
    fig = ggplot(df, aes(x="x", y="y")) + geom_point() + theme(base_size=9, base_family="Arial")
    assert fig.theme_base_size == 9.0
    assert fig.theme_family == "Arial"


def test_legend_title_can_be_hidden(df):
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y", colour="g")) + geom_point() + theme(legend_title=False)
    )
    assert spec["labs"]["color"] == ""
    assert [entry["label"] for entry in spec["legend"]] == ["a", "b"]


@pytest.mark.parametrize(
    "kwargs",
    [{"axis_text_x_angle": 120}, {"plot_title_hjust": 2}, {"legend_position": "top"}],
)
def test_theme_rejects_out_of_range_values(kwargs):
    with pytest.raises(ValueError):
        theme(**kwargs)


def _svg(tmp_path, fig, name="fig.svg"):
    path = tmp_path / name
    ggsave(str(path), fig, width=4, height=3, units="in")
    return path.read_text()


def test_static_draws_header_and_caption_rows(tmp_path, df):
    svg = _svg(
        tmp_path,
        ggplot(df, aes(x="x", y="y")) + geom_point() + theme_bw()
        + labs(title="Main", subtitle="Under it", caption="Source: here", tag="B"),
    )
    ys = {
        text: float(m.group(1))
        for text in ("B", "Main", "Under it", "Source: here")
        for m in [re.search(r'<text[^>]* y="([-\d.]+)"[^>]*>' + re.escape(text) + "<", svg)]
    }
    assert ys["B"] == ys["Main"] < ys["Under it"] < ys["Source: here"]


def test_static_panel_grid_off_and_turned_labels(tmp_path, df):
    with_grid = _svg(tmp_path, ggplot(df, aes(x="country", y="y")) + geom_col() + theme_bw(), "a.svg")
    no_grid = _svg(
        tmp_path,
        ggplot(df, aes(x="country", y="y")) + geom_col() + theme_bw()
        + theme(panel_grid=False, axis_text_x_angle=45),
        "b.svg",
    )
    assert no_grid.count("<line") < with_grid.count("<line")
    assert 'rotate(-45' in no_grid


def test_static_centred_title(tmp_path, df):
    svg = _svg(
        tmp_path,
        ggplot(df, aes(x="x", y="y")) + geom_point() + labs(title="Centre me") + theme(plot_title_hjust=0.5),
    )
    assert re.search(r'text-anchor="middle"[^>]*>Centre me<', svg)
