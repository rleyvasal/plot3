"""stat_* aliases, geom_blank, guide objects, vars(), position_dodge2, and
scale_colour_hue."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import plot3
from plot3 import (
    aes, facet_grid, facet_wrap, geom_bar, geom_blank, geom_boxplot, geom_col,
    geom_point, ggplot, ggsave, guide_colourbar, guide_legend, guide_none, guides,
    position_dodge2, scale_colour_hue, stat_bin, stat_count, stat_density,
    stat_function, stat_smooth, vars,
)
from plot3.build import build_doc, build_spec, expand_stat_geom
from plot3.special import dnorm


def _d():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 2.0, 4.0],
                         "g": ["a", "b", "a", "b"], "h": ["p", "p", "q", "q"]})


def test_stat_aliases_are_the_geoms():
    assert stat_smooth is plot3.geom_smooth and stat_bin is plot3.geom_histogram
    assert stat_count is plot3.geom_bar and stat_density is plot3.geom_density


def test_stat_function_with_fun_and_args():
    spec, _ = build_spec(ggplot() + stat_function(fun=dnorm, args={"mean": 2, "sd": 0.5}, xlim=(0, 4)))
    assert spec["scales"]["y"]["hi"] == pytest.approx(0.798, abs=0.01)
    spec, _ = build_spec(ggplot() + stat_function(fun="sin(x)", xlim=(0, 3)))
    assert spec["layers"][0]["kind"] == "line"
    with pytest.raises(ValueError, match="fun="):
        stat_function()


def test_geom_blank_reaches_the_scales_but_draws_nothing(tmp_path):
    d = _d()
    spec, _ = build_spec(ggplot(d, aes("x", "y")) + geom_point() + geom_blank(aes(y="y * 3")))
    assert spec["scales"]["y"]["hi"] == 12.0
    assert spec["layers"][1]["blank"] is True
    path = tmp_path / "blank.svg"
    ggsave(str(path), ggplot(d, aes("x", "y")) + geom_blank(), width=300, height=200)
    assert "<circle" not in path.read_text(encoding="utf-8")


def test_guide_objects_title_reverse_and_hide():
    d = _d()
    spec, _ = build_spec(ggplot(d, aes("x", "y", colour="g")) + geom_point()
                         + guides(colour=guide_legend(title="Group", reverse=True)))
    assert spec["labs"]["color"] == "Group"
    assert [e["label"] for e in spec["legend"]] == ["b", "a"]
    # Each row still names its own category, so clicking "b" hides b.
    assert [e["ci"] for e in spec["legend"]] == [1, 0]
    spec, _ = build_spec(ggplot(d, aes("x", "y", colour="y")) + geom_point()
                         + guides(colour=guide_colourbar(title="Height")))
    assert spec["labs"]["color"] == "Height"
    spec, _ = build_spec(ggplot(d, aes("x", "y", colour="g")) + geom_point() + guides(colour=guide_none()))
    assert spec["legend"] is None


def test_vars_names_facets_and_keeps_pythons_vars():
    d = _d()
    assert "g" in build_doc(ggplot(d, aes("x", "y")) + geom_point() + facet_wrap(vars("g")))
    grid = facet_grid(rows=vars("g"), cols=vars("h"))
    assert (grid.rows, grid.cols) == ("g", "h")
    assert facet_grid(vars("g")).rows == "g"
    with pytest.raises(ValueError):
        facet_wrap(vars("g", "h"))

    class Thing:
        def __init__(self):
            self.a = 1

    assert vars(Thing()) == {"a": 1}  # Python's vars still works
    local_name = 5
    assert vars()["local_name"] == 5


def _bar_widths(out):
    xs = out.data_override["x"].to_numpy(np.float64)
    return [float(np.ptp(xs[s:s + c])) for s, c in out._groups]


def test_position_dodge2_leaves_gaps_between_bars():
    d = _d()
    plain = expand_stat_geom(geom_col(position="dodge"), aes("g", "y", colour="h"), d)
    padded = expand_stat_geom(geom_col(position=position_dodge2(padding=0.2)), aes("g", "y", colour="h"), d)
    assert _bar_widths(padded)[0] == pytest.approx(0.8 * _bar_widths(plain)[0])
    with pytest.raises(ValueError):
        position_dodge2(padding=1.5)


def test_boxplots_dodge_with_dodge2_padding():
    d = pd.DataFrame({"g": ["a", "b"] * 20, "s": ["F"] * 20 + ["M"] * 20, "y": np.arange(40.0)})
    default = expand_stat_geom(geom_boxplot(), aes("g", "y", colour="s"), d)
    wider = expand_stat_geom(geom_boxplot(position=position_dodge2(padding=0.0)), aes("g", "y", colour="s"), d)
    assert default.width == pytest.approx(0.75 / 2 * 0.9)
    assert wider.width == pytest.approx(0.75 / 2)


def test_scale_colour_hue_matches_ggplot2():
    assert scale_colour_hue().colours(["a", "b", "c"], []) == ["#f8766d", "#00ba38", "#619cff"]
    five = scale_colour_hue().colours(list("abcde"), [])
    assert five == ["#f8766d", "#a3a500", "#00bf7d", "#00b0f6", "#e76bf3"]
    spec, _ = build_spec(ggplot(_d(), aes("x", "y", colour="g")) + geom_point() + scale_colour_hue())
    assert spec["color"]["palette"] == ["#f8766d", "#00bfc4"]
