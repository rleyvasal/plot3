"""Step 9: more colour, size, and alpha scales, coord_fixed, expand_limits,
ggplot2's theme elements, theme_grey / theme_linedraw, and labellers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes, coord_fixed, element_blank, element_line, element_rect, element_text,
    expand_limits, facet_grid, facet_wrap, geom_line, geom_point, geom_tile, ggplot,
    ggsave, label_both, labeller, scale_alpha, scale_colour_distiller,
    scale_colour_gradientn, scale_colour_identity, scale_fill_viridis_c, scale_size,
    scale_size_area, theme, theme_grey, theme_linedraw,
)
from plot3.build import build_doc, build_spec


def _d():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 2.0, 4.0],
                         "v": [-1.0, 0.0, 2.0, 5.0], "c": ["red", "#00ff00", "steelblue", "red"],
                         "g": ["a", "b", "a", "b"]})


def test_gradientn_distiller_viridis_fill():
    d = _d()
    spec, _ = build_spec(ggplot(d, aes("x", "y", colour="v")) + geom_point()
                         + scale_colour_gradientn(colours=["red", "white", "blue"]))
    assert spec["color"]["ramp"] == ["#ff0000", "#ffffff", "#0000ff"]
    spec, _ = build_spec(ggplot(d, aes("x", "y", colour="v")) + geom_point()
                         + scale_colour_gradientn(colours=["red", "blue"], values=[0, 0.25]))
    assert spec["color"]["ramp"][-1] == "#0000ff" and len(spec["color"]["ramp"]) == 65
    spec, _ = build_spec(ggplot(d, aes("x", "y", colour="v")) + geom_point()
                         + scale_colour_distiller(palette="Blues"))
    assert spec["color"]["ramp"][0] == "#08306b"  # darkest at the low end
    spec, _ = build_spec(ggplot(d, aes("x", "y", fill="v")) + geom_tile() + scale_fill_viridis_c())
    assert spec["color"]["ramp"][0] == "#440154"
    with pytest.raises(ValueError):
        scale_colour_gradientn(colours=["red"])


def test_identity_colours_have_no_legend():
    spec, _ = build_spec(ggplot(_d(), aes("x", "y", colour="c")) + geom_point() + scale_colour_identity())
    assert spec["legend"] is None
    assert set(spec["color"]["palette"]) == {"#ff0000", "#00ff00", "#4682b4"}


def test_size_scales_range_and_area():
    d = _d()
    ranged, _ = build_spec(ggplot(d, aes("x", "y", size="v")) + geom_point() + scale_size(range=(2, 10)))
    legend = ranged["sizeLegend"]
    assert legend["max"] == 10.0
    sizes = [b["t"] for b in legend["breaks"]]
    assert sizes == sorted(sizes) and min(sizes) >= 0.2 and max(sizes) <= 1.0  # 2 to 10 of 10
    assert ranged["layers"][0]["n"] == 4  # negatives kept on a ranged scale
    area, _ = build_spec(ggplot(d.assign(v=d.v.abs()), aes("x", "y", size="v")) + geom_point()
                         + scale_size_area(max_size=30))
    assert area["sizeLegend"]["max"] == 30.0


def test_alpha_aesthetic_and_scale(tmp_path):
    d = _d()
    spec, _ = build_spec(ggplot(d, aes("x", "y", alpha="v")) + geom_point() + scale_alpha(range=(0.2, 0.8)))
    assert spec["layers"][0]["opacity"]["dtype"] in {"u16", "f32"}
    alphas = [b["alpha"] for b in spec["alphaLegend"]["breaks"]]
    assert min(alphas) >= 0.2 - 1e-9 and max(alphas) <= 0.8 + 1e-9
    merged, _ = build_spec(ggplot(d, aes("x", "y", alpha="v", size="v")) + geom_point()
                           + scale_size(range=(2, 10)))
    assert merged["alphaLegend"] is None and "alpha" in merged["sizeLegend"]["breaks"][0]
    path = tmp_path / "alpha.svg"
    ggsave(str(path), ggplot(d, aes("x", "y", alpha="v")) + geom_point(), width=300, height=200)
    assert "fill-opacity" in path.read_text(encoding="utf-8")


def test_coord_fixed_and_expand_limits():
    d = _d()
    spec, _ = build_spec(ggplot(d, aes("x", "y")) + geom_point() + coord_fixed(ratio=2))
    assert spec["coord"] == {"aspect": "equal", "ratio": 2.0}
    spec, _ = build_spec(ggplot(d, aes("x", "y")) + geom_line() + expand_limits(y=0, x=[0, 10]))
    assert spec["scales"]["y"]["lo"] == 0.0 and spec["scales"]["x"]["hi"] == 10.0


def test_theme_elements_map_onto_the_figure():
    d = _d()
    spec, _ = build_spec(
        ggplot(d, aes("x", "y")) + geom_point()
        + theme(**{"axis.text.x": element_text(angle=45), "panel.grid": element_blank(),
                   "plot.title": element_text(hjust=0.5),
                   "panel.background": element_rect(fill="grey95"),
                   "axis.title.y": element_blank(), "axis.text.y": element_blank()})
    )
    opts = spec["themeOpts"]
    assert opts["xAngle"] == 45 and opts["panelGrid"] is False and opts["titleHjust"] == 0.5
    assert opts["yText"] is False and spec["labs"]["y"] == ""
    assert spec["theme"]["panel"] == "#f2f2f2"
    later, _ = build_spec(ggplot(d, aes("x", "y")) + geom_point()
                          + theme(panel_grid=element_line(colour="red"))
                          + theme(panel_border=element_rect(colour="blue")))
    assert later["theme"]["grid"] == "#ff0000" and later["theme"]["axis"] == "#0000ff"
    with pytest.warns(UserWarning, match="does not draw"):
        theme(strip_placement=element_blank())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        theme(panel_grid_minor=element_blank())  # accepted quietly
    with pytest.raises(TypeError):
        theme(axis_text_x=45)


def test_theme_grey_draws_a_grey_panel(tmp_path):
    d = _d()
    spec, _ = build_spec(ggplot(d, aes("x", "y")) + geom_point() + theme_grey())
    assert spec["theme"]["panel"] == "#ebebeb" and spec["theme"]["grid"] == "#ffffff"
    path = tmp_path / "grey.svg"
    ggsave(str(path), ggplot(d, aes("x", "y")) + geom_point() + theme_grey(), width=300, height=200)
    svg = path.read_text(encoding="utf-8")
    assert 'fill="#ebebeb"' in svg and 'stroke="#ffffff"' in svg
    lined, _ = build_spec(ggplot(d, aes("x", "y")) + geom_point() + theme_linedraw())
    assert lined["theme"]["axis"] == "#000000"


def test_labellers_name_the_strips():
    d = _d()
    doc = build_doc(ggplot(d, aes("x", "y")) + geom_point() + facet_wrap("g", labeller=label_both))
    assert "g: a" in doc and "g: b" in doc
    doc = build_doc(ggplot(d, aes("x", "y")) + geom_point()
                    + facet_wrap("g", labeller=labeller(g={"a": "Alpha", "b": "Beta"})))
    assert "Alpha" in doc and "Beta" in doc
    doc = build_doc(ggplot(d, aes("x", "y")) + geom_point()
                    + facet_grid("g ~ .", labeller=lambda value: value.upper()))
    assert ">A<" in doc or "A</" in doc
    with pytest.raises(ValueError):
        facet_wrap("g", labeller="label_nothing")
