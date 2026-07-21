"""ggplot grammar: layering, aes, themes, errors."""

from __future__ import annotations

import pytest

from plot3 import (
    aes,
    coord_3d,
    facet_wrap,
    geom_col,
    geom_point,
    geom_point3d,
    ggplot,
    labs,
    scale_colour_viridis_c,
    theme_light,
)
from plot3.build import build_spec
from tests.helpers import assert_html_figure, assert_layer_kind, assert_spec_3d


def test_pipe_and_add_layers(cars):
    fig = (
        cars
        >> ggplot(aes(x="wt", y="mpg", colour="cyl"))
        + geom_point(size=4)
        + theme_light()
        + labs(title="cars", x="weight", y="mpg")
    )
    assert fig.data is not None
    assert len(fig.layers) == 1
    assert fig.theme_name == "light"
    assert fig.labs["title"] == "cars"
    assert_html_figure(fig._repr_html_())


def test_ggplot_data_first(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    spec, _ = build_spec(fig)
    assert_layer_kind(spec, "point")
    assert spec["is3d"] is False


def test_deferred_ggplot_then_pipe(cars):
    template = ggplot(aes(x="wt", y="mpg")) + geom_point()
    fig = cars >> template
    assert len(fig.layers) == 1
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["n"] == len(cars)


def test_cannot_pipe_into_bound_figure(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    with pytest.raises(TypeError, match="already has data"):
        _ = cars >> fig


def test_colour_and_color_aliases(cars):
    a = aes(x="wt", y="mpg", colour="cyl")
    b = aes(x="wt", y="mpg", color="cyl")
    assert a["color"] == b["color"] == "cyl"


def test_fill_aliases_colour_for_surface_aes():
    a = aes(x="x", y="y", z="height", fill="height")
    assert a["color"] == "height"


def test_coord_and_facet_attach(cars, cloud):
    fig2 = ggplot(cars, aes(x="wt", y="mpg")) + geom_point() + facet_wrap("cyl")
    assert fig2.facet is not None
    fig3 = (
        ggplot(cloud, aes(x="x", y="y", z="z"))
        + geom_point3d()
        + coord_3d(aspect="equal")
    )
    assert fig3.coord is not None
    assert fig3.coord.aspect == "equal"
    assert_spec_3d(build_spec(fig3)[0])


def test_scale_colour_viridis_on_points(cloud):
    fig = (
        ggplot(cloud, aes(x="x", y="y", z="z", colour="intensity"))
        + geom_point3d(size=0.01)
        + scale_colour_viridis_c(option="turbo")
        + coord_3d()
    )
    spec, _ = build_spec(fig)
    assert spec["color"]["kind"] == "num"
    assert "ramp" in spec["color"]


def test_unknown_add_raises(cars):
    with pytest.raises(TypeError, match="cannot add"):
        _ = ggplot(cars, aes(x="wt", y="mpg")) + 123
