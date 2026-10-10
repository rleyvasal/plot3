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


def test_fill_is_its_own_aesthetic():
    # ggplot2: fill colours filled shapes; build_spec applies it per geom.
    a = aes(x="x", y="y", z="height", fill="height")
    assert a["fill"] == "height"
    assert "color" not in a


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


def test_set_colour_replaces_inherited_colour_mapping():
    # ggplot2: geom_point(colour = "black") under aes(colour = g) draws black,
    # and the layer's own data need not have g.
    import pandas as pd

    from plot3 import aes, geom_boxplot, geom_point, ggplot
    from plot3.build import build_spec

    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "g": ["a", "b", "c"]})
    centre = pd.DataFrame({"x": [2], "y": [2]})
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y", colour="g")) + geom_point()
        + geom_point(data=centre, colour="black", shape="cross")
    )
    assert spec["layers"][0].get("color") is not None
    assert spec["layers"][1].get("color") is None
    assert spec["layers"][1]["constColor"] == "#000000"
    # A filled layer keeps its mapped fill under a set outline colour.
    spec, _ = build_spec(
        ggplot(df, aes(x="g", y="y", fill="g")) + geom_boxplot(colour="black")
    )
    assert spec["layers"][0].get("color") is not None


def test_geom_smooth_takes_a_linetype():
    import pandas as pd
    import pytest

    from plot3 import aes, geom_smooth, ggplot
    from plot3.build import build_spec

    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.5, 2.9, 4.2]})
    spec, _ = build_spec(ggplot(df, aes(x="x", y="y")) + geom_smooth(method="lm", se=False, linetype="dashed"))
    assert [layer.get("dash") for layer in spec["layers"] if layer["kind"] == "line"] == [[4.0, 4.0]]
    with pytest.raises(ValueError):
        geom_smooth(linetype="wavy")


@pytest.mark.parametrize("kind", ["violin", "boxplot", "density", "histogram", "bar", "smooth", "summary"])
def test_categorical_colour_order_survives_stats(kind):
    # pd.Categorical(categories=["placebo", "low", "high"]) sets the colour
    # order for every geom, as in ggplot2; stats used to sort it as text, so
    # a violin and a scatter of the same column disagreed on colours.
    import numpy as np
    import pandas as pd

    from plot3 import (aes, geom_bar, geom_boxplot, geom_density, geom_histogram,
                       geom_smooth, geom_violin, ggplot, stat_summary)
    from plot3.build import build_spec

    rng = np.random.default_rng(0)
    arms = ["placebo", "low", "high"]
    df = pd.DataFrame({"arm": pd.Categorical(rng.choice(arms, 90), categories=arms),
                       "x": rng.uniform(0, 10, 90), "y": rng.normal(size=90),
                       "sex": rng.choice(["F", "M"], 90)})
    layer = {
        "violin": (aes(x="arm", y="y", fill="arm"), geom_violin()),
        "boxplot": (aes(x="sex", y="y", fill="arm"), geom_boxplot()),
        "density": (aes(x="y", colour="arm"), geom_density()),
        "histogram": (aes(x="y", fill="arm"), geom_histogram(bins=10)),
        "bar": (aes(x="sex", fill="arm"), geom_bar()),
        "smooth": (aes(x="x", y="y", colour="arm"), geom_smooth(method="lm")),
        "summary": (aes(x="sex", y="y", colour="arm"), stat_summary()),
    }[kind]
    spec, _ = build_spec(ggplot(df, layer[0]) + layer[1])
    assert spec["color"]["cats"] == arms
    assert [entry["label"] for entry in spec["legend"]] == arms


def test_theme_grey_legend_has_no_dark_frame(tmp_path):
    import pandas as pd

    from plot3 import aes, geom_point, ggplot, ggsave, theme_grey

    df = pd.DataFrame({"x": [1, 2], "y": [1, 2], "g": ["a", "b"]})
    out = tmp_path / "g.svg"
    ggsave(str(out), ggplot(df, aes(x="x", y="y", colour="g")) + geom_point() + theme_grey())
    assert 'stroke="#4d4d4d"' not in out.read_text()
