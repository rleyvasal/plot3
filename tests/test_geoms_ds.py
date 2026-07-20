"""Smoke tests for data-science geoms and facets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plot3 import (
    aes,
    facet_wrap,
    geom_bar,
    geom_boxplot,
    geom_col,
    geom_density,
    geom_histogram,
    geom_point,
    geom_violin,
    ggplot,
    labs,
)
from plot3.build import build_spec


def test_geom_col_builds_html():
    df = pd.DataFrame({"g": ["a", "b", "c"], "n": [3, 7, 2]})
    fig = ggplot(df, aes(x="g", y="n")) + geom_col() + labs(title="counts")
    html = fig._repr_html_()
    assert len(html) > 500


def test_geom_bar_and_histogram_expand():
    df = pd.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "b", "c"],
            "x": [0.1, 0.2, 0.8, 1.1, 1.0, 2.5],
        }
    )
    bar = ggplot(df, aes(x="cat")) + geom_bar()
    hist = ggplot(df, aes(x="x")) + geom_histogram(bins=5)
    assert "layers" in str(type(bar)) or bar.layers
    assert len(bar._repr_html_()) > 500
    assert len(hist._repr_html_()) > 500


def test_point_still_works():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 4.0]})
    fig = ggplot(df, aes(x="x", y="y")) + geom_point()
    assert len(fig._repr_html_()) > 500


def test_geom_boxplot_builds_and_marks_outliers():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "g": ["a"] * 40 + ["b"] * 40,
            "y": list(rng.normal(0, 1, 40))
            + list(rng.normal(2, 1, 39))
            + [20.0],  # clear outlier in group b
        }
    )
    fig = ggplot(df, aes(x="g", y="y")) + geom_boxplot() + labs(title="boxes")
    html = fig._repr_html_()
    assert len(html) > 500
    spec, _payloads = build_spec(fig)
    layer = spec["layers"][0]
    assert layer["kind"] == "box"
    assert layer["n"] == 2
    assert "ymin" in layer and "middle" in layer and "ymax" in layer
    assert layer["nOut"] >= 1


def test_geom_density_area_and_grouped():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "x": np.concatenate(
                [rng.normal(0, 1, 80), rng.normal(3, 0.8, 80)]
            ),
            "g": ["a"] * 80 + ["b"] * 80,
        }
    )
    fig = (
        ggplot(df, aes(x="x", colour="g"))
        + geom_density(n=64, fill=True)
        + labs(title="density")
    )
    assert len(fig._repr_html_()) > 500
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["kind"] == "area"
    assert len(spec["layers"][0]["groups"]) == 2


def test_geom_violin_builds_polygons():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "g": ["a"] * 60 + ["b"] * 60,
            "y": np.concatenate(
                [rng.normal(0, 1, 60), rng.normal(1.5, 0.7, 60)]
            ),
        }
    )
    fig = ggplot(df, aes(x="g", y="y")) + geom_violin(n=48)
    assert len(fig._repr_html_()) > 500
    spec, _ = build_spec(fig)
    layer = spec["layers"][0]
    assert layer["kind"] == "poly"
    assert len(layer["groups"]) == 2
    assert layer["n"] > 50


def test_facet_wrap_grid_html():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=90),
            "y": rng.normal(size=90),
            "panel": np.repeat(["p1", "p2", "p3"], 30),
        }
    )
    fig = (
        ggplot(df, aes(x="x", y="y"))
        + geom_point(size=3)
        + facet_wrap("panel", ncol=2)
        + labs(title="faceted")
    )
    html = fig.html()
    assert "grid-template-columns" in html
    assert html.count("<iframe") == 3
    assert "p1" in html and "p2" in html and "p3" in html
