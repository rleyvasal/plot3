"""Smoke tests for data-science geoms (col / bar / histogram)."""

from __future__ import annotations

import pandas as pd

from plot3 import (
    aes,
    geom_bar,
    geom_boxplot,
    geom_col,
    geom_histogram,
    geom_point,
    ggplot,
    labs,
)


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
    # Building the document should not raise.
    assert len(bar._repr_html_()) > 500
    assert len(hist._repr_html_()) > 500


def test_point_still_works():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 4.0]})
    fig = ggplot(df, aes(x="x", y="y")) + geom_point()
    assert len(fig._repr_html_()) > 500


def test_geom_boxplot_builds_and_marks_outliers():
    rng = __import__("numpy").random.default_rng(0)
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
    # Spec should include box layer channels.
    from plot3.build import build_spec

    spec, _payloads = build_spec(fig)
    layer = spec["layers"][0]
    assert layer["kind"] == "box"
    assert layer["n"] == 2
    assert "ymin" in layer and "middle" in layer and "ymax" in layer
    assert layer["nOut"] >= 1
