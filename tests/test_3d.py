"""Formal 3D API: geom_point3d, coord_3d, validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
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
)
from plot3.build import build_spec


def _cloud(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "z": rng.normal(size=n),
            "intensity": rng.uniform(0, 1, n),
        }
    )


def test_aes_z_enables_3d_and_encodes_z():
    df = _cloud()
    fig = (
        ggplot(df, aes(x="x", y="y", z="z", colour="intensity"))
        + geom_point(size=0.01)
    )
    spec, payloads = build_spec(fig)
    assert spec["is3d"] is True
    assert "z" in spec["scales"]
    assert spec["layers"][0]["kind"] == "point"
    assert "z" in spec["layers"][0]
    assert len(payloads) >= 3


def test_geom_point3d_matches_point_kind():
    df = _cloud()
    a = build_spec(
        ggplot(df, aes(x="x", y="y", z="z")) + geom_point(size=0.02)
    )[0]
    b = build_spec(
        ggplot(df, aes(x="x", y="y", z="z")) + geom_point3d(size=0.02)
    )[0]
    assert a["layers"][0]["kind"] == b["layers"][0]["kind"] == "point"
    assert a["is3d"] and b["is3d"]
    assert a["layers"][0]["size"] == b["layers"][0]["size"] == 0.02


def test_coord_3d_in_spec():
    df = _cloud()
    fig = (
        ggplot(df, aes(x="x", y="y", z="z", colour="intensity"))
        + geom_point3d(size=0.008)
        + coord_3d(aspect="equal", size_mode="screen", max_points=50)
        + scale_colour_viridis_c(option="turbo")
        + labs(title="cloud")
    )
    spec, _ = build_spec(fig)
    assert spec["coord"]["aspect"] == "equal"
    assert spec["coord"]["sizeMode"] == "screen"
    assert spec["coord"]["maxPoints"] == 50
    # max_points stride-subsamples
    assert spec["layers"][0]["n"] <= 50


def test_mix_2d_3d_layers_errors():
    df = _cloud()
    fig = (
        ggplot(df, aes(x="x", y="y"))
        + geom_point()
        + geom_point(aes(z="z"))
    )
    with pytest.raises(ValueError, match="mix of 2D and 3D"):
        build_spec(fig)


def test_2d_only_geom_rejected_in_3d():
    df = _cloud()
    fig = (
        ggplot(df, aes(x="x", y="y", z="z"))
        + geom_col()
    )
    with pytest.raises(ValueError, match="2D-only"):
        build_spec(fig)


def test_coord_3d_requires_3d_figure():
    df = _cloud()
    fig = ggplot(df, aes(x="x", y="y")) + geom_point() + coord_3d()
    with pytest.raises(ValueError, match="coord_3d"):
        build_spec(fig)


def test_facet_wrap_rejected_in_3d():
    df = _cloud()
    df["panel"] = np.where(df["x"] > 0, "a", "b")
    fig = (
        ggplot(df, aes(x="x", y="y", z="z"))
        + geom_point3d()
        + facet_wrap("panel")
    )
    with pytest.raises(ValueError, match="facet_wrap"):
        build_spec(fig)


def test_3d_html_builds():
    df = _cloud(80)
    html = (
        ggplot(df, aes(x="x", y="y", z="z", colour="intensity"))
        + geom_point3d(size=0.01)
        + coord_3d()
    )._repr_html_()
    assert len(html) > 500
