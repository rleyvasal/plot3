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
    geom_surface,
    geom_isosurface,
    ggplot,
    labs,
    scale_colour_viridis_c,
    stat_density_3d,
)
from plot3.build import build_spec
from plot3.stats3d import density_grid_3d, isosurface_levels, regular_grid_mesh


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


def _grid(nx=12, ny=10):
    xs = np.linspace(-2, 2, nx)
    ys = np.linspace(-1.5, 1.5, ny)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.sin(xx) * np.cos(yy)
    return pd.DataFrame(
        {"x": xx.ravel(), "y": yy.ravel(), "height": zz.ravel()}
    )


def test_regular_grid_mesh_counts():
    df = _grid(5, 4)
    verts, indices, nx, ny = regular_grid_mesh(df, "x", "y", "height")
    assert nx == 5 and ny == 4
    assert len(verts) == 20
    assert indices.shape == ((5 - 1) * (4 - 1) * 2, 3)


def test_geom_surface_builds_mesh_layer():
    df = _grid(12, 10)
    fig = (
        ggplot(df, aes(x="x", y="y", z="height", fill="height"))
        + geom_surface()
        + coord_3d()
        + scale_colour_viridis_c(option="viridis")
        + labs(title="surface")
    )
    spec, payloads = build_spec(fig)
    assert spec["is3d"] is True
    layer = spec["layers"][0]
    assert layer["kind"] == "surface"
    assert layer["n"] == 12 * 10
    assert layer["indices"]["count"] == (12 - 1) * (10 - 1) * 2 * 3
    assert any(pid.endswith("idx") or "idx" in pid for pid, _ in payloads)
    html = fig._repr_html_()
    assert len(html) > 500


def test_geom_surface_rejects_incomplete_grid():
    df = _grid(6, 5).iloc[:-3]
    fig = ggplot(df, aes(x="x", y="y", z="height")) + geom_surface()
    with pytest.raises(ValueError, match="complete regular"):
        build_spec(fig)


def test_geom_surface_wireframe_flag():
    df = _grid(6, 6)
    fig = ggplot(df, aes(x="x", y="y", z="height")) + geom_surface(wireframe=True)
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["wireframe"] is True


def test_density_grid_and_isosurface_levels():
    rng = np.random.default_rng(0)
    # Tight cluster so density peaks clearly.
    pts = rng.normal(scale=0.3, size=(400, 3))
    dens, xs, ys, zs = density_grid_3d(pts, n=16)
    assert dens.shape == (16, 16, 16)
    assert dens.max() == pytest.approx(1.0, rel=1e-6)
    verts, faces, used = isosurface_levels(pts, [0.2, 0.5], n=16)
    assert len(used) >= 1
    assert len(verts) > 0
    assert faces.shape[1] == 3
    assert faces.max() < len(verts)


def test_geom_isosurface_builds():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=300),
            "y": rng.normal(size=300),
            "z": rng.normal(size=300),
        }
    )
    fig = (
        ggplot(df, aes(x="x", y="y", z="z"))
        + stat_density_3d(n=16)
        + geom_isosurface(levels=[0.3, 0.6], n=16)
        + coord_3d()
    )
    spec, payloads = build_spec(fig)
    assert spec["is3d"] is True
    layer = spec["layers"][0]
    assert layer["kind"] == "isosurface"
    assert layer["indices"]["count"] > 0
    assert any("idx" in pid for pid, _ in payloads)
    assert len(fig._repr_html_()) > 500
