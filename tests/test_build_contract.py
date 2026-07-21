"""Wire-format / build_spec contracts shared by the viewer."""

from __future__ import annotations

import pytest

from plot3 import (
    aes,
    coord_3d,
    geom_col,
    geom_point,
    geom_point3d,
    geom_surface,
    ggplot,
)
from plot3.build import build_spec
from tests.helpers import assert_layer_kind, assert_spec_3d


def test_2d_spec_has_required_keys(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg", colour="cyl")) + geom_point()
    spec, payloads = build_spec(fig)
    for key in ("v", "is3d", "theme", "labs", "scales", "layers", "gz"):
        assert key in spec
    assert spec["is3d"] is False
    assert "x" in spec["scales"] and "y" in spec["scales"]
    assert payloads


def test_3d_spec_coord_and_z(cloud):
    fig = (
        ggplot(cloud, aes(x="x", y="y", z="z", colour="intensity"))
        + geom_point3d(size=0.01)
        + coord_3d(aspect="data", size_mode="scene")
    )
    spec, _ = build_spec(fig)
    assert_spec_3d(spec)
    assert spec["coord"]["aspect"] == "data"
    assert spec["coord"]["sizeMode"] == "scene"
    assert_layer_kind(spec, "point")


def test_surface_spec_has_indices(grid):
    fig = ggplot(grid, aes(x="x", y="y", z="height", fill="height")) + geom_surface()
    spec, payloads = build_spec(fig)
    layer = spec["layers"][0]
    assert layer["kind"] == "surface"
    assert layer["indices"]["count"] > 0
    assert any("idx" in pid for pid, _ in payloads)


def test_payload_ids_unique(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    _spec, payloads = build_spec(fig)
    ids = [pid for pid, _ in payloads]
    assert len(ids) == len(set(ids))


def test_error_messages_are_actionable(cloud):
    with pytest.raises(ValueError, match="2D-only"):
        build_spec(
            ggplot(cloud, aes(x="x", y="y", z="z")) + geom_col()
        )
    with pytest.raises(ValueError, match="coord_3d"):
        build_spec(ggplot(cloud, aes(x="x", y="y")) + geom_point() + coord_3d())
