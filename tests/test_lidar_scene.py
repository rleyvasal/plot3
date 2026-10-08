"""geom_box3d, theme_lidar, and the coord_3d camera."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from plot3 import aes, coord_3d, geom_box3d, geom_point3d, ggplot, ggsave, theme_lidar
from plot3.build import build_spec
from plot3.themes import _CAT_LIDAR, _LIDAR


def _scene():
    rng = np.random.default_rng(1)
    cloud = pd.DataFrame({
        "x": rng.uniform(-20, 20, 2000), "y": rng.uniform(-20, 20, 2000),
        "z": rng.uniform(-1.8, 3, 2000),
    })
    boxes = pd.DataFrame({
        "x": [5.0, -8.0, 12.0], "y": [3.0, -4.0, 6.0], "z": [-1.0, -1.0, -0.2],
        "l": [4.5, 4.5, 8.0], "w": [1.9, 1.9, 2.5], "h": [1.6, 1.6, 3.2],
        "yaw": [0.0, math.pi / 2, 0.3], "cls": ["car", "car", "truck"],
    })
    return cloud, boxes


def _boxes_layer(boxes, **kw):
    return geom_box3d(
        aes(length="l", width="w", height="h", angle="yaw", **kw), data=boxes,
    )


def test_box3d_draws_twelve_edges_per_box_with_class_colours_and_height_points():
    cloud, boxes = _scene()
    fig = (ggplot(cloud, aes(x="x", y="y", z="z")) + geom_point3d()
           + _boxes_layer(boxes, colour="cls") + theme_lidar())
    spec, _ = build_spec(fig)
    lines = [layer for layer in spec["layers"] if layer["kind"] == "line"]
    assert len(lines) == 2  # car, truck
    assert [len(layer["groups"]) for layer in lines] == [12, 6]  # 6 polylines per box
    assert [layer["n"] for layer in lines] == [36, 18]
    assert [layer["constColor"] for layer in lines] == _CAT_LIDAR[:2]
    # The cloud keeps its height colours in the lidar ramp.
    assert spec["color"]["kind"] == "num"
    assert spec["color"]["ramp"] == _LIDAR
    # Classes and the colour bar each have a title.
    assert [entry["label"] for entry in spec["legend"]] == ["car", "truck"]
    assert spec["labs"]["color"] == "cls"
    assert spec["labs"]["colorBar"] == "z"


def test_box3d_corners_follow_the_heading():
    boxes = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0], "l": [4.0], "w": [2.0],
                          "h": [1.0], "yaw": [math.pi / 2]})
    cloud = pd.DataFrame({"x": [-3.0, 3.0], "y": [-3.0, 3.0], "z": [-1.0, 1.0]})
    spec, _ = build_spec(ggplot(cloud, aes(x="x", y="y", z="z")) + geom_point3d() + _boxes_layer(boxes))
    scales = spec["scales"]
    # Turned a quarter, the 4 m length runs along y: y reaches +-2, x +-1.
    assert scales["x"]["lo"] <= -1.0 and scales["y"]["lo"] <= -2.0


def test_box3d_needs_its_sizes():
    cloud, boxes = _scene()
    with pytest.raises(ValueError, match="length"):
        build_spec(ggplot(cloud, aes(x="x", y="y", z="z")) + geom_point3d()
                   + geom_box3d(aes(width="w", height="h"), data=boxes))


def test_camera_direction_and_zoom_reach_the_spec():
    cloud, _boxes = _scene()
    spec, _ = build_spec(ggplot(cloud, aes(x="x", y="y", z="z")) + geom_point3d()
                         + coord_3d(azim=180, elev=30, zoom=2))
    cam = spec["coord"]["camera"]
    assert cam["zoom"] == 2.0
    dx, dy, dz = cam["dir"]
    assert dx < -0.8 and abs(dy) < 1e-9 and abs(dz - 0.5) < 1e-9
    # Points keep their on-screen grain when the camera moves closer.
    plain, _ = build_spec(ggplot(cloud, aes(x="x", y="y", z="z")) + geom_point3d())
    assert abs(spec["layers"][0]["size"] - plain["layers"][0]["size"] / 2) < 1e-5
    with pytest.raises(ValueError):
        coord_3d(elev=120)
    with pytest.raises(ValueError):
        coord_3d(zoom=0)


def test_lidar_theme_saves_without_box_or_ticks(tmp_path):
    cloud, boxes = _scene()
    fig = (ggplot(cloud, aes(x="x", y="y", z="z")) + geom_point3d()
           + _boxes_layer(boxes, colour="cls") + coord_3d(azim=180, elev=25) + theme_lidar())
    path = tmp_path / "scene.svg"
    ggsave(str(path), fig, width=640, height=360)
    svg = path.read_text(encoding="utf-8")
    assert 'fill="#000000"' in svg  # black page
    assert "<polyline" in svg  # box edges
    assert ">x</text>" not in svg and ">-20</text>" not in svg  # no axis titles or ticks
    assert "<clipPath" in svg
    assert ">car</text>" in svg and ">cls</text>" in svg
