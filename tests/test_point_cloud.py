"""Point clouds: height colours, visible sizes, tall-cloud aspect, back walls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plot3 import aes, coord_3d, geom_point3d, ggplot, ggsave
from plot3.build import _size_breaks, build_spec


def _terrain(n=500):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 50, n)
    y = rng.uniform(0, 30, n)
    return pd.DataFrame({"x": x, "y": y, "elev": np.sin(x / 8) * 3, "g": np.where(x > 25, "a", "b")})


def test_cloud_without_a_colour_is_coloured_by_height_in_viridis():
    spec, _ = build_spec(ggplot(_terrain(), aes(x="x", y="y", z="elev")) + geom_point3d())
    assert spec["color"]["kind"] == "num"
    assert spec["color"]["ramp"][0] == "#440154"
    assert spec["labs"]["color"] == "elev"


def test_a_colour_of_your_own_turns_height_colouring_off():
    const, _ = build_spec(ggplot(_terrain(), aes(x="x", y="y", z="elev")) + geom_point3d(colour="steelblue"))
    assert (const.get("color") or {}).get("kind") != "num"
    groups, _ = build_spec(ggplot(_terrain(), aes(x="x", y="y", z="elev", colour="g")) + geom_point3d())
    assert groups["color"]["kind"] == "cat"


def test_tall_cloud_is_shortened_unless_true_proportions_are_asked_for():
    t = np.linspace(0, 6 * np.pi, 200)
    helix = pd.DataFrame({"x": np.cos(t), "y": np.sin(t), "z": t})
    auto, _ = build_spec(ggplot(helix, aes(x="x", y="y", z="z")) + geom_point3d())
    assert auto["coord"]["aspect"] == "auto"
    ex, ey, ez = auto["coord"]["ext"]
    assert ez == 1.0 and abs(ex - 0.5) < 1e-3 and abs(ey - 0.5) < 1e-3
    true, _ = build_spec(ggplot(helix, aes(x="x", y="y", z="z")) + geom_point3d() + coord_3d(aspect="data"))
    assert true["coord"]["aspect"] == "data" and "ext" not in true["coord"]
    flat, _ = build_spec(ggplot(_terrain(), aes(x="x", y="y", z="elev")) + geom_point3d())
    assert flat["coord"]["aspect"] == "data"


def test_size_legend_breaks_are_round():
    assert _size_breaks(18.85) == [5.0, 10.0, 15.0]
    assert _size_breaks(100) == [25.0, 50.0, 75.0, 100.0]


def test_saved_cloud_draws_back_wall_grid_and_round_colour_bar_labels(tmp_path):
    from plot3 import theme_bw

    path = tmp_path / "cloud.svg"
    ggsave(str(path), ggplot(_terrain(), aes(x="x", y="y", z="elev")) + geom_point3d() + theme_bw(),
           width=480, height=360)
    svg = path.read_text(encoding="utf-8")
    assert 'stroke="#ebebeb"' in svg  # back-wall grid in theme_bw's grid colour
    # Nice colour-bar values only, never the raw data ends (-2.997).
    assert ">2</text>" in svg and ">-2</text>" in svg
    assert "2.99</text>" not in svg
    # Nine of the twelve box edges: the near corner's three are left out.
    assert svg.count('stroke="#333333"') == 9
