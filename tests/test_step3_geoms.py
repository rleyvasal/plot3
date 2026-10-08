"""coord_flip, geom_tile, geom_area, geom_step, segments, rects, Q-Q, ECDF, layer data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes,
    coord_flip,
    geom_area,
    geom_bar,
    geom_boxplot,
    geom_col,
    geom_density,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    geom_qq,
    geom_qq_line,
    geom_rect,
    geom_segment,
    geom_step,
    geom_text,
    geom_tile,
    ggplot,
    ggsave,
    stat_ecdf,
)
from plot3.build import build_spec, expand_stat_geom
from plot3 import special


@pytest.fixture
def heat():
    return pd.DataFrame(
        [(r, c, float(i)) for i, (r, c) in enumerate((r, c) for r in ["g1", "g2", "g3"] for c in ["a", "b"])],
        columns=["row", "col", "v"],
    )


def test_tiles_cover_every_cell_including_the_last(heat):
    spec, _ = build_spec(ggplot(heat, aes(x="col", y="row", fill="v")) + geom_tile())
    assert spec["scales"]["x"]["cats"] == ["a", "b"]
    assert spec["scales"]["y"]["cats"] == ["g1", "g2", "g3"]
    layer = expand_stat_geom(geom_tile(), aes(x="col", y="row", fill="v"), heat)
    assert len(layer._groups) == 6
    # The last column / row edge (n - 0.5) must survive the categorical axis.
    assert spec["layers"][0]["n"] == 24


def test_tile_corners_on_the_outer_edge_are_kept(heat):
    from plot3.build import build_spec as _bs

    spec, pairs = _bs(ggplot(heat, aes(x="col", y="row", fill="v")) + geom_tile())
    from plot3.static import _decode

    layer = spec["layers"][0]
    ys = _decode(dict(pairs)[layer["y"]["id"]], layer["y"]["dtype"], True)
    assert np.all(np.isfinite(ys.astype(float)))


def test_stacked_area_first_level_on_top():
    frame = pd.DataFrame({"x": [0, 1, 0, 1], "y": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    layer = expand_stat_geom(geom_area(), aes(x="x", y="y", fill="g"), frame)
    a_start, a_n = layer._groups[0]
    upper_a = layer.data_override.y.iloc[a_start + a_n // 2:a_start + a_n].to_numpy()
    assert upper_a.tolist() == [6.0, 4.0]  # a sits on b: 3+1 at x=0, 4+2 at x=1 (reversed)


def test_area_fill_sums_to_one():
    frame = pd.DataFrame({"x": [0, 1, 0, 1], "y": [1.0, 3.0, 3.0, 1.0], "g": ["a", "a", "b", "b"]})
    layer = expand_stat_geom(geom_area(position="fill"), aes(x="x", y="y", fill="g"), frame)
    assert layer.data_override.y.max() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "direction, xs, ys",
    [("hv", [0, 1, 1, 2, 2], [0, 0, 5, 5, 1]), ("vh", [0, 0, 1, 1, 2], [0, 5, 5, 1, 1])],
)
def test_step_paths(direction, xs, ys):
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 5.0, 1.0]})
    layer = expand_stat_geom(geom_step(direction=direction), aes(x="x", y="y"), frame)
    assert layer.data_override.x.tolist() == xs
    assert layer.data_override.y.tolist() == ys


def test_segments_and_rects_from_layer_data():
    base = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    periods = pd.DataFrame({"s": [1.0, 5.0], "e": [2.0, 7.0], "lo": [0.0, 0.0], "hi": [10.0, 10.0]})
    arrows = pd.DataFrame({"x0": [1.0], "y0": [1.0], "x1": [4.0], "y1": [6.0]})
    spec, _ = build_spec(
        ggplot(base, aes(x="x", y="y")) + geom_point()
        + geom_rect(aes(xmin="s", xmax="e", ymin="lo", ymax="hi"), data=periods)
        + geom_segment(aes(x="x0", y="y0", xend="x1", yend="y1"), data=arrows)
    )
    kinds = [layer["kind"] for layer in spec["layers"]]
    assert kinds == ["point", "poly", "line"]
    assert spec["layers"][1]["groups"] == [[0, 4], [4, 4]]


def test_layer_data_without_plot_data():
    rows = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    spec, _ = build_spec(ggplot() + geom_point(aes(x="x", y="y"), data=rows))
    assert spec["layers"][0]["n"] == 2


def test_qq_uses_r_ppoints_and_a_quartile_line():
    values = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    frame = pd.DataFrame({"v": values})
    points = expand_stat_geom(geom_qq(), aes(sample="v"), frame)
    a = 3 / 8
    expected = special.qnorm((np.arange(1, 6) - a) / (5 + 1 - 2 * a))
    assert points.data_override.x.to_numpy() == pytest.approx(expected)
    assert points.data_override.y.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    line = expand_stat_geom(geom_qq_line(), aes(sample="v"), frame)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    t1, t3 = special.qnorm(np.array([0.25, 0.75]))
    slope = (q3 - q1) / (t3 - t1)
    xs, ys = line.data_override.x.to_numpy(), line.data_override.y.to_numpy()
    assert (ys[1] - ys[0]) / (xs[1] - xs[0]) == pytest.approx(slope)


def test_ecdf_steps_from_zero_to_one():
    frame = pd.DataFrame({"v": [3.0, 1.0, 2.0, 2.0]})
    layer = expand_stat_geom(stat_ecdf(), aes(x="v"), frame)
    ys = layer.data_override.y.to_numpy()
    assert ys[0] == 0.0 and ys[-1] == 1.0
    assert sorted(set(ys.tolist())) == [0.0, 0.25, 0.75, 1.0]
    assert layer._axis_labels["y"] == "ECDF"


# ── coord_flip ───────────────────────────────────────────────────────────────


@pytest.fixture
def groups():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"g": rng.choice(["alpha", "beta"], 40), "v": rng.normal(5, 1, 40)})
    return frame


def test_flipped_bars_put_categories_on_y(groups):
    spec, _ = build_spec(ggplot(groups, aes(x="g")) + geom_bar() + coord_flip())
    assert spec["scales"]["y"]["cats"] == ["alpha", "beta"]
    assert spec["scales"]["x"]["kind"] == "num"
    assert spec["labs"]["x"] == "count" and spec["labs"]["y"] == "g"


def test_flipped_boxplot_lowers_to_shapes(groups):
    spec, _ = build_spec(ggplot(groups, aes(x="g", y="v")) + geom_boxplot() + coord_flip())
    assert {layer["kind"] for layer in spec["layers"]} <= {"poly", "line", "point"}
    assert spec["scales"]["y"]["cats"] == ["alpha", "beta"]
    assert spec["labs"]["x"] == "v"


def test_flipped_errorbars_and_reference_line():
    frame = pd.DataFrame({"g": ["a", "b"], "m": [2.0, 3.0], "lo": [1.5, 2.5], "hi": [2.5, 3.5]})
    spec, _ = build_spec(
        ggplot(frame, aes(x="g", y="m")) + geom_col()
        + geom_errorbar(aes(ymin="lo", ymax="hi")) + geom_hline(yintercept=2.5)
        + coord_flip()
    )
    assert spec["refs"][0]["kind"] == "vline" and spec["refs"][0]["value"] == 2.5
    assert spec["scales"]["y"]["cats"] == ["a", "b"]


def test_flipped_density_and_line_keep_their_order():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": np.arange(5.0), "y": [3.0, 1.0, 4.0, 1.0, 5.0]})
    spec, _ = build_spec(ggplot(frame, aes(x="x", y="y")) + geom_line() + coord_flip())
    assert spec["labs"]["x"] == "y" and spec["labs"]["y"] == "x"
    dens = pd.DataFrame({"v": rng.normal(size=50)})
    spec, _ = build_spec(ggplot(dens, aes(x="v")) + geom_density() + coord_flip())
    assert spec["labs"]["x"] == "density" and spec["labs"]["y"] == "v"


def test_flipped_text_moves_with_its_points():
    frame = pd.DataFrame({"g": ["a", "b"], "v": [1.0, 2.0]})
    spec, _ = build_spec(
        ggplot(frame, aes(x="g", y="v", label="g")) + geom_point() + geom_text(nudge_y=0.5) + coord_flip()
    )
    positions = {ann["text"]: (ann["x"], ann["y"]) for ann in spec["ann"]}
    assert positions == {"a": (1.5, 0.0), "b": (2.5, 1.0)}


def test_static_flip_and_tile_save(tmp_path, groups, heat):
    ggsave(str(tmp_path / "flip.svg"), ggplot(groups, aes(x="g", y="v")) + geom_boxplot() + coord_flip())
    ggsave(str(tmp_path / "tile.svg"), ggplot(heat, aes(x="col", y="row", fill="v")) + geom_tile() + geom_text(aes(label="v")))
    svg = (tmp_path / "tile.svg").read_text()
    assert ">5<" in svg  # numeric labels are formatted, not 5.0
