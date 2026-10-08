"""Step 10: 2D bins, hexagons, densities, contours, ellipses, composition
heights, and adaptive sampling of narrow peaks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes, geom_bin_2d, geom_contour, geom_density_2d, geom_density_2d_filled,
    geom_function, geom_hex, geom_point, ggplot, ggsave, plot_annotation,
    plot_layout, stat_ellipse,
)
from plot3.build import build_spec, expand_stat_geom
from plot3.special import qf


def _cloud(n=600, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"x": rng.normal(size=n)})
    d["y"] = 0.6 * d.x + 0.8 * rng.normal(size=n)
    d["g"] = rng.choice(["a", "b"], n)
    return d


def test_qf_matches_r():
    assert float(qf(0.95, 2, 30)) == pytest.approx(3.315830, abs=1e-5)
    assert float(qf(0.5, 5, 10)) == pytest.approx(0.9319332, abs=1e-6)


def test_bin_2d_counts_every_row():
    d = _cloud()
    out = expand_stat_geom(geom_bin_2d(bins=10), aes("x", "y"), d)
    counts = out.data_override.groupby(out.data_override.index // 4)["count"].first()
    assert counts.sum() == len(d)
    spec, _ = build_spec(ggplot(d, aes("x", "y")) + geom_bin_2d(bins=10))
    assert spec["color"]["kind"] == "num" and spec["labs"]["color"] == "count"


def test_hexagons_count_every_row_and_have_six_corners():
    d = _cloud()
    out = expand_stat_geom(geom_hex(bins=12), aes("x", "y"), d)
    frame = out.data_override
    assert all(count == 6 for _start, count in out._groups)
    assert frame["count"].iloc[::6].sum() == len(d)


def test_density_2d_lines_per_group_and_filled_bands():
    d = _cloud()
    spec, _ = build_spec(ggplot(d, aes("x", "y")) + geom_density_2d())
    assert spec["layers"][0]["kind"] == "line" and spec["layers"][0]["constColor"] == "#3366ff"
    grouped, _ = build_spec(ggplot(d, aes("x", "y", colour="g")) + geom_density_2d())
    assert grouped["color"]["cats"] == ["a", "b"]
    filled, _ = build_spec(ggplot(d, aes("x", "y")) + geom_density_2d_filled())
    cats = filled["color"]["cats"]
    assert cats[0].startswith("(0, ") and filled["color"]["palette"][0] == "#440154"
    assert filled["labs"]["color"] == "level"


def test_contour_of_gridded_z_stays_2d():
    grid = pd.DataFrame([(i, j, np.sin(i / 3) * np.cos(j / 4)) for i in range(30) for j in range(20)],
                        columns=["x", "y", "z"])
    spec, _ = build_spec(ggplot(grid, aes("x", "y", z="z")) + geom_contour(bins=6))
    assert spec["is3d"] is False and spec["layers"][0]["n"] > 50
    with pytest.raises(ValueError, match="z="):
        build_spec(ggplot(grid, aes("x", "y")) + geom_contour())


def test_ellipse_matches_the_normal_formula():
    rng = np.random.default_rng(3)
    pts = rng.multivariate_normal([1.0, 2.0], [[1.0, 0.5], [0.5, 2.0]], size=400)
    d = pd.DataFrame(pts, columns=["x", "y"])
    out = expand_stat_geom(stat_ellipse(type="norm", level=0.95), aes("x", "y"), d)
    curve = out.data_override[["x", "y"]].to_numpy()
    centre = pts.mean(axis=0)
    inv = np.linalg.inv(np.cov(pts, rowvar=False))
    d2 = np.einsum("ij,jk,ik->i", curve - centre, inv, curve - centre)
    assert np.allclose(d2, 2 * float(qf(0.95, 2, len(pts) - 1)), rtol=1e-6)
    robust = expand_stat_geom(stat_ellipse(), aes("x", "y"), d)  # type="t"
    assert len(robust.data_override) == 51
    with pytest.raises(ValueError):
        stat_ellipse(type="box")


def test_composition_height_follows_its_rows():
    d = _cloud(20)
    a = ggplot(d, aes("x", "y")) + geom_point()
    assert (a | a).figure_height() == "480px"
    assert ((a | a) / a).figure_height() == "800px"
    assert (((a | a) / a) + plot_annotation(title="T")).figure_height() == "840px"
    assert ((a | a) + plot_layout(height=700)).figure_height() == "700px"
    assert 'height:800px' in ((a | a) / a)._repr_html_()


@pytest.mark.parametrize("formula,top", [
    ("y = exp(-2000*(x-0.013)^2)", 1.0),
    ("y = 1/(1 + 1000*(x - 2.01)^2)", 1.0),
])
def test_narrow_peaks_reach_their_height(formula, top):
    spec, _ = build_spec(ggplot() + geom_function(formula))
    assert spec["scales"]["y"]["hi"] == pytest.approx(top, abs=0.01)


def test_a_chosen_n_is_kept():
    spec, _ = build_spec(ggplot() + geom_function("y = exp(-2000*(x-0.013)^2)", n=200))
    assert spec["layers"][0]["n"] == 200


def test_saved_hexagons(tmp_path):
    path = tmp_path / "hex.svg"
    ggsave(str(path), ggplot(_cloud(), aes("x", "y")) + geom_hex(bins=10), width=400, height=300)
    assert path.read_text(encoding="utf-8").count("<polygon") > 20
