"""Semantic checks for stats that should match base R / ggplot2 conventions.

These tests do not require R. When R is available, test_r_oracle_parity.py
cross-checks the same quantities against ggplot2/base R.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plot3 import aes, geom_boxplot, geom_density, geom_histogram, ggplot
from plot3.build import build_spec, expand_stat_geom, _boxplot_stats
from plot3.stats3d import density_grid_3d, isosurface_levels
from tests.helpers import nearly_equal, tukey_boxplot_stats


def test_boxplot_stats_match_python_reference(box_frame):
    for g, piece in box_frame.groupby("g", sort=False):
        ref = tukey_boxplot_stats(piece["y"].to_numpy())
        got = _boxplot_stats(piece["y"].to_numpy(), coef=1.5)
        assert got is not None
        ymin, lower, middle, upper, ymax, outliers = got
        assert nearly_equal(ymin, ref["ymin"], atol=1e-9)
        assert nearly_equal(lower, ref["lower"], atol=1e-9)
        assert nearly_equal(middle, ref["middle"], atol=1e-9)
        assert nearly_equal(upper, ref["upper"], atol=1e-9)
        assert nearly_equal(ymax, ref["ymax"], atol=1e-9)
        assert nearly_equal(sorted(outliers), sorted(ref["outliers"]), atol=1e-9)


def test_geom_boxplot_layer_encodes_groups(box_frame):
    fig = ggplot(box_frame, aes(x="g", y="y")) + geom_boxplot()
    spec, _ = build_spec(fig)
    layer = spec["layers"][0]
    assert layer["kind"] == "box"
    assert layer["n"] == 2
    assert layer["nOut"] >= 1


def test_histogram_expand_bin_count(cars):
    from plot3.geoms import geom_histogram as GH

    geom = GH(bins=5)
    expanded = expand_stat_geom(geom, {"x": "mpg"}, cars)
    assert expanded.kind == "col"
    frame = expanded.data_override
    assert frame is not None
    assert len(frame) == 5
    assert set(frame.columns) >= {"x", "y"}
    assert frame["y"].sum() == len(cars)


def test_density_expand_nonnegative_and_integrates(cars):
    from plot3.geoms import geom_density as GD

    geom = GD(n=64, fill=True)
    expanded = expand_stat_geom(geom, {"x": "mpg"}, cars)
    frame = expanded.data_override
    assert frame is not None
    assert (frame["y"] >= -1e-12).all()
    # Trapezoid integral of KDE should be near 1
    x = frame["x"].to_numpy()
    y = frame["y"].to_numpy()
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area = float(trapz(y, x))
    assert 0.85 < area < 1.15


def test_density_grid_3d_normalized(cloud):
    pts = cloud[["x", "y", "z"]].to_numpy()
    dens, xs, ys, zs = density_grid_3d(pts, n=12)
    assert dens.shape == (12, 12, 12)
    assert dens.min() >= 0
    assert dens.max() == pytest.approx(1.0, rel=1e-6)
    assert len(xs) == 12


def test_isosurface_levels_produce_mesh(cloud):
    pts = cloud[["x", "y", "z"]].to_numpy()
    verts, faces, used = isosurface_levels(pts, [0.2, 0.5], n=12)
    assert len(used) >= 1
    assert len(verts) > 0
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert int(faces.max()) < len(verts)
