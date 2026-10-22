"""Vectorised level, code and density paths match the per-value rules."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import plot3.build as build
import plot3.stat2d as stat2d
from plot3.build import _kde_1d, _remap_codes, _rug_positions
from plot3.scales import Scale, _is_na, ordered_levels
from plot3.stat2d import _kde2d


def _levels_per_value(values, keep_order=False):
    """ordered_levels as it was: test every row for missing, then dedupe."""
    values = list(values)
    present = [v for v in values if not _is_na(v)]
    distinct = list(dict.fromkeys(present))
    out = ordered_levels(distinct, keep_order=keep_order)
    return out + ([None] if len(present) != len(values) else [])


@pytest.mark.parametrize("values", [
    ["Adelie", "Gentoo", None, "Adelie", "Chinstrap", float("nan"), "Gentoo"] * 500,
    [4.0, 6.0, np.nan, 8.0, 10.0, 6.0] * 500,
    [True, False, None, True] * 500,
    [3, 1.0, 1, 2, np.int64(3)] * 500,
    ["Biscoe", pd.NA, "Dream", pd.NA, pd.NaT, "Biscoe"] * 500,
    [float("nan")] * 50 + ["Torgersen"],
    [],
])
@pytest.mark.parametrize("keep_order", [False, True])
def test_ordered_levels_matches_the_per_row_rule(values, keep_order):
    assert ordered_levels(values, keep_order=keep_order) == _levels_per_value(values, keep_order)


def test_remap_codes_matches_the_dict_rule():
    rng = np.random.default_rng(0)
    remap = {0: 2, 1: 0, 3: 1}
    for codes in (rng.integers(-1, 5, 1000).astype(float), rng.integers(0, 4, 1000), np.array([], dtype=float)):
        expected = np.array([remap.get(int(c), 7) for c in codes])
        assert np.array_equal(_remap_codes(codes, remap, 7), expected)
    assert np.array_equal(_remap_codes(np.array([1.9, -0.5]), remap, 7), [0, 2])
    assert np.array_equal(_remap_codes(np.array([2.0]), {}, 0), [0])
    with pytest.raises(ValueError):
        _remap_codes(np.array([0.0, np.nan]), remap, 0)


@pytest.mark.parametrize("series", [
    pd.Series([4.0, 6.0, np.nan, 8.0, 4.0]),
    pd.Series([4, 6, 8, 6]),
    pd.Series([True, False, True]),
    pd.Series(np.array([4.0, 6.5], dtype=np.float32)),
])
def test_rug_positions_on_a_discrete_scale_match_text_lookup(series):
    scale = Scale("cat")
    scale.cats = ["4", "6", "8", "True", "4.0", "6.0", "6.5", "nan"]
    index = {c: i for i, c in enumerate(scale.cats)}
    expected = np.array([index.get(str(v), np.nan) for v in series], dtype=np.float64)
    np.testing.assert_array_equal(_rug_positions(scale, series), expected)


def _exact(module, fn, *args, **kwargs):
    saved = module.EXACT_WORK
    module.EXACT_WORK = math.inf
    try:
        return fn(*args, **kwargs)
    finally:
        module.EXACT_WORK = saved


@pytest.mark.parametrize("values, adjust", [
    (np.random.default_rng(1).normal(4200, 800, 20_000), 1.0),
    (np.random.default_rng(2).lognormal(0, 1.5, 20_000), 1.0),
    (np.random.default_rng(3).normal(17, 2, 20_000), 0.2),
    (np.r_[np.random.default_rng(4).normal(0, 1, 20_000), [1e6]], 1.0),
    (np.random.default_rng(5).integers(2007, 2010, 20_000).astype(float), 1.0),
])
def test_binned_density_matches_the_exact_sum(values, adjust):
    grid, dens = _kde_1d(values, adjust=adjust)
    exact_grid, exact = _exact(build, _kde_1d, values, adjust=adjust)
    np.testing.assert_array_equal(grid, exact_grid)
    assert np.max(np.abs(dens - exact)) <= 2e-4 * exact.max()


def test_small_densities_keep_the_exact_sum():
    values = np.random.default_rng(6).normal(44, 5, 300)
    np.testing.assert_array_equal(_kde_1d(values)[1], _exact(build, _kde_1d, values)[1])
    xs, ys = np.random.default_rng(7).normal(0, 1, (2, 300))
    np.testing.assert_array_equal(_kde2d(xs, ys, 50)[2], _exact(stat2d, _kde2d, xs, ys, 50)[2])


@pytest.mark.parametrize("lims, h", [(None, None), (None, (2.0, 1.0)), ((-1, 1, -0.5, 0.5), None)])
def test_binned_2d_density_matches_the_exact_sum(lims, h):
    rng = np.random.default_rng(8)
    xs, ys = rng.normal(44, 5, 60_000), rng.exponential(2, 60_000)
    gx, gy, z = _kde2d(xs, ys, 100, lims, h)
    ex, ey, exact = _exact(stat2d, _kde2d, xs, ys, 100, lims, h)
    np.testing.assert_array_equal(gx, ex)
    np.testing.assert_array_equal(gy, ey)
    # A quarter-bandwidth grid: the error stays under 0.2% of the peak.
    assert np.max(np.abs(z - exact)) <= 2e-3 * exact.max()
