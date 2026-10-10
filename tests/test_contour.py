"""Vectorized marching squares, active-cell refinement, and contourpy."""

from __future__ import annotations

import subprocess
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

from plot3.contour import (
    _backend_name,
    _contour_contourpy,
    _contour_lines,
    _contour_numpy,
    _refine_active_cells,
    _reset_contour_backend,
    _version_at_least,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_contour_cache():
    _reset_contour_backend()
    yield
    _reset_contour_backend()


def _oracle_march(xs, ys, field, level):
    """Original pure-Python marching squares, kept as a point-set oracle.

    Polyline order and the start point may differ from the vectorized marcher.
    Joins use rounded coordinates, so very small curves can split; the tests
    that cover that scale check the vectorized result on its own.
    """
    segments = []
    ny, nx = field.shape
    for j in range(ny - 1):
        for i in range(nx - 1):
            corners = (
                (float(xs[i]), float(ys[j]), float(field[j, i])),
                (float(xs[i + 1]), float(ys[j]), float(field[j, i + 1])),
                (float(xs[i + 1]), float(ys[j + 1]), float(field[j + 1, i + 1])),
                (float(xs[i]), float(ys[j + 1]), float(field[j + 1, i])),
            )
            if not all(np.isfinite(corner[2]) for corner in corners):
                continue
            segments.extend(_oracle_cell(corners, level))
    return _oracle_chain(segments)


def _oracle_cell(corners, level):
    inside = [corner[2] >= level for corner in corners]
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    hits = [edge for edge, (a, b) in enumerate(pairs) if inside[a] != inside[b]]
    if len(hits) == 2:
        return [(_oracle_cross(corners, pairs[hits[0]], level), _oracle_cross(corners, pairs[hits[1]], level))]
    if len(hits) != 4:
        return []
    center = sum(corner[2] for corner in corners) / 4.0
    pairing = ((0, 1), (2, 3)) if center >= level else ((0, 3), (1, 2))
    return [
        (_oracle_cross(corners, pairs[a], level), _oracle_cross(corners, pairs[b], level))
        for a, b in pairing
    ]


def _oracle_cross(corners, edge, level):
    a = corners[edge[0]]
    b = corners[edge[1]]
    span = b[2] - a[2]
    t = 0.5 if span == 0.0 else (level - a[2]) / span
    t = min(1.0, max(0.0, t))
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _oracle_chain(segments):
    if not segments:
        return []

    def key(point):
        return (round(point[0], 8), round(point[1], 8))

    unused = set(range(len(segments)))
    by_point = {}
    for index, (start, end) in enumerate(segments):
        by_point.setdefault(key(start), []).append((index, 0))
        by_point.setdefault(key(end), []).append((index, 1))
    polylines = []
    while unused:
        index = unused.pop()
        start, end = segments[index]
        poly = [start, end]
        _oracle_extend(poly, unused, segments, by_point, key, forward=True)
        _oracle_extend(poly, unused, segments, by_point, key, forward=False)
        if key(poly[0]) == key(poly[-1]) and len(poly) > 2:
            poly = poly[:-1]
            poly.append(poly[0])
        elif abs(poly[0][0] - poly[-1][0]) <= 1e-8 and abs(poly[0][1] - poly[-1][1]) <= 1e-8:
            poly.append(poly[0])
        polylines.append(poly)
    return polylines


def _oracle_extend(poly, unused, segments, by_point, key, forward):
    while True:
        tip = poly[-1] if forward else poly[0]
        nxt = None
        for index, _end in by_point.get(key(tip), []):
            if index in unused:
                nxt = index
                break
        if nxt is None:
            return
        unused.remove(nxt)
        start, end = segments[nxt]
        other = end if key(start) == key(tip) else start
        if forward:
            poly.append(other)
        else:
            poly.insert(0, other)


def _field(kind, n=40, lo=-2.0, hi=2.0):
    xs = np.linspace(lo, hi, n)
    ys = np.linspace(lo, hi, n)
    xx, yy = np.meshgrid(xs, ys)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "circle":
            field = xx**2 + yy**2 - 1.0
        elif kind == "saddle":
            field = xx * yy
        elif kind == "hyperbola":
            field = xx * yy - 0.01
        elif kind == "two":
            field = (xx**2 + yy**2 - 0.25) * ((xx - 1.0) ** 2 + yy**2 - 0.16)
        elif kind == "nan":
            field = np.log(xx) + yy
        elif kind == "open":
            field = yy - xx
        else:
            raise AssertionError(kind)
    return xs, ys, np.asarray(field, dtype=np.float64)


def _points(polylines):
    rows = [point for poly in polylines for point in poly]
    if not rows:
        return np.zeros((0, 2))
    return np.asarray(rows, dtype=np.float64)


def _assert_same_points(left, right, atol=1e-12):
    got = _points(left)
    expect = _points(right)
    assert got.size and expect.size
    gap_forward = np.linalg.norm(got[:, None, :] - expect[None, :, :], axis=-1).min(axis=1).max()
    gap_back = np.linalg.norm(expect[:, None, :] - got[None, :, :], axis=-1).min(axis=1).max()
    assert float(gap_forward) <= atol
    assert float(gap_back) <= atol


@pytest.mark.parametrize("kind", ["circle", "saddle", "hyperbola", "two", "nan", "open"])
def test_numpy_matches_the_cell_loop(kind):
    xs, ys, field = _field(kind)
    old = _oracle_march(xs, ys, field, 0.0)
    new = _contour_numpy(xs, ys, field, 0.0)
    _assert_same_points(old, new)
    if kind == "circle":
        assert len(new) == 1
        assert new[0][0] == new[0][-1]
    if kind == "two":
        assert len(new) == 2
    if kind == "saddle":
        pts = _points(new)
        assert np.all((np.abs(pts[:, 0]) <= 1e-8) | (np.abs(pts[:, 1]) <= 1e-8))
    if kind == "nan":
        assert np.all(_points(new)[:, 0] > 0.0)
    if kind == "open":
        assert len(new) == 1
        for point in (new[0][0], new[0][-1]):
            on_box = min(abs(point[0] - lo) for lo in (-2.0, 2.0)) < 1e-8
            on_box = on_box or min(abs(point[1] - lo) for lo in (-2.0, 2.0)) < 1e-8
            assert on_box


def test_numpy_skips_nonfinite_corners():
    xs, ys, field = _field("open", n=25)
    field = field.copy()
    field[0, 0] = np.inf
    field[4, 4] = -np.inf
    field[8, 2] = np.nan
    new = _contour_numpy(xs, ys, field, 0.0)
    _assert_same_points(new, _oracle_march(xs, ys, field, 0.0))
    assert np.isfinite(_points(new)).all()


@pytest.mark.parametrize(
    ("lo", "radius"),
    [(2e-8, 1e-8), (2e6, 1e6)],
)
@pytest.mark.parametrize("backend", ["numpy", "contourpy"])
def test_extreme_scales_stay_one_closed_loop(lo, radius, backend, monkeypatch):
    if backend == "contourpy":
        pytest.importorskip("contourpy")
    monkeypatch.setenv("PLOT3_CONTOUR_BACKEND", backend)
    _reset_contour_backend()
    xs, ys, field = _field("circle", n=40, lo=-lo, hi=lo)
    field = xs[None, :] ** 2 + ys[:, None] ** 2 - radius**2
    lines = _contour_lines(xs, ys, field, 0.0)
    assert len(lines) == 1
    poly = np.asarray(lines[0], dtype=np.float64)
    assert np.allclose(poly[0], poly[-1])
    radial = np.abs(np.hypot(poly[:, 0], poly[:, 1]) - radius) / radius
    assert float(np.median(radial)) < 0.01
    assert float(poly.max() - poly.min()) > radius


def test_contourpy_matches_numpy_away_from_saddles():
    pytest.importorskip("contourpy")
    for kind in ("circle", "hyperbola", "two", "nan", "open"):
        xs, ys, field = _field(kind, n=36)
        _assert_same_points(
            _contour_contourpy(xs, ys, field, 0.0),
            _contour_numpy(xs, ys, field, 0.0),
        )


def test_contourpy_saddle_point_set_matches_and_loop_is_closed():
    pytest.importorskip("contourpy")
    xs, ys, field = _field("saddle", n=32)
    _assert_same_points(
        _contour_contourpy(xs, ys, field, 0.0),
        _contour_numpy(xs, ys, field, 0.0),
    )
    xs, ys, field = _field("circle", n=32)
    lines = _contour_contourpy(xs, ys, field, 0.0)
    assert len(lines) == 1
    assert lines[0][0] == lines[0][-1]
    assert len(lines[0]) >= 3


def test_version_floor_and_backend_selection(monkeypatch):
    assert _version_at_least("1.0.0", 1, 0)
    assert _version_at_least("1.4.0", 1, 0)
    assert not _version_at_least("0.9.9", 1, 0)

    monkeypatch.setenv("PLOT3_CONTOUR_BACKEND", "numpy")
    _reset_contour_backend()
    assert _backend_name() == "numpy"

    monkeypatch.setenv("PLOT3_CONTOUR_BACKEND", "gpu")
    with pytest.raises(ValueError, match="auto, numpy, or contourpy"):
        _contour_lines(np.linspace(0, 1, 2), np.linspace(0, 1, 2), np.zeros((2, 2)), 0.0)

    monkeypatch.setitem(sys.modules, "contourpy", None)
    monkeypatch.setenv("PLOT3_CONTOUR_BACKEND", "auto")
    _reset_contour_backend()
    assert _backend_name() == "numpy"
    monkeypatch.setenv("PLOT3_CONTOUR_BACKEND", "contourpy")
    _reset_contour_backend()
    with pytest.raises(ValueError, match=r"pip install contourpy"):
        _contour_lines(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), np.zeros((4, 4)), 0.0)

    fake = types.ModuleType("contourpy")
    fake.__version__ = "0.4.2"
    monkeypatch.setitem(sys.modules, "contourpy", fake)
    monkeypatch.setenv("PLOT3_CONTOUR_BACKEND", "auto")
    _reset_contour_backend()
    assert _backend_name() == "numpy"
    monkeypatch.setenv("PLOT3_CONTOUR_BACKEND", "contourpy")
    _reset_contour_backend()
    with pytest.raises(ValueError, match="contourpy>=1.0"):
        _backend_name()


def test_auto_uses_contourpy_when_importable(monkeypatch):
    pytest.importorskip("contourpy")
    monkeypatch.delenv("PLOT3_CONTOUR_BACKEND", raising=False)
    _reset_contour_backend()
    assert _backend_name() == "contourpy"


def test_contourpy_failure_falls_back_once_unless_requested(monkeypatch):
    pytest.importorskip("contourpy")
    xs, ys, field = _field("circle", n=12)

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("plot3.contour._contour_contourpy", boom)
    monkeypatch.setenv("PLOT3_CONTOUR_BACKEND", "auto")
    _reset_contour_backend()
    with pytest.warns(
        UserWarning,
        match=r"plot3: contourpy failed \(boom\), using the NumPy contour backend",
    ):
        fell_back = _contour_lines(xs, ys, field, 0.0)
    assert fell_back
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _contour_lines(xs, ys, field, 0.0)

    monkeypatch.setenv("PLOT3_CONTOUR_BACKEND", "contourpy")
    _reset_contour_backend()
    with pytest.raises(RuntimeError, match="boom"):
        _contour_lines(xs, ys, field, 0.0)


def test_import_plot3_does_not_import_contourpy():
    script = (
        "import sys, plot3, plot3.contour; "
        "raise SystemExit(0 if 'contourpy' not in sys.modules else 1)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_refine_returns_none_when_the_fine_grid_is_too_large(monkeypatch):
    monkeypatch.setattr("plot3.contour._REFINE_NODE_BUDGET", 1)
    xs, ys, field = _field("circle", n=30)
    assert _refine_active_cells(xs, ys, field, 0.0, lambda a, b: a**2 + b**2 - 1.0) is None
    assert _contour_lines(xs, ys, field, 0.0)


@pytest.mark.parametrize("backend", ["numpy", "contourpy"])
def test_contour_400_is_fast(backend, monkeypatch):
    if backend == "contourpy":
        pytest.importorskip("contourpy")
    monkeypatch.setenv("PLOT3_CONTOUR_BACKEND", backend)
    _reset_contour_backend()
    xs = np.linspace(-2.0, 2.0, 400)
    ys = np.linspace(-2.0, 2.0, 400)
    xx, yy = np.meshgrid(xs, ys)
    field = np.sin(xx) * np.cos(yy)
    _contour_lines(xs, ys, field, 0.0)
    started = time.perf_counter()
    lines = _contour_lines(xs, ys, field, 0.0)
    elapsed = time.perf_counter() - started
    assert lines
    assert elapsed < 0.2


def test_cusp_curve_closes_and_never_jumps_across_the_window():
    from plot3 import geom_function, ggplot
    from plot3.function import expand_function

    def lines(formula):
        layer = expand_function(geom_function(formula), None, None)[0]
        pts = layer.data_override[["x", "y"]].to_numpy()
        return [pts[start : start + count] for start, count in layer._groups]

    heart = lines("(x^2 + y^2 - 1)^3 = x^2 y^3")
    assert len(heart) == 1
    assert np.hypot(*(heart[0][-1] - heart[0][0])) < 1e-6  # closed at the bottom cusp
    for line in lines("y^2 = x^3"):
        steps = np.hypot(*np.diff(line, axis=0).T)
        assert steps.max() < 1.0  # no segment from the top edge to the bottom one


def test_crossings_sit_on_the_curve_at_a_triple_root():
    # At y = 0 the heart reads (x^2 - 1)^3 = 0: linear interpolation of the
    # cube missed the curve by half a cell there, which showed as kinks.
    from plot3 import geom_function
    from plot3.function import expand_function

    layer = expand_function(geom_function("(x^2 + y^2 - 1)^3 = x^2 y^3"), None, None)[0]
    x, y = layer.data_override[["x", "y"]].to_numpy().T
    g = x**2 + y**2 - 1 - np.cbrt(x**2 * y**3)
    near = np.abs(y) < 0.1
    assert near.any() and np.abs(g[near]).max() < 1e-4
