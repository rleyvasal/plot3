"""Phase B: NumPy ArrayTable backend and integer aesthetics."""

from __future__ import annotations

import numpy as np
import pytest

from plot3 import (
    aes,
    coord_3d,
    geom_bar,
    geom_histogram,
    geom_point,
    geom_point3d,
    ggplot,
)
from plot3.build import build_spec
from plot3.geoms import _as_column_name
from plot3.table import (
    ArrayTable,
    as_table,
    count_by,
    detect_backend,
    get_columns,
    materialize_columns,
    n_rows,
    numeric_array,
)


def test_as_column_name_integers():
    assert _as_column_name(0) == "0"
    assert _as_column_name(3) == "3"
    assert _as_column_name(np.int64(2)) == "2"
    assert _as_column_name("x") == "x"
    # bool must not become "0"/"1" via int path
    assert _as_column_name(True) is True or _as_column_name(True) == True


def test_aes_integer_positions():
    a = aes(x=0, y=1, z=2, colour=3)
    assert a == {"x": "0", "y": "1", "z": "2", "color": "3"}


def test_as_table_2d_array():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tab = as_table(pts)
    assert isinstance(tab, ArrayTable)
    assert detect_backend(tab) == "array"
    assert get_columns(tab) == ["0", "1", "2"]
    assert n_rows(tab) == 2
    assert np.allclose(tab.column(0), [1.0, 4.0])
    assert np.allclose(tab.column("1"), [2.0, 5.0])


def test_as_table_1d_expands_to_index_and_values():
    y = np.array([10.0, 20.0, 30.0])
    tab = as_table(y)
    assert get_columns(tab) == ["0", "1"]
    assert n_rows(tab) == 3
    assert np.allclose(tab.column(0), [0.0, 1.0, 2.0])
    assert np.allclose(tab.column(1), [10.0, 20.0, 30.0])


def test_ggplot_keeps_array_backend():
    pts = np.random.default_rng(0).normal(size=(50, 2))
    fig = ggplot(pts, aes(x=0, y=1)) + geom_point()
    assert fig.backend == "array"
    assert isinstance(fig.data, ArrayTable)
    assert n_rows(fig.data) == 50


def test_geom_point_array_2d():
    pts = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.5]], dtype=np.float64)
    fig = ggplot(pts, aes(x=0, y=1)) + geom_point()
    spec, payloads = build_spec(fig)
    assert spec["is3d"] is False
    assert spec["layers"][0]["kind"] == "point"
    assert spec["layers"][0]["n"] == 3
    assert payloads
    assert len(fig.html()) > 500


def test_geom_point3d_array():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(100, 4))  # x,y,z,color
    fig = (
        ggplot(pts, aes(x=0, y=1, z=2, colour=3))
        + geom_point3d(size=0.02)
        + coord_3d()
    )
    assert fig.backend == "array"
    spec, _ = build_spec(fig)
    assert spec["is3d"] is True
    assert spec["layers"][0]["n"] == 100
    assert "z" in spec["scales"]
    payload = fig.to_payload()
    assert payload["kind"] == "figure"
    assert payload["spec"]["layers"][0]["n"] == 100


def test_array_1d_line_like_point():
    y = np.linspace(0, 1, 20)
    fig = ggplot(y, aes(x=0, y=1)) + geom_point()
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["n"] == 20


def test_array_histogram():
    x = np.random.default_rng(2).normal(size=200)
    # 1D → cols 0=index, 1=values; histogram the values
    fig = ggplot(x, aes(x=1)) + geom_histogram()
    assert fig.backend == "array"
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["kind"] == "col"
    assert spec["layers"][0]["n"] >= 1
    # explicit bins still work
    fig2 = ggplot(x, aes(x=1)) + geom_histogram(bins=8)
    assert build_spec(fig2)[0]["layers"][0]["n"] == 8


def test_array_bar_counts():
    # column 0 = category codes 0/1/2
    data = np.array(
        [
            [0, 1.0],
            [0, 2.0],
            [1, 3.0],
            [1, 4.0],
            [1, 5.0],
            [2, 6.0],
        ],
        dtype=np.float64,
    )
    fig = ggplot(data, aes(x=0)) + geom_bar()
    spec, _ = build_spec(fig)
    assert spec["scales"]["x"]["kind"] == "cat"
    assert spec["layers"][0]["n"] == 3


def test_array_count_by_and_numeric():
    data = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 3.0]])
    tab = as_table(data)
    counts = count_by(tab, "0")
    assert list(counts["y"]) == [2, 1]
    arr = numeric_array(tab, "1", dropna=True)
    assert np.allclose(arr, [1.0, 2.0, 3.0])


def test_array_materialize_and_subsample():
    from plot3.table import subsample_rows

    data = np.arange(20, dtype=np.float64).reshape(10, 2)
    tab = as_table(data)
    sub = materialize_columns(tab, ["0", "1"])
    assert list(sub.columns) == ["0", "1"]
    assert len(sub) == 10
    stepped = subsample_rows(tab, 3)
    assert n_rows(stepped) == 4  # rows 0,3,6,9


def test_array_pipe_deferred():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    fig = pts >> (ggplot(aes(x=0, y=1)) + geom_point())
    assert fig.backend == "array"
    assert build_spec(fig)[0]["layers"][0]["n"] == 2


def test_array_bad_column_raises():
    pts = np.zeros((5, 2))
    fig = ggplot(pts, aes(x=0, y=5)) + geom_point()
    with pytest.raises(KeyError, match="column"):
        build_spec(fig)


def test_array_ndim_rejected():
    with pytest.raises(TypeError, match="1D or 2D"):
        as_table(np.zeros((2, 2, 2)))


def test_coord_max_points_array():
    pts = np.random.default_rng(3).normal(size=(1000, 3))
    fig = (
        ggplot(pts, aes(x=0, y=1, z=2))
        + geom_point3d()
        + coord_3d(max_points=100)
    )
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["n"] <= 100
