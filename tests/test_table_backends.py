"""Multi-backend table support: pandas / polars / tidy3."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes,
    geom_bar,
    geom_boxplot,
    geom_density,
    geom_histogram,
    geom_isosurface,
    geom_point,
    geom_surface,
    geom_violin,
    ggplot,
)
from plot3.build import build_spec, expand_stat_geom
from plot3.table import (
    as_table,
    category_labels,
    count_by,
    detect_backend,
    get_columns,
    group_pieces,
    materialize_columns,
    n_rows,
    numeric_array,
)


def _bar_frame():
    return {
        "cat": ["a", "a", "b", "b", "b", "c"],
        "x": [0.1, 0.2, 0.8, 1.1, 1.0, 2.5],
    }


def test_as_table_keeps_pandas():
    df = pd.DataFrame(_bar_frame())
    out = as_table(df)
    assert out is df
    assert detect_backend(out) == "pandas"


def test_as_table_wraps_mapping():
    out = as_table(_bar_frame())
    assert isinstance(out, pd.DataFrame)
    assert detect_backend(out) == "pandas"


def test_ggplot_keeps_pandas_backend():
    df = pd.DataFrame(_bar_frame())
    fig = ggplot(df, aes(x="cat")) + geom_bar()
    assert fig.backend == "pandas"
    assert fig.data is df


def test_geom_bar_pandas_counts():
    df = pd.DataFrame(_bar_frame())
    fig = ggplot(df, aes(x="cat")) + geom_bar()
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["kind"] == "col"
    assert spec["layers"][0]["n"] == 3
    assert len(fig.html()) > 500


@pytest.fixture
def polars_mod():
    return pytest.importorskip("polars")


def test_as_table_keeps_polars(polars_mod):
    pl = polars_mod
    df = pl.DataFrame(_bar_frame())
    out = as_table(df)
    assert out is df
    assert detect_backend(out) == "polars"
    assert get_columns(out) == ["cat", "x"]
    assert n_rows(out) == 6


def test_count_by_polars_matches_pandas(polars_mod):
    pl = polars_mod
    raw = _bar_frame()
    pdf = pd.DataFrame(raw)
    pldf = pl.DataFrame(raw)
    p_counts = count_by(pdf, "cat")
    pl_counts = count_by(pldf, "cat")
    assert list(p_counts["cat"]) == pl_counts.get_column("cat").to_list()
    assert list(p_counts["y"]) == pl_counts.get_column("y").to_list()


def test_ggplot_keeps_polars_backend(polars_mod):
    pl = polars_mod
    df = pl.DataFrame(_bar_frame())
    fig = ggplot(df, aes(x="cat")) + geom_bar()
    assert fig.backend == "polars"
    assert type(fig.data).__name__ == "DataFrame"
    assert not isinstance(fig.data, pd.DataFrame)


def test_geom_bar_polars_pilot(polars_mod):
    pl = polars_mod
    df = pl.DataFrame(_bar_frame())
    fig = ggplot(df, aes(x="cat")) + geom_bar()
    # Input stays polars
    assert fig.backend == "polars"
    # Expand uses backend count (polars result as data_override)
    expanded = expand_stat_geom(fig.layers[0], fig.mapping, fig.data)
    assert expanded.kind == "col"
    override = expanded.data_override
    assert detect_backend(override) == "polars"
    assert get_columns(override) == ["cat", "y"]
    assert n_rows(override) == 3
    # Full build / encode works
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["kind"] == "col"
    assert spec["layers"][0]["n"] == 3
    assert len(fig.html()) > 500


def test_geom_point_polars_materializes_at_render(polars_mod):
    pl = polars_mod
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 4.0]})
    fig = ggplot(df, aes(x="x", y="y")) + geom_point()
    assert fig.backend == "polars"
    assert fig.data is df
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["kind"] == "point"
    assert spec["layers"][0]["n"] == 3


def test_pipe_polars_into_deferred_ggplot(polars_mod):
    pl = polars_mod
    df = pl.DataFrame(_bar_frame())
    template = ggplot(aes(x="cat")) + geom_bar()
    fig = df >> template
    assert fig.backend == "polars"
    assert build_spec(fig)[0]["layers"][0]["n"] == 3


def test_materialize_columns_selects_only(polars_mod):
    pl = polars_mod
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [10.0, 20.0, 30.0], "c": [0, 0, 0]})
    sub = materialize_columns(df, ["a", "b"])
    assert list(sub.columns) == ["a", "b"]
    assert len(sub) == 2  # null row dropped


def test_lazyframe_backend(polars_mod):
    pl = polars_mod
    lf = pl.DataFrame(_bar_frame()).lazy()
    fig = ggplot(lf, aes(x="cat")) + geom_bar()
    assert fig.backend == "polars"
    assert build_spec(fig)[0]["layers"][0]["n"] == 3


def test_tidy_backend_optional():
    """tidy3 TidyFrame resolves to polars lazily (skip if tidy3 missing)."""
    pytest.importorskip("polars")
    try:
        from tidy3 import tidy
    except ImportError:
        pytest.skip("tidy3 not installed")

    raw = _bar_frame()
    tf = tidy(raw)
    assert detect_backend(as_table(tf)) == "tidy"
    fig = ggplot(tf, aes(x="cat")) + geom_bar()
    assert fig.backend == "tidy"
    # count path resolves tidy → polars
    expanded = expand_stat_geom(fig.layers[0], fig.mapping, fig.data)
    assert detect_backend(expanded.data_override) == "polars"
    assert build_spec(fig)[0]["layers"][0]["n"] == 3


def test_numeric_array_and_group_pieces_polars(polars_mod):
    pl = polars_mod
    df = pl.DataFrame(
        {"g": ["a", "a", "b"], "y": [1.0, 2.0, 3.0], "x": [0.1, 0.2, 0.3]}
    )
    arr = numeric_array(df, "y", dropna=True)
    assert arr.tolist() == [1.0, 2.0, 3.0]
    pieces = group_pieces(df, ["g"])
    assert len(pieces) == 2
    assert pieces[0][0] == ("a",)
    assert n_rows(pieces[0][1]) == 2
    assert category_labels(df, "g") == ["a", "b"]


def test_geom_histogram_polars(polars_mod):
    pl = polars_mod
    df = pl.DataFrame({"x": [0.1, 0.2, 0.8, 1.1, 1.0, 2.5]})
    fig = ggplot(df, aes(x="x")) + geom_histogram(bins=5)
    assert fig.backend == "polars"
    assert fig.data is df
    expanded = expand_stat_geom(fig.layers[0], fig.mapping, fig.data)
    assert expanded.kind == "col"
    assert len(expanded.data_override) == 5
    assert expanded.data_override["y"].sum() == 6
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["kind"] == "col"
    assert spec["layers"][0]["n"] == 5


def test_geom_boxplot_polars(polars_mod):
    pl = polars_mod
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "g": ["a"] * 40 + ["b"] * 40,
            "y": list(rng.normal(0, 1, 40))
            + list(rng.normal(2, 1, 39))
            + [20.0],
        }
    )
    fig = ggplot(df, aes(x="g", y="y")) + geom_boxplot()
    assert fig.backend == "polars"
    assert fig.data is df
    spec, _ = build_spec(fig)
    layer = spec["layers"][0]
    assert layer["kind"] == "box"
    assert layer["n"] == 2
    assert layer["nOut"] >= 1


def test_geom_density_polars_grouped(polars_mod):
    pl = polars_mod
    rng = np.random.default_rng(1)
    df = pl.DataFrame(
        {
            "x": np.concatenate(
                [rng.normal(0, 1, 80), rng.normal(3, 0.8, 80)]
            ),
            "g": ["a"] * 80 + ["b"] * 80,
        }
    )
    fig = ggplot(df, aes(x="x", colour="g")) + geom_density(n=64, fill=True)
    assert fig.backend == "polars"
    assert fig.data is df
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["kind"] == "area"
    assert len(spec["layers"][0]["groups"]) == 2


def test_geom_violin_polars(polars_mod):
    pl = polars_mod
    rng = np.random.default_rng(2)
    df = pl.DataFrame(
        {
            "g": ["a"] * 60 + ["b"] * 60,
            "y": np.concatenate(
                [rng.normal(0, 1, 60), rng.normal(1.5, 0.7, 60)]
            ),
        }
    )
    fig = ggplot(df, aes(x="g", y="y")) + geom_violin(n=48)
    assert fig.backend == "polars"
    assert fig.data is df
    spec, _ = build_spec(fig)
    layer = spec["layers"][0]
    assert layer["kind"] == "poly"
    assert len(layer["groups"]) == 2
    assert layer["n"] > 50


def test_geom_surface_polars(polars_mod):
    pl = polars_mod
    xs = np.linspace(-1, 1, 8)
    ys = np.linspace(-1, 1, 6)
    xx, yy = np.meshgrid(xs, ys)
    df = pl.DataFrame(
        {
            "x": xx.ravel().astype(float),
            "y": yy.ravel().astype(float),
            "z": (xx**2 + yy**2).ravel().astype(float),
        }
    )
    fig = ggplot(df, aes(x="x", y="y", z="z")) + geom_surface()
    assert fig.backend == "polars"
    assert fig.data is df
    spec, _ = build_spec(fig)
    assert spec["is3d"] is True
    assert spec["layers"][0]["kind"] == "surface"
    assert spec["layers"][0]["n"] == 8 * 6


def test_geom_isosurface_polars(polars_mod):
    pl = polars_mod
    rng = np.random.default_rng(4)
    n = 200
    df = pl.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )
    fig = ggplot(df, aes(x="x", y="y", z="z")) + geom_isosurface(
        levels=[0.4], n=12
    )
    assert fig.backend == "polars"
    assert fig.data is df
    spec, _ = build_spec(fig)
    assert spec["is3d"] is True
    assert spec["layers"][0]["kind"] == "isosurface"
    assert spec["layers"][0]["n"] > 0


def test_stat_parity_pandas_vs_polars(polars_mod):
    """Expanded stat frames match across backends for the same data."""
    pl = polars_mod
    raw = {
        "cat": ["a", "a", "b", "b", "b", "c"],
        "x": [0.1, 0.2, 0.8, 1.1, 1.0, 2.5],
        "y": [1.0, 2.0, 1.5, 3.0, 2.5, 0.5],
        "g": ["u", "u", "u", "v", "v", "v"],
    }
    pdf = pd.DataFrame(raw)
    pldf = pl.DataFrame(raw)

    # histogram
    h_pd = expand_stat_geom(geom_histogram(bins=4), aes(x="x"), pdf)
    h_pl = expand_stat_geom(geom_histogram(bins=4), aes(x="x"), pldf)
    assert list(h_pd.data_override["y"]) == list(h_pl.data_override["y"])
    assert np.allclose(h_pd.data_override["x"], h_pl.data_override["x"])

    # density (single group)
    d_pd = expand_stat_geom(geom_density(n=32), aes(x="x"), pdf)
    d_pl = expand_stat_geom(geom_density(n=32), aes(x="x"), pldf)
    assert np.allclose(d_pd.data_override["y"], d_pl.data_override["y"])

    # boxplot
    b_pd = expand_stat_geom(geom_boxplot(), aes(x="cat", y="y"), pdf)
    b_pl = expand_stat_geom(geom_boxplot(), aes(x="cat", y="y"), pldf)
    assert list(b_pd.data_override["middle"]) == list(
        b_pl.data_override["middle"]
    )


def test_tidy_stats_optional():
    pytest.importorskip("polars")
    try:
        from tidy3 import tidy
    except ImportError:
        pytest.skip("tidy3 not installed")

    raw = _bar_frame()
    tf = tidy(raw)
    fig = ggplot(tf, aes(x="x")) + geom_histogram(bins=4)
    assert fig.backend == "tidy"
    assert build_spec(fig)[0]["layers"][0]["n"] == 4

    fig2 = ggplot(tf, aes(x="cat", y="x")) + geom_boxplot()
    assert fig2.backend == "tidy"
    assert build_spec(fig2)[0]["layers"][0]["kind"] == "box"
