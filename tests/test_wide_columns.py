"""Wide-form input: geom_*(columns=[...]) stacks measurement columns."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from plot3 import (
    aes,
    facet_wrap,
    geom_boxplot,
    geom_col,
    geom_histogram,
    geom_line,
    geom_point,
    geom_violin,
    ggplot,
    labs,
    scale_y_log10,
)
from plot3.build import build_doc, build_spec
from plot3.table import ColumnNotFound
from plot3.wide import melt_wide_layers


def _wide():
    return pd.DataFrame({
        "observation": [1, 2, 3],
        "measurement_a": [10.0, 15.0, 8.0],
        "measurement_b": [12.0, 14.0, 10.0],
        "measurement_c": [9.0, 11.0, 7.0],
        "site": ["n", "s", "n"],
    })


def _long_equivalent(df, columns):
    return pd.concat(
        [df.assign(name=c, value=df[c]) for c in columns], ignore_index=True
    ).assign(name=lambda d: pd.Categorical(d["name"], categories=columns))


def test_boxplot_one_box_per_column_in_the_given_order():
    spec, _ = build_spec(ggplot(_wide()) + geom_boxplot(columns=["measurement_b", "measurement_a"]))
    assert spec["scales"]["x"]["cats"] == ["measurement_b", "measurement_a"]
    assert spec["color"]["cats"] == ["measurement_b", "measurement_a"]
    assert (spec["labs"]["x"], spec["labs"]["y"]) == ("name", "value")


def test_wide_boxplot_matches_the_long_form_plot():
    df = _wide()
    cols = ["measurement_a", "measurement_b", "measurement_c"]
    wide, _ = build_spec(ggplot(df) + geom_boxplot(columns=cols))
    long, _ = build_spec(
        ggplot(_long_equivalent(df, cols), aes(x="name", y="value", fill="name")) + geom_boxplot()
    )
    assert wide["scales"] == long["scales"]
    assert wide["layers"][0]["nOut"] == long["layers"][0]["nOut"]


def test_pattern_dict_labels_and_custom_names():
    spec, _ = build_spec(
        ggplot(_wide(), aes(x="observation"))
        + geom_point(columns={"measurement_a": "A", "measurement_c": "C"}, names_to="measure", values_to="mm")
    )
    assert spec["color"]["cats"] == ["A", "C"]
    assert spec["labs"]["y"] == "mm"
    violin, _ = build_spec(ggplot(_wide()) + geom_violin(columns="measurement_*"))
    assert violin["scales"]["x"]["cats"] == ["measurement_a", "measurement_b", "measurement_c"]


def test_histogram_and_line_and_bars():
    hist, _ = build_spec(ggplot(_wide()) + geom_histogram(columns="measurement_*", bins=3))
    assert (hist["labs"]["x"], hist["labs"]["y"]) == ("value", "count")
    line, _ = build_spec(ggplot(_wide(), aes(x="observation")) + geom_line(columns="measurement_*"))
    assert line["layers"][0]["kind"] == "line"
    assert line["scales"]["y"]["lo"] <= 7.0 and line["scales"]["y"]["hi"] >= 15.0
    stacked, _ = build_spec(ggplot(_wide(), aes(x="site")) + geom_col(columns="measurement_*"))
    dodged, _ = build_spec(ggplot(_wide(), aes(x="site")) + geom_col(columns="measurement_*", position="dodge"))
    # Stacked components reach the total of a site's columns; dodged bars don't.
    assert stacked["scales"]["y"]["hi"] >= 10 + 12 + 9 + 8 + 10 + 7 - 1e-9
    assert dodged["scales"]["y"]["hi"] < 20


def test_polars_data_and_log_scale():
    df = pl.DataFrame({"a": [1.0, 10.0, 100.0], "b": [2.0, 20.0, 200.0]})
    spec, _ = build_spec(ggplot(df) + geom_boxplot(columns=["a", "b"]) + scale_y_log10())
    assert spec["scales"]["y"]["trans"] == "log10"
    assert spec["scales"]["y"]["lo"] == pytest.approx(0.0, abs=0.2)


def test_facets_split_the_stacked_rows():
    html = build_doc(ggplot(_wide(), aes(x="observation")) + geom_line(columns="measurement_*") + facet_wrap("site"))
    assert "measurement_a" in html


def test_labs_win_and_other_layers_keep_wide_data():
    fig = ggplot(_wide(), aes(x="observation", y="measurement_a")) + geom_point() + geom_line(
        columns=["measurement_b"]
    ) + labs(y="reading")
    spec, _ = build_spec(fig)
    assert spec["labs"]["y"] == "reading"
    melted = melt_wide_layers(fig)
    assert melted.layers[0] is fig.layers[0]
    assert len(melted.layers[1].layer_data) == 3


def test_errors():
    with pytest.raises(ColumnNotFound):
        build_spec(ggplot(_wide()) + geom_boxplot(columns=["nope"]))
    with pytest.raises(ValueError, match="not numeric"):
        build_spec(ggplot(_wide()) + geom_boxplot(columns=["site"]))
    with pytest.raises(ValueError, match="matches no column"):
        build_spec(ggplot(_wide()) + geom_boxplot(columns="zzz_*"))
    with pytest.raises(ValueError, match="shared x"):
        build_spec(ggplot(_wide()) + geom_line(columns="measurement_*"))
    with pytest.raises(ValueError, match="remove"):
        build_spec(ggplot(_wide()) + geom_boxplot(aes(y="measurement_a"), columns=["measurement_b"]))
    with pytest.raises(ValueError, match="names_to"):
        build_spec(ggplot(_wide().assign(name=1)) + geom_boxplot(columns=["measurement_a"]))
