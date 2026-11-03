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


# ── aes(y=columns(...)): the stacking written in the mapping ────────────────

from plot3 import column_names, columns, facet_grid  # noqa: E402
from plot3.build import facet_cells  # noqa: E402


def test_aes_columns_matches_the_layer_form():
    df = _wide()
    cols = ["measurement_a", "measurement_b", "measurement_c"]
    in_aes, _ = build_spec(
        ggplot(df, aes(x=column_names(), y=columns(cols), fill=column_names())) + geom_boxplot()
    )
    in_layer, _ = build_spec(ggplot(df) + geom_boxplot(columns=cols))
    assert in_aes["scales"] == in_layer["scales"]
    assert in_aes["color"]["cats"] == cols
    assert (in_aes["labs"]["x"], in_aes["labs"]["y"]) == ("name", "value")


def test_aes_columns_reach_every_layer():
    fig = (ggplot(_wide(), aes(x=column_names(), y=columns("measurement_*")))
           + geom_boxplot() + geom_point())
    spec, _ = build_spec(fig)
    assert [layer["kind"] for layer in spec["layers"]] == ["box", "point"]
    assert spec["layers"][1]["n"] == 9


def test_aes_columns_takes_names_dicts_and_new_column_names():
    spec, _ = build_spec(
        ggplot(_wide(), aes(x="observation",
                            y=columns("measurement_a", "measurement_c", names_to="m", values_to="mm"),
                            colour=column_names()))
        + geom_line()
    )
    assert spec["color"]["cats"] == ["measurement_a", "measurement_c"]
    assert (spec["labs"]["y"], spec["labs"]["color"]) == ("mm", "m")
    renamed, _ = build_spec(
        ggplot(_wide(), aes(x="observation", y=columns({"measurement_a": "A"}), colour=column_names()))
        + geom_point()
    )
    assert renamed["color"]["cats"] == ["A"]


def test_lines_without_a_colour_are_still_one_per_column():
    fig = ggplot(_wide(), aes(x="observation", y=columns("measurement_*"))) + geom_line()
    spec, _ = build_spec(fig)
    assert len(spec["layers"][0]["groups"]) == 3


def test_facet_wrap_by_the_stacked_names():
    fig = (ggplot(_wide(), aes(x="observation", y=columns("measurement_a", "measurement_b")))
           + geom_line() + facet_wrap("name"))
    cells = facet_cells(fig)["cells"]
    assert [cell["strip"] for cell in cells] == ["measurement_a", "measurement_b"]
    assert [list(cell["fig"].data["value"]) for cell in cells] == [[10.0, 15.0, 8.0], [12.0, 14.0, 10.0]]
    assert "measurement_b" in build_doc(fig)


def test_facet_grid_with_stacked_names_and_another_column():
    fig = (ggplot(_wide(), aes(x="observation", y=columns("measurement_*")))
           + geom_point() + facet_grid(rows="site", cols="name"))
    layout = facet_cells(fig)
    assert (layout["nrow"], layout["ncol"]) == (2, 3)


def test_layer_columns_facet_by_name_splits_the_layer_data():
    fig = ggplot(_wide()) + geom_boxplot(columns="measurement_*") + facet_wrap("name")
    cells = facet_cells(fig)["cells"]
    assert len(cells) == 3
    assert [len(cell["fig"].layers[0].layer_data) for cell in cells] == [3, 3, 3]


def test_aes_columns_errors():
    with pytest.raises(ValueError, match="column_names"):
        build_spec(ggplot(_wide(), aes(x="observation", y="measurement_a", colour=column_names())) + geom_point())
    with pytest.raises(ValueError, match="one set"):
        build_spec(ggplot(_wide(), aes(x=columns("measurement_a"), y=columns("measurement_b"))) + geom_point())
    with pytest.raises(ValueError, match="not both"):
        build_spec(ggplot(_wide()) + geom_boxplot(aes(y=columns("measurement_a")), columns=["measurement_b"]))


def test_notebook_bare_names_inside_columns_become_strings():
    import ast

    from plot3.masking import apply_masking, default_known_names

    src = "aes(x=column_names(), y=columns(measurement_a, measurement_b))"
    out = ast.unparse(ast.parse(apply_masking(src, known=default_known_names())))
    assert out == "aes(x=column_names(), y=columns('measurement_a', 'measurement_b'))"
