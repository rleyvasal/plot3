"""fill vs colour, reference lines, text and annotate, shapes and linetypes."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes,
    annotate,
    geom_abline,
    geom_col,
    geom_errorbar,
    geom_hline,
    geom_label,
    geom_line,
    geom_point,
    geom_text,
    geom_vline,
    ggplot,
    ggsave,
    scale_y_log10,
    theme_bw,
    theme_dark,
)
from plot3.build import build_spec


@pytest.fixture
def summ():
    df = pd.DataFrame({
        "day": ["Fri", "Fri", "Sat", "Sat"],
        "sex": ["F", "M", "F", "M"],
        "mean": [14.0, 18.0, 20.0, 24.0],
    })
    df["lo"], df["hi"] = df["mean"] - 1, df["mean"] + 1
    return df


@pytest.fixture
def lines():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": np.tile(np.arange(5.0), 3), "g": np.repeat(["a", "b", "c"], 5)})
    df["y"] = df.x + rng.normal(0, 0.1, 15)
    return df


# ── fill ─────────────────────────────────────────────────────────────────────


def test_fill_colours_bars_but_not_error_bars(summ):
    spec, _ = build_spec(
        ggplot(summ, aes(x="day", y="mean", fill="sex"))
        + geom_col(position="dodge")
        + geom_errorbar(aes(ymin="lo", ymax="hi"), position="dodge")
        + theme_bw()
    )
    bars, bars_err = spec["layers"]
    assert bars["color"]["kind"] == "cat"
    assert "color" not in bars_err
    assert bars_err["constColor"] == spec["theme"]["ink"]
    assert spec["labs"]["color"] == "sex"
    assert [entry["label"] for entry in spec["legend"]] == ["F", "M"]


def test_error_bars_dodge_by_fill_group(summ):
    from plot3.build import expand_stat_geom

    layer = expand_stat_geom(
        geom_errorbar(aes(ymin="lo", ymax="hi"), position="dodge"),
        aes(x="day", fill="sex"), summ,
    )
    stems = sorted(
        round(float(layer.data_override.x.iloc[start + 2]), 6)
        for start, _n in layer._groups
    )
    assert stems == [-0.225, 0.225, 0.775, 1.225]


def test_ink_follows_the_theme(summ):
    spec, _ = build_spec(
        ggplot(summ, aes(x="day", y="mean"))
        + geom_errorbar(aes(ymin="lo", ymax="hi"))
        + theme_dark()
    )
    assert spec["layers"][0]["constColor"] == "#ffffff"


# ── reference lines ──────────────────────────────────────────────────────────


def test_reference_lines_widen_the_scale_and_carry_dashes():
    df = pd.DataFrame({"x": np.arange(5.0), "y": np.arange(5.0) + 5})
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y"))
        + geom_point()
        + geom_hline(yintercept=[0, 20], linetype="dashed")
        + geom_vline(xintercept=2, colour="red")
        + geom_abline(slope=2, intercept=1)
    )
    assert spec["scales"]["y"]["lo"] == 0.0
    assert spec["scales"]["y"]["hi"] == 20.0
    kinds = [(r["kind"], r.get("value"), r["dash"]) for r in spec["refs"]]
    assert kinds[:3] == [("hline", 0.0, [4.0, 4.0]), ("hline", 20.0, [4.0, 4.0]), ("vline", 2.0, None)]
    assert spec["refs"][2]["color"] == "#ff0000"
    assert spec["refs"][3]["slope"] == 2.0


def test_reference_lines_on_categorical_and_log_axes():
    df = pd.DataFrame({"g": ["a", "b", "c"], "y": [1.0, 10.0, 100.0]})
    spec, _ = build_spec(
        ggplot(df, aes(x="g", y="y"))
        + geom_point()
        + geom_vline(xintercept="b")
        + geom_hline(yintercept=10)
        + scale_y_log10()
    )
    values = {r["kind"]: r["value"] for r in spec["refs"]}
    assert values == {"vline": 1.0, "hline": 1.0}


def test_abline_needs_linear_axes():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 10.0]})
    with pytest.raises(ValueError, match="linear"):
        build_spec(ggplot(df, aes(x="x", y="y")) + geom_point() + geom_abline() + scale_y_log10())


def test_reference_line_alone_asks_for_a_data_layer():
    with pytest.raises(ValueError, match="data layer"):
        build_spec(ggplot(pd.DataFrame({"x": [1]}), aes(x="x")) + geom_hline(yintercept=0))


def test_unknown_linetype_is_a_clear_error():
    with pytest.raises(ValueError, match="linetype"):
        geom_hline(yintercept=0, linetype="wiggly")


def test_static_dashed_reference_line_is_split_into_dashes(tmp_path):
    df = pd.DataFrame({"x": np.arange(5.0), "y": np.arange(5.0)})
    path = tmp_path / "refs.svg"
    ggsave(
        str(path),
        ggplot(df, aes(x="x", y="y")) + geom_point() + geom_hline(yintercept=2, linetype="dashed", colour="#ff0000"),
        width=4, height=3, units="in",
    )
    dashes = re.findall(r'<line[^>]*stroke="#ff0000"', path.read_text())
    assert len(dashes) > 10


# ── text and annotate ────────────────────────────────────────────────────────


def test_geom_text_lands_on_categories_by_name():
    df = pd.DataFrame({"g": ["b", "a"], "y": [1.0, 2.0], "name": ["B", "A"]})
    spec, _ = build_spec(
        ggplot(df, aes(x="g", y="y", label="name")) + geom_point() + geom_text(nudge_y=0.5)
    )
    texts = {ann["text"]: (ann["x"], ann["y"]) for ann in spec["ann"]}
    assert texts == {"A": (0.0, 2.5), "B": (1.0, 1.5)}
    assert all(ann["style"] == "text" for ann in spec["ann"])
    assert len(spec["layers"]) == 1  # the text layer became annotations


def test_geom_label_and_colour_groups():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "g": ["p", "q"]})
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y", colour="g", label="g")) + geom_point() + geom_label()
    )
    styles = {ann["text"]: (ann["style"], ann["color"]) for ann in spec["ann"]}
    palette = spec["theme"]["cat"]
    assert styles == {"p": ("label", palette[0]), "q": ("label", palette[1])}


def test_geom_text_needs_a_label():
    df = pd.DataFrame({"x": [1.0], "y": [1.0]})
    with pytest.raises(ValueError, match="label"):
        build_spec(ggplot(df, aes(x="x", y="y")) + geom_text())


def test_annotate_text_rect_segment_point():
    df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y"))
        + geom_point()
        + annotate("rect", xmin=2, xmax=4, ymin=1, ymax=3)
        + annotate("segment", x=1, y=1, xend=5, yend=5)
        + annotate("text", x=[1, 2], y=[8, 9], label=["one", "two"])
        + annotate("point", x=5, y=5, size=9)
    )
    kinds = [layer["kind"] for layer in spec["layers"]]
    assert kinds == ["point", "poly", "line", "point"]
    assert [ann["text"] for ann in spec["ann"]] == ["one", "two"]
    with pytest.raises(ValueError):
        annotate("circle", x=1, y=1)


def test_static_text_is_written_into_the_svg(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "name": ["alpha", "beta"]})
    path = tmp_path / "text.svg"
    ggsave(str(path), ggplot(df, aes(x="x", y="y", label="name")) + geom_point() + geom_text(), width=4, height=3, units="in")
    svg = path.read_text()
    assert ">alpha<" in svg and ">beta<" in svg


# ── shapes and linetypes ─────────────────────────────────────────────────────


def test_shape_and_colour_share_one_legend(lines):
    spec, _ = build_spec(ggplot(lines, aes(x="x", y="y", colour="g", shape="g")) + geom_point())
    assert spec["layers"][0]["shape"]["names"] == ["circle", "triangle", "square"]
    assert [entry["shape"] for entry in spec["legend"]] == ["circle", "triangle", "square"]
    assert spec["shapeLegend"] is None


def test_shape_on_its_own_column_gets_its_own_legend(lines):
    lines["kind"] = np.where(lines.x > 2, "big", "small")
    spec, _ = build_spec(ggplot(lines, aes(x="x", y="y", colour="g", shape="kind")) + geom_point())
    assert spec["shapeLegend"]["label"] == "kind"
    assert [e["shape"] for e in spec["shapeLegend"]["entries"]] == ["circle", "triangle"]


def test_constant_shape_and_r_numbers():
    df = pd.DataFrame({"x": [1.0], "y": [1.0]})
    spec, _ = build_spec(ggplot(df, aes(x="x", y="y")) + geom_point(shape=17))
    assert spec["layers"][0]["shape"] == "triangle"
    with pytest.raises(ValueError):
        geom_point(shape="star")


def test_too_many_shape_levels_is_an_error():
    df = pd.DataFrame({"x": np.arange(7.0), "y": np.arange(7.0), "g": list("abcdefg")})
    with pytest.raises(ValueError, match="shape"):
        build_spec(ggplot(df, aes(x="x", y="y", shape="g")) + geom_point())


def test_mapped_linetype_splits_lines_and_assigns_dashes(lines):
    spec, _ = build_spec(ggplot(lines, aes(x="x", y="y", linetype="g")) + geom_line())
    layer = spec["layers"][0]
    assert layer["groups"] == [[0, 5], [5, 5], [10, 5]]
    assert layer["dashes"] == [None, [4.0, 4.0], [1.0, 3.0]]
    assert spec["linetypeLegend"]["label"] == "g"


def test_constant_linetype_on_a_line(lines):
    spec, _ = build_spec(ggplot(lines, aes(x="x", y="y", colour="g")) + geom_line(linetype="dotdash"))
    assert spec["layers"][0]["dash"] == [1.0, 3.0, 4.0, 3.0]


def test_static_mapped_linetype_draws_dash_segments(tmp_path, lines):
    path = tmp_path / "lt.svg"
    ggsave(str(path), ggplot(lines, aes(x="x", y="y", linetype="g")) + geom_line(), width=4, height=3, units="in")
    svg = path.read_text()
    assert svg.count("<polyline") >= 1  # the solid level
    assert svg.count("<line") > 20  # dashed and dotted levels
