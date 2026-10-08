"""scale_* functions, xlim/ylim, palettes, and colour names."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes,
    geom_bar,
    geom_line,
    geom_point,
    ggplot,
    ggsave,
    lims,
    scale_colour_brewer,
    scale_colour_gradient,
    scale_colour_gradient2,
    scale_colour_grey,
    scale_colour_manual,
    scale_colour_viridis_d,
    scale_fill_manual,
    scale_linetype_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_x_date,
    scale_x_discrete,
    scale_x_log10,
    scale_y_continuous,
    scale_y_reverse,
    xlim,
    ylim,
)
from plot3.build import build_spec
from plot3.scaling import extend_palette, format_label, to_hex
from plot3.static import _rgb


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    out = pd.DataFrame({"g": rng.choice(["b", "a", "c"], 60), "x": rng.uniform(0, 10, 60)})
    out["y"] = out.x / 10
    return out


# ── position ─────────────────────────────────────────────────────────────────


def test_breaks_labels_and_name(df):
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y")) + geom_point()
        + scale_x_continuous("Dose", breaks=[2, 4, 6], labels=["two", "four", "six"])
        + scale_y_continuous(labels="percent")
    )
    assert spec["labs"]["x"] == "Dose"
    assert spec["scales"]["x"]["ticks"] == [[2.0, "two"], [4.0, "four"], [6.0, "six"]]
    assert spec["scales"]["x"]["fixed"] is True
    assert all(label.endswith("%") for _pos, label in spec["scales"]["y"]["ticks"])


@pytest.mark.parametrize(
    "value, labels, expected",
    [(0.25, "percent", "25%"), (12345, "comma", "12,345"), (1500, "dollar", "$1,500"),
     (-2.5, "dollar", "-$2.50"), (3.14159, "{:.2f} m", "3.14 m"), (2, lambda v: f"<{v}>", "<2>")],
)
def test_label_formats(value, labels, expected):
    assert format_label(value, labels) == expected


def test_limits_drop_rows_and_say_so(df):
    spec, _ = build_spec(ggplot(df, aes(x="x", y="y")) + geom_point() + xlim(2, 8) + ylim(0, 1))
    assert (spec["scales"]["x"]["lo"], spec["scales"]["x"]["hi"]) == (2.0, 8.0)
    removed = int(((df.x < 2) | (df.x > 8)).sum())
    assert spec["layers"][0]["n"] == len(df) - removed
    assert any(f"Removed {removed} rows outside the scale limits" in n for n in spec["notes"])


def test_open_ended_limits(df):
    spec, _ = build_spec(ggplot(df, aes(x="x", y="y")) + geom_point() + scale_y_continuous(limits=(0, None)))
    assert spec["scales"]["y"]["lo"] == 0.0
    assert spec["scales"]["y"]["hi"] == pytest.approx(df.y.max())


def test_reverse_swaps_the_range(df):
    spec, _ = build_spec(ggplot(df, aes(x="x", y="y")) + geom_point() + scale_y_reverse())
    assert spec["scales"]["y"]["lo"] > spec["scales"]["y"]["hi"]


def test_log10_through_scale_x_continuous():
    frame = pd.DataFrame({"x": [1.0, 10.0, 100.0], "y": [1.0, 2.0, 3.0]})
    spec, _ = build_spec(ggplot(frame, aes(x="x", y="y")) + geom_point() + scale_x_continuous(trans="log10"))
    assert spec["scales"]["x"]["trans"] == "log10"


def test_discrete_limits_order_and_drop(df):
    spec, _ = build_spec(
        ggplot(df, aes(x="g")) + geom_bar()
        + scale_x_discrete(limits=["c", "a"], labels={"c": "Charlie"})
    )
    assert spec["scales"]["x"]["cats"] == ["Charlie", "a"]
    assert xlim("a", "b").kind == "discrete"


def test_date_breaks_and_labels():
    ts = pd.DataFrame({"t": pd.date_range("2021-01-01", periods=24, freq="MS"), "v": range(24)})
    spec, _ = build_spec(
        ggplot(ts, aes(x="t", y="v")) + geom_line()
        + scale_x_date(date_breaks="6 months", date_labels="%b %Y")
    )
    labels = [label for _pos, label in spec["scales"]["x"]["ticks"]]
    assert labels[:3] == ["Jan 2021", "Jul 2021", "Jan 2022"]


def test_lims_sets_both_axes(df):
    fig = ggplot(df, aes(x="x", y="y")) + geom_point() + lims(x=(0, 10), y=(0, 1))
    assert fig.xscale.limits == (0, 10) and fig.yscale.limits == (0, 1)


def test_later_scale_replaces_log(df):
    fig = ggplot(df, aes(x="x", y="y")) + scale_x_log10() + scale_x_continuous(breaks=[1, 2])
    assert fig.scale_x is None


# ── colour ───────────────────────────────────────────────────────────────────


def test_manual_colours_by_name_breaks_and_labels(df):
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y", colour="g")) + geom_point()
        + scale_colour_manual(
            values={"a": "grey50", "b": "steelblue", "c": "#f00"},
            breaks=["c", "b", "a"], labels=["C", "B", "A"], name="Group",
        )
    )
    assert spec["color"]["palette"] == ["#808080", "#4682b4", "#ff0000"]
    assert [(e["label"], e["color"]) for e in spec["legend"]] == [
        ("C", "#ff0000"), ("B", "#4682b4"), ("A", "#808080"),
    ]
    assert spec["labs"]["color"] == "Group"


def test_manual_list_needs_enough_colours(df):
    with pytest.raises(ValueError, match="3 groups"):
        build_spec(ggplot(df, aes(x="x", y="y", colour="g")) + geom_point() + scale_colour_manual(["red"]))


def test_fill_manual_colours_bars(df):
    spec, _ = build_spec(ggplot(df, aes(x="g", fill="g")) + geom_bar() + scale_fill_manual(["red", "green", "blue"]))
    assert spec["color"]["palette"] == ["#ff0000", "#008000", "#0000ff"]


@pytest.mark.parametrize(
    "scale",
    [scale_colour_brewer(palette="Dark2"), scale_colour_viridis_d(), scale_colour_grey()],
)
def test_discrete_palettes(df, scale):
    spec, _ = build_spec(ggplot(df, aes(x="x", y="y", colour="g")) + geom_point() + scale)
    palette = spec["color"]["palette"]
    assert len(palette) == 3 and len(set(palette)) == 3


def test_more_than_eight_groups_get_distinct_colours():
    frame = pd.DataFrame({"x": range(12), "y": range(12), "g": [f"g{i:02d}" for i in range(12)]})
    spec, _ = build_spec(ggplot(frame, aes(x="x", y="y", colour="g")) + geom_point())
    assert len(set(spec["color"]["palette"])) == 12
    assert len(extend_palette(["#000000"], 5)) == 5


def test_gradients(df):
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y", colour="y")) + geom_point()
        + scale_colour_gradient(low="white", high="darkblue")
    )
    assert spec["color"]["ramp"] == ["#ffffff", "#00008b"]  # names as hex for the viewer
    assert spec["color"]["lo"] == pytest.approx(df.y.min())  # full range, as ggplot2
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y", colour="y")) + geom_point()
        + scale_colour_gradient2(midpoint=0.5)
    )
    ramp = spec["color"]["ramp"]
    centre = (0.5 - spec["color"]["lo"]) / (spec["color"]["hi"] - spec["color"]["lo"])
    near_mid = _rgb(ramp[round(centre * 64)])
    assert min(near_mid) >= 0xF0  # white sits on the midpoint


def test_scale_kind_must_match_the_data(df):
    with pytest.raises(ValueError, match="numeric"):
        build_spec(ggplot(df, aes(x="x", y="y", colour="y")) + geom_point() + scale_colour_manual(["red"]))
    with pytest.raises(ValueError, match="categorical"):
        build_spec(ggplot(df, aes(x="x", y="y", colour="g")) + geom_point() + scale_colour_gradient())


# ── shape, linetype, colour names ────────────────────────────────────────────


def test_shape_and_linetype_manual(df):
    spec, _ = build_spec(
        ggplot(df, aes(x="x", y="y", colour="g", shape="g")) + geom_point()
        + scale_shape_manual(["diamond", 4, "square"])
    )
    assert spec["layers"][0]["shape"]["names"] == ["diamond", "cross", "square"]
    assert [e["shape"] for e in spec["legend"]] == ["diamond", "cross", "square"]
    spec, _ = build_spec(
        ggplot(df.sort_values("x"), aes(x="x", y="y", linetype="g")) + geom_line()
        + scale_linetype_manual(["dotted", "solid", "dashed"])
    )
    assert spec["layers"][0]["dashes"] == [[1.0, 3.0], None, [4.0, 4.0]]


@pytest.mark.parametrize(
    "name, rgb",
    [("grey", (128, 128, 128)), ("grey50", (128, 128, 128)), ("gray80", (204, 204, 204)),
     ("SteelBlue", (70, 130, 180)), ("rgb(1, 2, 3)", (1, 2, 3)), ("#abc", (170, 187, 204))],
)
def test_colour_names_in_saved_files(name, rgb):
    assert _rgb(name) == rgb


def test_unknown_colour_name_is_an_error():
    with pytest.raises(ValueError, match="not a CSS name"):
        to_hex("blurple")


def test_named_colours_reach_the_png(tmp_path):
    frame = pd.DataFrame({"x": [1.0], "y": [1.0]})
    path = tmp_path / "grey.svg"
    ggsave(str(path), ggplot(frame, aes(x="x", y="y")) + geom_point(colour="grey50", size=20))
    assert 'fill="#808080"' in path.read_text()
