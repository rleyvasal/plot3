"""ggtitle/xlab/ylab, guides(), coord_cartesian, theme_void, geom_rug, and
filled boxplots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes, coord_cartesian, coord_flip, geom_boxplot, geom_density, geom_point,
    geom_rug, geom_smooth, ggplot, ggsave, ggtitle, guides, theme_void, xlab, ylab,
)
from plot3.build import build_spec


def _data():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.normal(size=120), "g": rng.choice(list("abc"), 120)})
    d["y"] = 2 * d["x"] + rng.normal(size=120)
    return d


def test_title_helpers_are_labs():
    spec, _ = build_spec(
        ggplot(_data(), aes("x", "y")) + geom_point()
        + ggtitle("Dose", subtitle="trial 2") + xlab("mg") + ylab("")
    )
    assert spec["labs"]["title"] == "Dose"
    assert spec["labs"]["subtitle"] == "trial 2"
    assert spec["labs"]["x"] == "mg"
    assert spec["labs"]["y"] == ""


def test_guides_hide_legends_but_keep_colours():
    d = _data()
    groups, _ = build_spec(ggplot(d, aes("x", "y", colour="g")) + geom_point() + guides(colour="none"))
    assert groups["legend"] is None and groups["color"]["kind"] == "cat"
    numeric, _ = build_spec(ggplot(d, aes("x", "y", colour="y")) + geom_point() + guides(fill="none"))
    assert numeric["color"]["guide"] is False
    sized, _ = build_spec(ggplot(d.assign(s=d.x.abs()), aes("x", "y", size="s")) + geom_point() + guides(size="none"))
    assert sized["sizeLegend"] is None
    with pytest.raises(ValueError, match="guides"):
        guides(alpha="none")


def test_coord_cartesian_zooms_without_dropping_rows():
    d = _data()
    zoomed, _ = build_spec(
        ggplot(d, aes("x", "y")) + geom_point() + geom_smooth(method="lm")
        + coord_cartesian(xlim=(0, 1), ylim=(0, 2))
    )
    full, _ = build_spec(ggplot(d, aes("x", "y")) + geom_point() + geom_smooth(method="lm"))
    assert (zoomed["scales"]["x"]["lo"], zoomed["scales"]["x"]["hi"]) == (0.0, 1.0)
    # Every row is still there, kept as floats outside the 0..1 window.
    assert zoomed["layers"][0]["n"] == full["layers"][0]["n"] == len(d)
    assert zoomed["layers"][0]["x"]["dtype"] == "f32"
    assert not zoomed.get("notes")
    with pytest.raises(ValueError):
        coord_cartesian(xlim=5)


def test_coord_cartesian_expand_false_reaches_the_spec():
    spec, _ = build_spec(ggplot(_data(), aes("x", "y")) + geom_point() + coord_cartesian(expand=False))
    assert spec["coord"]["expand"] is False


def test_rug_ticks_per_side_with_group_colours():
    d = _data()
    spec, _ = build_spec(ggplot(d, aes("x", "y", colour="g")) + geom_point() + geom_rug())
    sides = {rug["side"]: rug for rug in spec["rugs"]}
    assert set(sides) == {"b", "l"}
    assert len(sides["b"]["values"]) == len(d)
    assert set(sides["b"]["colors"]) == set(spec["color"]["palette"][:3])
    # A rug under a density needs only x.
    dens, _ = build_spec(ggplot(d, aes("x")) + geom_density() + geom_rug(sides="b"))
    assert [rug["side"] for rug in dens["rugs"]] == ["b"]
    with pytest.raises(ValueError, match="sides"):
        geom_rug(sides="x")


def test_rug_moves_with_coord_flip():
    d = _data()
    spec, _ = build_spec(ggplot(d, aes("g", "y")) + geom_boxplot() + geom_rug(sides="l") + coord_flip())
    (rug,) = spec["rugs"]
    assert rug["side"] == "b"  # y values now run along the bottom


def test_saved_files_clip_marks_and_draw_rugs(tmp_path):
    d = _data()
    path = tmp_path / "zoom.svg"
    ggsave(str(path), ggplot(d, aes("x", "y")) + geom_point() + geom_rug()
           + coord_cartesian(xlim=(0, 1)), width=400, height=300)
    svg = path.read_text(encoding="utf-8")
    assert "<clipPath" in svg and 'clip-path="url(#plot3clip1)"' in svg


def test_theme_void_draws_no_axes(tmp_path):
    path = tmp_path / "void.svg"
    ggsave(str(path), ggplot(_data(), aes("x", "y")) + geom_point() + theme_void(), width=400, height=300)
    svg = path.read_text(encoding="utf-8")
    assert ">x</text>" not in svg and ">y</text>" not in svg
    assert "<circle" in svg


def test_filled_boxplot_has_dark_outlines(tmp_path):
    d = _data()
    filled, _ = build_spec(ggplot(d, aes("g", "y", fill="g")) + geom_boxplot())
    outlined, _ = build_spec(ggplot(d, aes("g", "y", colour="g")) + geom_boxplot())
    assert filled["layers"][0]["fillMapped"] is True
    assert "fillMapped" not in outlined["layers"][0]
    from plot3 import theme_bw

    path = tmp_path / "box.svg"
    ggsave(str(path), ggplot(d, aes("g", "y", fill="g")) + geom_boxplot() + theme_bw(), width=400, height=300)
    svg = path.read_text(encoding="utf-8")
    assert svg.count('stroke="#222222"') >= 3 * 6  # whiskers, caps, median per box
