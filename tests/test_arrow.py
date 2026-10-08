"""arrow() heads on segments, paths, and annotate("segment")."""

from __future__ import annotations

import pandas as pd
import pytest

from plot3 import aes, annotate, arrow, geom_path, geom_point, geom_segment, ggplot, ggsave
from plot3.build import build_spec


def _segments():
    return pd.DataFrame({
        "x": [1.0, 2.0], "y": [1.0, 3.0],
        "xend": [4.0, 3.0], "yend": [2.0, 6.0],
        "g": ["a", "b"],
    })


def test_arrow_validates_its_arguments():
    with pytest.raises(ValueError, match="ends"):
        arrow(ends="middle")
    with pytest.raises(ValueError, match="type"):
        arrow(type="filled")


def test_arrow_length_is_inches_at_96_px():
    assert arrow().spec() == {"angle": 30.0, "length": 24.0, "ends": "last", "type": "open"}
    assert arrow(length=0.5, angle=20).spec()["length"] == 48.0


def test_no_arrow_means_no_arrow_spec():
    spec, _ = build_spec(ggplot(_segments(), aes("x", "y", xend="xend", yend="yend")) + geom_segment())
    assert spec["arrows"] is None


def test_segment_arrow_points_at_the_end_in_each_group_colour():
    fig = ggplot(_segments(), aes("x", "y", xend="xend", yend="yend", colour="g")) + geom_segment(
        arrow=arrow(type="closed")
    )
    spec, _ = build_spec(fig)
    heads = spec["arrows"]
    assert len(heads) == 2
    tips = sorted((h["x1"], h["y1"]) for h in heads)
    assert tips == [(3.0, 6.0), (4.0, 2.0)]
    assert {h["type"] for h in heads} == {"closed"}
    assert len({h["color"] for h in heads}) == 2


def test_first_and_both_ends():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.0]})
    first, _ = build_spec(ggplot(frame, aes("x", "y")) + geom_path(arrow=arrow(ends="first")))
    assert [(h["x1"], h["y1"], h["x0"], h["y0"]) for h in first["arrows"]] == [(0.0, 0.0, 1.0, 1.0)]
    both, _ = build_spec(ggplot(frame, aes("x", "y")) + geom_path(arrow=arrow(ends="both")))
    assert sorted((h["x1"], h["y1"]) for h in both["arrows"]) == [(0.0, 0.0), (2.0, 0.0)]


def test_annotate_segment_takes_an_arrow():
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    fig = ggplot(frame, aes("x", "y")) + geom_point() + annotate(
        "segment", x=1, y=2, xend=2, yend=1, arrow=arrow()
    )
    spec, _ = build_spec(fig)
    (head,) = spec["arrows"]
    assert (head["x1"], head["y1"]) == (2.0, 1.0)
    assert head["type"] == "open"


def test_saved_svg_draws_the_heads(tmp_path):
    base = ggplot(_segments(), aes("x", "y", xend="xend", yend="yend"))
    plain, open_, closed = (tmp_path / f"{name}.svg" for name in ("plain", "open", "closed"))
    ggsave(str(plain), base + geom_segment(), width=400, height=300)
    ggsave(str(open_), base + geom_segment(arrow=arrow()), width=400, height=300)
    ggsave(str(closed), base + geom_segment(arrow=arrow(type="closed")), width=400, height=300)
    plain_svg, open_svg, closed_svg = (p.read_text(encoding="utf-8") for p in (plain, open_, closed))
    # Two open heads add two strokes each; closed heads add one filled triangle each.
    assert open_svg.count("<line") - plain_svg.count("<line") == 4
    assert closed_svg.count("<polygon") - plain_svg.count("<polygon") == 2
