"""facet_grid, shared facet furniture, and multi-plot figures."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes,
    facet_grid,
    facet_wrap,
    geom_point,
    ggplot,
    ggsave,
    labs,
    plot_annotation,
    plot_layout,
    theme_bw,
)
from plot3.build import build_doc, build_spec, facet_cells
from plot3.compose import Composition
from plot3.masking import apply_masking


@pytest.fixture
def tips():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "day": ["Thu", "Thu", "Fri", "Fri", "Sat", "Sat"] * 4,
        "sex": ["F", "M"] * 12,
        "smoker": ["No", "Yes", "No", "No", "Yes", "No"] * 4,
    })
    df["bill"] = rng.uniform(5, 50, len(df))
    df["tip"] = 0.15 * df.bill
    # Drop one combination so the grid has an empty cell.
    return df[~((df.day == "Sat") & (df.sex == "M"))].reset_index(drop=True)


def test_facet_grid_formula_and_keywords():
    assert (facet_grid("sex ~ day").rows, facet_grid("sex ~ day").cols) == ("sex", "day")
    assert (facet_grid(". ~ day").rows, facet_grid(". ~ day").cols) == (None, "day")
    assert facet_grid(rows="sex").cols is None
    with pytest.raises(ValueError):
        facet_grid()
    with pytest.raises(ValueError):
        facet_grid("sex")


def test_facet_grid_cells_strips_and_empty_combinations(tips):
    layout = facet_cells(ggplot(tips, aes(x="bill", y="tip")) + geom_point() + facet_grid("sex ~ day"))
    assert (layout["nrow"], layout["ncol"]) == (2, 3)
    assert layout["col_strips"] == ["Fri", "Sat", "Thu"]  # ggplot2 order
    assert layout["row_strips"] == ["F", "M"]
    empty = [(c["row"], c["col"]) for c in layout["cells"] if c["fig"] is None]
    assert empty == [(1, 1)]  # Sat x M


def test_facet_panels_share_colour_levels(tips):
    # Thu has only "No" smokers for F; that panel must still map "Yes" second.
    layout = facet_cells(
        ggplot(tips, aes(x="bill", y="tip", colour="smoker")) + geom_point() + facet_grid("sex ~ day")
    )
    specs = [build_spec(c["fig"])[0] for c in layout["cells"] if c["fig"] is not None]
    assert all(spec["color"]["cats"] == ["No", "Yes"] for spec in specs)
    assert all(spec["legendPosition"] == "none" and spec["facetChild"] for spec in specs)


def test_static_facet_grid_has_strips_one_legend_and_shared_titles(tmp_path, tips):
    path = tmp_path / "grid.svg"
    ggsave(
        str(path),
        ggplot(tips, aes(x="bill", y="tip", colour="smoker")) + geom_point()
        + facet_grid("sex ~ day") + theme_bw(),
        width=7, height=4, units="in",
    )
    svg = path.read_text()
    for strip in ("Thu", "Fri", "Sat", "F", "M"):
        assert f">{strip}<" in svg
    assert svg.count(">smoker<") == 1  # one legend for the figure
    assert svg.count(">bill<") == 1 and svg.count(">tip<") == 1


def test_facet_wrap_html_has_strips_and_one_legend(tips):
    doc = build_doc(
        ggplot(tips, aes(x="bill", y="tip", colour="smoker")) + geom_point() + facet_wrap("day")
    )
    assert "class='plab'>Thu<" in doc
    assert doc.count("id='flegend'") == 1


def test_facet_grid_html_has_column_and_row_strips(tips):
    doc = build_doc(ggplot(tips, aes(x="bill", y="tip")) + geom_point() + facet_grid("sex ~ day"))
    assert "class='cstrip'>Sat<" in doc
    assert "class='rstrip'><span>M<" in doc
    assert "class='empty'" in doc


def test_masking_reads_bare_names_in_facet_grid():
    out = apply_masking("facet_grid(rows=sex, cols=day, scales='free')", known=set())
    assert "rows='sex'" in out and "cols='day'" in out and "scales='free'" in out


def test_empty_label_removes_an_axis_title(tips):
    spec, _ = build_spec(ggplot(tips, aes(x="day", y="tip")) + geom_point() + labs(x=""))
    assert spec["labs"]["x"] == ""


# ── composition ──────────────────────────────────────────────────────────────


@pytest.fixture
def plots(tips):
    p1 = ggplot(tips, aes(x="bill", y="tip")) + geom_point()
    p2 = ggplot(tips, aes(x="day", y="tip")) + geom_point()
    p3 = ggplot(tips, aes(x="bill", y="tip")) + geom_point() + labs(tag="Z")
    return p1, p2, p3


def test_operators_build_nested_layouts(plots):
    p1, p2, p3 = plots
    fig = (p1 | p2) / p3
    assert isinstance(fig, Composition) and fig.direction == "col"
    assert fig.items[0].direction == "row" and len(fig.items[0].items) == 2
    assert len((p1 | p2 | p3).items) == 3  # same direction flattens
    with pytest.raises(TypeError):
        p1 | 3


def test_rects_follow_relative_widths(plots):
    p1, p2, _p3 = plots
    fig = (p1 | p2) + plot_layout(widths=[2, 1])
    (a, ax, _ay, aw, _ah), (b, bx, _by, bw, _bh) = fig.rects(0, 0, 310, 100, gap=10)
    assert (a, b) == (p1, p2)
    assert aw == pytest.approx(200) and bw == pytest.approx(100) and bx == pytest.approx(210)
    with pytest.raises(ValueError):
        (p1 | p2) + plot_layout(widths=[1, 2, 3])


@pytest.mark.parametrize(
    "levels, expected",
    [("A", ["A", "B", "Z"]), ("a", ["a", "b", "Z"]), ("1", ["1", "2", "Z"]), ("I", ["I", "II", "Z"])],
)
def test_tags_number_the_plots_but_keep_a_plots_own(plots, levels, expected):
    p1, p2, p3 = plots
    fig = (p1 | p2 | p3) + plot_annotation(tag_levels=levels)
    assert [leaf.labs.get("tag") for leaf in fig.tagged().leaves()] == expected


def test_composition_saves_static_and_html(tmp_path, plots):
    p1, p2, p3 = plots
    fig = (p1 | p2) / p3 + plot_annotation(title="Overview", caption="src", tag_levels="A")
    svg_path = tmp_path / "fig.svg"
    ggsave(str(svg_path), fig, width=7, height=5, units="in")
    svg = svg_path.read_text()
    assert ">Overview<" in svg and ">src<" in svg
    assert ">A<" in svg and ">B<" in svg and ">Z<" in svg
    html_path = tmp_path / "fig.html"
    fig.save(str(html_path))
    doc = html_path.read_text()
    assert doc.count("<iframe") == 3
    assert "Overview" in doc
    assert re.search(r"flex-direction:row", doc) and re.search(r"flex-direction:column", doc)


def _facet_frame():
    import pandas as pd

    return pd.DataFrame({
        "g": ["a"] * 3 + ["b"] * 3,
        "x": [1, 2, 3, 10, 20, 30],
        "y": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
    })


def test_facet_free_y_frees_only_y():
    from plot3 import aes, facet_wrap, geom_point, ggplot
    from plot3.build import facet_cells

    layout = facet_cells(ggplot(_facet_frame(), aes(x="x", y="y")) + geom_point() + facet_wrap("g", scales="free_y"))
    forced = [cell["fig"]._force_scales for cell in layout["cells"]]
    assert all(set(f) == {"x"} and f["x"] == (1.0, 30.0) for f in forced)
    layout = facet_cells(ggplot(_facet_frame(), aes(x="x", y="y")) + geom_point() + facet_wrap("g", scales="free_x"))
    assert all(set(cell["fig"]._force_scales) == {"y"} for cell in layout["cells"])


def test_facet_scales_rejects_unknown_values():
    import pytest

    from plot3 import facet_grid, facet_wrap

    with pytest.raises(ValueError, match="free_y"):
        facet_wrap("g", scales="loose")
    facet_grid("g ~ .", scales="free_x")


def test_facet_legend_position_none_hides_the_figure_legend(tmp_path):
    from plot3 import aes, facet_wrap, geom_point, ggplot, ggsave, theme
    from plot3.build import build_doc

    p = (ggplot(_facet_frame(), aes(x="x", y="y", colour="g")) + geom_point()
         + facet_wrap("g") + theme(legend_position="none"))
    assert "id='flegend'" not in build_doc(p)
    out = tmp_path / "f.svg"
    ggsave(str(out), p)
    assert ">a<" in out.read_text()  # strip label
    shown = tmp_path / "shown.svg"
    ggsave(str(shown), ggplot(_facet_frame(), aes(x="x", y="y", colour="g")) + geom_point() + facet_wrap("g"))
    assert shown.read_text().count(">a<") > out.read_text().count(">a<")  # legend key too


def test_facet_strips_are_grey85_in_ggplot2_themes(tmp_path):
    # ggplot2 theme_grey / theme_bw: strips are grey85, not the grid colour
    # (white in theme_grey, which made them invisible).
    from plot3 import aes, facet_wrap, geom_point, ggplot, ggsave, theme_dark, theme_grey
    from plot3.build import build_doc

    base = ggplot(_facet_frame(), aes(x="x", y="y")) + geom_point() + facet_wrap("g")
    for theme, colour in ((theme_grey(), "#d9d9d9"), (None, "#d9d9d9"), (theme_dark(), "#1c2742")):
        fig = base + theme if theme is not None else base
        out = tmp_path / "f.svg"
        ggsave(str(out), fig)
        assert colour in out.read_text().lower()
    assert "#d9d9d9" in build_doc(base + theme_grey()).lower()
