"""R-style bare-name / backtick masking for aes / facet_wrap."""

from __future__ import annotations

import ast

import pandas as pd
import pytest

from plot3 import aes, facet_wrap, geom_point, ggplot
from plot3.build import build_spec
from plot3.masking import (
    BT_NAME,
    TIDY3_BT_NAME,
    apply_masking,
    default_known_names,
    plot3_backtick_transform,
    rewrite_backticks,
)


def _norm(src: str, known: set[str] | None = None) -> str:
    return ast.unparse(
        ast.parse(apply_masking(src, known=known or default_known_names()))
    )


def test_backtick_rewrite_to_sentinel():
    assert rewrite_backticks("aes(x=`First Name`)") == (
        f"aes(x={BT_NAME}('First Name'))"
    )
    assert rewrite_backticks("aes(y=`Age (%)`)") == f"aes(y={BT_NAME}('Age (%)'))"


def test_backtick_transformer_lines():
    lines = ["aes(x=`Phone-Number!`, y=mpg)\n"]
    out = plot3_backtick_transform(lines)
    joined = "".join(out)
    assert BT_NAME in joined
    assert "`" not in joined


def test_aes_bare_names_become_strings():
    out = _norm("aes(x=wt, y=mpg, colour=cyl)")
    assert 'x="wt"' in out or "x='wt'" in out
    assert 'y="mpg"' in out or "y='mpg'" in out
    assert 'colour="cyl"' in out or "colour='cyl'" in out


def test_aes_positional_bare_names():
    out = _norm("aes(wt, mpg)")
    assert '"wt"' in out or "'wt'" in out
    assert '"mpg"' in out or "'mpg'" in out


def test_aes_backtick_spaced_names():
    out = _norm("aes(x=`First Name`, y=`Age (%)`)")
    assert "First Name" in out
    assert "Age (%)" in out
    assert BT_NAME not in out


def test_aes_accepts_tidy3_bt_sentinel():
    # tidy3 preparser may run first and emit __tidy3_bt__
    src = f"aes(x={TIDY3_BT_NAME}('First Name'), y=mpg)"
    out = apply_masking(src, backticks=False, known=default_known_names())
    assert "First Name" in out
    assert TIDY3_BT_NAME not in out


def test_facet_wrap_bare_name():
    out = _norm("facet_wrap(cyl)")
    assert '"cyl"' in out or "'cyl'" in out


def test_facet_wrap_backtick():
    out = _norm("facet_wrap(`gear type`)")
    assert "gear type" in out


def test_nested_aes_in_geom_point():
    out = _norm("geom_point(aes(colour=cyl))")
    assert 'colour="cyl"' in out or "colour='cyl'" in out


def test_ggplot_aes_chain_source():
    out = _norm("ggplot(df, aes(x=wt, y=mpg)) + geom_point()")
    assert 'x="wt"' in out or "x='wt'" in out
    assert 'y="mpg"' in out or "y='mpg'" in out


def test_known_names_not_rewritten():
    known = default_known_names({"wt", "my_x"})
    out = apply_masking("aes(x=wt, y=mpg)", known=known)
    # wt is known → left as Name; mpg unknown → string
    tree = ast.parse(out)
    call = tree.body[0].value
    assert isinstance(call, ast.Call)
    x_kw = next(k for k in call.keywords if k.arg == "x")
    y_kw = next(k for k in call.keywords if k.arg == "y")
    assert isinstance(x_kw.value, ast.Name) and x_kw.value.id == "wt"
    assert isinstance(y_kw.value, ast.Constant) and y_kw.value.value == "mpg"


def test_labs_not_masked_as_columns():
    # labs is not in selector funcs; bare names there would be invalid for
    # titles anyway — ensure we do not rewrite unrelated calls wrongly.
    out = _norm('labs(title="hi", x="weight")')
    assert "hi" in out
    assert "weight" in out


def test_runtime_aes_strings_still_work():
    a = aes(x="wt", y="mpg", colour="cyl")
    assert dict(a) == {"x": "wt", "y": "mpg", "color": "cyl"}


def test_runtime_aes_coerces_named_object():
    class Col:
        name = "horse power"

    a = aes(x=Col(), y="mpg")
    assert a["x"] == "horse power"
    assert a["y"] == "mpg"


def test_end_to_end_masked_source_builds_figure():
    """Simulate Jupyter masking then build a real figure."""
    df = pd.DataFrame(
        {
            "First Name": ["a", "b", "c"],
            "Age (%)": [1.0, 2.0, 3.0],
            "group": ["x", "y", "x"],
        }
    )
    src = apply_masking(
        "aes(x=`First Name`, y=`Age (%)`, colour=group)",
        known=default_known_names(),
    )
    mapping = eval(src, {"aes": aes})
    assert dict(mapping) == {
        "x": "First Name",
        "y": "Age (%)",
        "color": "group",
    }
    fig = ggplot(df, mapping) + geom_point()
    spec, _ = build_spec(fig)
    assert spec["layers"][0]["n"] == 3


def test_facet_wrap_masked_end_to_end(cars):
    src = apply_masking("facet_wrap(cyl, ncol=2)", known=default_known_names())
    fac = eval(src, {"facet_wrap": facet_wrap})
    assert fac.variable == "cyl"
    assert fac.ncol == 2
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point() + fac
    html = fig._repr_html_()
    assert "iframe" in html or "three" in html.lower() or len(html) > 100
