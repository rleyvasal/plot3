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


def test_geom_function_formula_is_quoted():
    out = _norm("geom_function(y = 2*x + 2)")
    assert "geom_function" in out
    assert "2 * x + 2" in out or "2*x+2" in out
    assert out.count("geom_function") == 1


def test_geom_function_caret_is_power():
    out = _norm("geom_function(y = x^2 + 1)")
    assert "**" in out
    assert "^" not in out


def test_geom_function_leaves_parameters_and_limits():
    out = _norm("geom_function(y = a*x**2 + b, a=2, b=1, xlim=(0, x_max))")
    assert "x**2" in out or "x ** 2" in out
    assert "a=2" in out or "a = 2" in out
    assert "x_max" in out
    assert "xlim" in out


def test_geom_function_lambda_stays_python():
    out = _norm("geom_function(lambda x: x**2)")
    assert "lambda" in out
    assert 'lambda x: x**2' in out or "lambda x: x ** 2" in out


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


def test_aes_expression_over_columns_passes_whole():
    # aes(colour = factor(cyl)) in a notebook: aes() reads "factor(cyl)";
    # quoting each name (factor("cyl")) would break it.
    from plot3.masking import apply_masking, default_known_names

    known = default_known_names({"np", "scale"})
    assert apply_masking("aes(colour=factor(cyl))", known=known) == "aes(colour='factor(cyl)')"
    assert apply_masking("aes(x=log10(pop), y=lifeExp)", known=known) == "aes(x='log10(pop)', y='lifeExp')"
    assert apply_masking("aes(colour=cyl > 4)", known=known) == "aes(colour='cyl > 4')"
    # Notebook variables and modules keep the old behaviour.
    assert apply_masking("aes(x=np.log(x))", known=known) == "aes(x=np.log('x'))"
    assert apply_masking("aes(x=wt)", known=known) == "aes(x='wt')"


def test_plus_lines_continue_a_ggplot():
    # ggplot(...) then "+ geom_point()" lines, as in R: to Python each "+"
    # line is its own statement (unary plus, or an IndentationError).
    import ast

    from plot3.masking import join_layer_lines

    cell = (
        'ggplot(df, aes(x="a", y="b"))\n'
        ' + geom_point(alpha=0.7)\n'
        ' + scale_x_log10()'
    )
    joined = join_layer_lines(cell)
    assert joined == (
        '(ggplot(df, aes(x="a", y="b"))\n'
        ' + geom_point(alpha=0.7)\n'
        ' + scale_x_log10())'
    )
    tree = ast.parse(joined)
    assert len(tree.body) == 1

    # An assignment, with comments and a blank line between layers.
    cell = 'p = ggplot(df)   # base\n# layers\n+ geom_point()  # points\n\n+ theme_bw()\np'
    joined = join_layer_lines(cell)
    assert joined.startswith("p = (ggplot(df)") and "+ theme_bw())" in joined
    assert len(ast.parse(joined).body) == 2

    # patchwork: a composition takes "+ plot_layout(...)" lines too.
    assert join_layer_lines("fig = (a | b)\n+ plot_layout(widths=[3, 2])").startswith("fig = ((a | b)")


def test_plus_lines_leave_other_python_alone():
    from plot3.masking import join_layer_lines

    assert join_layer_lines("x = 1\n+2") is None                      # unary plus
    assert join_layer_lines("y = 1\n+ foo(2)") is None                # not a plot3 name
    assert join_layer_lines("df\n>> ggplot()\n+ geom_point()") is None  # tidy3's pipe
    assert join_layer_lines("for i in r:\n    p = ggplot(df)\n    + geom_point()") is None
    assert join_layer_lines("p = ggplot(df) + geom_point()") is None  # already one line
