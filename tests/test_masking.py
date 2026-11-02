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


def test_backticks_inside_strings_and_comments_stay():
    # labs(title="see `hp`") keeps its text; single quotes stay valid Python.
    cell = 'ggplot(df, aes(x=`a b`)) + labs(title="see `hp`")  # `note`'
    out = rewrite_backticks(cell)
    assert out == (
        f"ggplot(df, aes(x={BT_NAME}('a b'))) + labs(title=\"see `hp`\")  # `note`"
    )
    assert rewrite_backticks("labs(title='a `b` c', x=`d`)") == (
        f"labs(title='a `b` c', x={BT_NAME}('d'))"
    )
    assert rewrite_backticks('s = """x `y`\n z"""\naes(x=`q`)') == (
        f's = """x `y`\n z"""\naes(x={BT_NAME}(\'q\'))'
    )
    assert rewrite_backticks(r"labs(title='it\'s `x`')") == r"labs(title='it\'s `x`')"


@pytest.mark.parametrize(
    "cell, expected",
    [
        ("facet_wrap(~cyl)", "facet_wrap('~cyl')"),
        ("facet_wrap( ~ cyl , ncol=2)", "facet_wrap( '~ cyl' , ncol=2)"),
        ("facet_wrap(~`my col`)", "facet_wrap('~my col')"),
        (f"facet_wrap(~{TIDY3_BT_NAME}('my col'))", "facet_wrap('~my col')"),
        ("facet_grid(drv ~ cyl)", "facet_grid('drv ~ cyl')"),
        ('facet_grid(. ~ cyl, scales="free")', "facet_grid('. ~ cyl', scales=\"free\")"),
        ("facet_grid(facets = drv ~ .)", "facet_grid(facets = 'drv ~ .')"),
        ("p + facet_wrap(\n    ~ gear,\n    ncol=2)", "p + facet_wrap(\n    '~ gear',\n    ncol=2)"),
        # Left alone: strings, other calls, ~ inside a nested call.
        ('facet_wrap("~cyl")', 'facet_wrap("~cyl")'),
        ("x = ~mask", "x = ~mask"),
        ("facet_wrap(cyl, labeller=f(~x))", "facet_wrap(cyl, labeller=f(~x))"),
        ("s = 'facet_wrap(~cyl)'", "s = 'facet_wrap(~cyl)'"),
    ],
)
def test_facet_formulas_are_quoted(cell, expected):
    from plot3.masking import rewrite_facet_formulas

    assert rewrite_facet_formulas(cell) == expected


def test_facet_formulas_build():
    df = pd.DataFrame({"a": [1, 2], "cyl": [4, 6], "gear": [3, 4]})
    ns = {"df": df}
    exec("from plot3 import *", ns)
    exec(apply_masking("p = ggplot(df, aes(x=a, y=a)) + geom_point() + facet_wrap(~cyl)"), ns)
    assert ns["p"].facet.variable == "cyl"
    exec(apply_masking("p = ggplot(df, aes(x=a, y=a)) + geom_point() + facet_grid(cyl ~ gear)"), ns)
    assert (ns["p"].facet.rows, ns["p"].facet.cols) == ("cyl", "gear")
    with pytest.raises(ValueError, match="single facet column"):
        exec(apply_masking("facet_wrap(~ cyl + gear)"), ns)


def _shell(*cleanup):
    from types import SimpleNamespace

    return SimpleNamespace(
        user_ns={}, input_transformers_cleanup=list(cleanup),
        input_transformers_post=[], ast_transformers=[],
    )


def _fake_tidy3_transform(lines):
    return lines


_fake_tidy3_transform.__module__ = "tidy3.masking"
_fake_tidy3_transform.__name__ = "tidy3_backtick_transform"


def test_formula_rewrites_run_after_tidy3_preparser():
    # tidy3 imported first: plot3 still rewrites ^ in geom_function and ~ in
    # facets, after tidy3's backticks, and installs no second backtick pass.
    from plot3.jupyter import enable_r_style
    from plot3.masking import plot3_formula_transform

    def other(lines):
        return lines

    ip = _shell(_fake_tidy3_transform, other)
    enable_r_style(ip)
    enable_r_style(ip)  # idempotent
    assert ip.input_transformers_cleanup == [
        _fake_tidy3_transform, plot3_formula_transform, other,
    ]
    out = "".join(plot3_formula_transform(["p = geom_function(y = x^3) + facet_wrap(~cyl)\n"]))
    assert out == "p = geom_function(y = x**3) + facet_wrap('~cyl')\n"

    alone = _shell(other)
    enable_r_style(alone)
    assert alone.input_transformers_cleanup == [plot3_backtick_transform, other]


# ── Valid Python keeps its meaning outside plot3's calls ─────────────────────
# ^ (XOR), ~x (bitwise not) and a "+ name()" line (unary plus) are valid
# Python, so plot3 rewrites them only in its own calls.

USER_GEOM_FUNCTION = "def geom_function(fun):\n    return fun(3)\n"


@pytest.mark.parametrize(
    "cell, expected",
    [
        (USER_GEOM_FUNCTION + "r = geom_function(lambda x: x ^ 2)", None),
        ("from plot3 import *\n" + USER_GEOM_FUNCTION + "r = geom_function(lambda x: x ^ 2)", None),
        ("def f(geom_function):\n    return geom_function(lambda x: x ^ 2)", None),
        ("import numpy as p3\np3.geom_function(lambda x: x ^ 2)", None),
        ("geom_function(lambda x: x ^ 2) + facet_wrap(~cyl)\n!broken", None),  # cannot read ownership
        ("from plot3 import *\ngeom_function(lambda x: 2*x^3)", "from plot3 import *\ngeom_function(lambda x: 2*x**3)"),
        ("import plot3 as p3\np3.geom_function(lambda x: x ^ 2)", "import plot3 as p3\np3.geom_function(lambda x: x ** 2)"),
        ("geom_function(lambda x: np.sin(x^2))", "geom_function(lambda x: np.sin(x**2))"),
        ("def facet_wrap(v):\n    return v\nf = facet_wrap(~5)", None),
        ("def facet_wrap(v):\n    return v\nf = facet_wrap(drv ~ cyl)", "def facet_wrap(v):\n    return v\nf = facet_wrap('drv ~ cyl')"),
    ],
)
def test_reinterpretations_only_in_plot3_calls(cell, expected):
    assert rewrite_backticks(cell, namespace=None) == (cell if expected is None else expected)


def test_session_values_decide_ownership():
    from plot3 import geom_function
    from plot3.masking import rewrite_formulas

    def own(fun):
        return fun(3)

    cell = "p = geom_function(y = x^3) + facet_wrap(~cyl)"
    assert rewrite_formulas(cell, {"geom_function": own, "facet_wrap": own}) == cell
    assert rewrite_formulas(cell, {"geom_function": geom_function, "facet_wrap": facet_wrap}) == (
        "p = geom_function(y = x**3) + facet_wrap('~cyl')"
    )


def test_plus_lines_join_only_onto_plot3():
    from plot3.masking import join_layer_lines

    def geom_point():
        return 1

    assert join_layer_lines("x = 2\n+ geom_point()", namespace=None) is None  # no plot3 head
    assert join_layer_lines("p = ggplot(df)\n+ geom_point()", namespace={"geom_point": geom_point}) is None
    assert join_layer_lines("def geom_point():\n    return 1\np = ggplot(df)\n+ geom_point()", namespace=None) is None
    assert join_layer_lines("p = foo(df)\n+ geom_point()", namespace={"foo": len}) is None
    assert join_layer_lines("p = n\n+ geom_point()", namespace={"n": 2}) is None
    assert join_layer_lines("p = ggplot(df)\n+ geom_point()", namespace=None) == "p = (ggplot(df)\n+ geom_point())"
    assert join_layer_lines("p = base\n+ geom_point()", namespace={"base": ggplot()}) == "p = (base\n+ geom_point())"
    # An indented "+" line is not Python on its own, so it joins by name.
    assert join_layer_lines("x = 2\n  + geom_point()", namespace=None) == "x = (2\n  + geom_point())"


def _ipython():
    IPython = pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    shell.run_line_magic("load_ext", "plot3")
    return shell, IPython


@pytest.mark.parametrize(
    "cell, name, value",
    [
        (USER_GEOM_FUNCTION + "r = geom_function(lambda x: x ^ 2)", "r", 1),
        ("def facet_wrap(v):\n    return v\nf = facet_wrap(~5)", "f", -6),
        ("def geom_point():\n    return 1\nx = 2\n+ geom_point()", "x", 2),
    ],
)
def test_notebook_functions_named_like_plot3_keep_python_meaning(cell, name, value):
    shell, _ = _ipython()
    try:
        result = shell.run_cell(cell)
        assert result.success, result.error_before_exec or result.error_in_exec
        assert shell.user_ns[name] == value
        # Defined in an earlier cell, the session's value decides.
        result = shell.run_cell(cell.rsplit("\n", 1)[-1] if name != "x" else "x = 2\n+ geom_point()")
        assert result.success and shell.user_ns[name] == value
    finally:
        from IPython.core.interactiveshell import InteractiveShell

        InteractiveShell.clear_instance()


def test_plot3_calls_still_take_r_syntax_in_a_notebook():
    shell, _ = _ipython()
    try:
        result = shell.run_cell(
            "import pandas as pd\nfrom plot3 import *\ndf = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})\n"
            "f = geom_function(y = x^3)\np = ggplot(df, aes(x=a, y=b))\n+ geom_point()\n"
        )
        assert result.success, result.error_before_exec or result.error_in_exec
        assert len(shell.user_ns["p"].layers) == 1  # the "+ geom_point()" line joined
    finally:
        from IPython.core.interactiveshell import InteractiveShell

        InteractiveShell.clear_instance()


def test_the_shared_backtick_placeholder_is_a_column_name():
    # tidy3a's notebook pass may rewrite `First Name` in aes() before plot3's runs.
    assert _norm("aes(x=__r_name__('First Name'), y=mpg)") == "aes(x='First Name', y='mpg')"
