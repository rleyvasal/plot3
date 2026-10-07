"""LaTeX as a second way to write a formula, not a replacement."""

from __future__ import annotations

import ast

import numpy as np
import pytest

from plot3 import geom_function, ggplot
from plot3.build import build_spec
from plot3.expr import ExprError, _evaluated_tree, evaluate, parse_formula
from tests.test_mathtext import _CASES

_ALIAS = {
    "ln": "log",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
}


def _canon_name(name: str) -> str:
    """``x0`` and ``x_0`` are one display name. ``ln`` and ``log`` are one log."""
    if "_" not in name:
        index = len(name)
        while index > 0 and name[index - 1].isdigit():
            index -= 1
        if 0 < index < len(name) and name[:index].isalpha():
            name = f"{name[:index]}_{name[index:]}"
    return _ALIAS.get(name, name)


def _canon(tree: ast.AST) -> str:
    """Same maths, including the aliases the LaTeX writer collapses."""

    class Fix(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            node.id = _canon_name(node.id)
            return node

        def visit_BinOp(self, node: ast.BinOp):
            self.generic_visit(node)
            # e^{x} is exp(x). A power of the constant e is the same tree.
            if (
                isinstance(node.op, ast.Pow)
                and isinstance(node.left, ast.Name)
                and node.left.id == "e"
            ):
                return ast.Call(
                    func=ast.Name(id="exp", ctx=ast.Load()),
                    args=[node.right],
                    keywords=[],
                )
            return node

        def visit_Call(self, node: ast.Call):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name):
                node.func.id = _canon_name(node.func.id)
            return node

    return ast.dump(Fix().visit(tree), include_attributes=False)


_MORE = [
    "y = sin(x+1)",
    "y = sin(2x)",
    "y = 2*sin(x)^2",
    "y = exp(-x)",
    "y = sqrt(x+1)",
    "y = abs(x+1)",
    "y = floor(x)",
    "y = ceil(x+1)",
    "y = x % 2",
    "y = x // 2",
    "y = -2x",
    "y = 2*-3",
    "y = sinh(x)",
    "y = asin(x)",
    "y = arcsin(x)",
    "y = ln(x)",
    "y = log(x)",
    "y = x0",
    "y = a_1",
    "y = x^2*y",
    "y = 2*e",
    "y = e*x",
    "y = e^x",
    "y = exp(x)",
    "y = (x+1)*x",
    "y = x*(x+1)",
    "y = pi*x",
    "y = theta*x",
    "y = sin(x)*cos(x)",
    "y = sin(x)*y",
    "z = x*y",
    "y = sin(x*cos(x))",
    "y = sin(cos(x))",
    "y = 2*sin(x)*cos(x)",
    "y = sin(x)**2*cos(x)",
    "y = cbrt(x)",
    "y = cbrt(x+1)",
    "y = abs(abs(x))",
    "y = log10(x+1)",
    "y = 3(x-1)^2",
    "v = 9.8 t",
    "y = 2x^3 + 3x^3",
]


def _sources():
    seen = []
    for source, _latex, _pretty in _CASES:
        if source not in seen:
            seen.append(source)
    for source in _MORE:
        if source not in seen:
            seen.append(source)
    return seen


@pytest.mark.parametrize("source", _sources())
def test_latex_round_trip_keeps_the_tree(source):
    first = _evaluated_tree(source)
    latex = parse_formula(source).latex
    second = _evaluated_tree(f"${latex}$")
    assert _canon(first) == _canon(second)


def test_pasted_fraction_and_user_latex_is_what_the_legend_shows():
    formula = parse_formula(r"\frac{\sin x}{x}")
    assert formula.latex == r"\frac{\sin x}{x}"
    assert formula.legend_latex == r"\frac{\sin x}{x}"
    assert formula.pretty.startswith("y =")
    xs = np.array([0.5, 1.0])
    assert np.allclose(evaluate(formula, {"x": xs}), np.sin(xs) / xs)

    spec, _payloads = build_spec(ggplot() + geom_function(r"\frac{1}{x^2}"))
    assert spec["labsMath"]["title"][0]["latex"] == r"\frac{1}{x^2}"
    assert spec["legend"] is None
    both, _payloads = build_spec(
        ggplot()
        + geom_function(r"\frac{1}{x}", xlim=(0.2, 1), n=4)
        + geom_function("y = x", xlim=(0, 1), n=4)
    )
    assert both["legend"][0]["latex"] == r"\frac{1}{x}"
    assert both["legend"][1]["latex"] == r"y = x"


def test_sin_power_conventions_and_inverse():
    xs = np.array([0.4])
    power_on_arg = parse_formula(r"\sin x^2")
    power_on_sin = parse_formula(r"\sin^2 x")
    assert np.allclose(evaluate(power_on_arg, {"x": xs}), np.sin(xs**2))
    assert np.allclose(evaluate(power_on_sin, {"x": xs}), np.sin(xs) ** 2)
    inverse = parse_formula(r"\sin^{-1}(x)")
    assert np.allclose(evaluate(inverse, {"x": np.array([0.5])}), np.arcsin(0.5))


def test_latex_letters_are_separate_variables_plain_text_is_not():
    product = parse_formula("$xy$")
    assert product.variables == ("x", "y")
    assert product.dependent == "z"
    grid = {"x": np.array([2.0]), "y": np.array([3.0])}
    assert np.allclose(evaluate(product, grid), [6.0])
    with pytest.raises(ExprError, match=r"did you mean x\*y"):
        parse_formula("y = xy")

    times = parse_formula(r"$x(x+1)$")
    assert np.allclose(evaluate(times, {"x": np.array([2.0])}), [6.0])

    def f(t):
        return t + 1

    called = parse_formula("$f(x)$", {"f": f})
    assert np.allclose(evaluate(called, {"x": np.array([2.0])}), [3.0])


def test_plain_braces_group_and_subscripts_name():
    assert _canon(_evaluated_tree("x^{2}")) == _canon(_evaluated_tree("x**2"))
    assert _canon(_evaluated_tree("y = x^{2y}")) == _canon(_evaluated_tree("y = x^(2*y)"))
    named = parse_formula("y = x_{0} + 1")
    assert named.variables == ("x_0",)
    assert named.latex == r"y = x_{0} + 1"


def test_cube_root_and_operatorname():
    root = parse_formula(r"\sqrt[3]{8}")
    assert float(evaluate(root, {})) == pytest.approx(2.0)
    other = parse_formula(r"\sqrt[4]{16}")
    assert float(evaluate(other, {})) == pytest.approx(2.0)

    def damp(t):
        return t * 2

    formula = parse_formula(r"\operatorname{damp}(x)", {"damp": damp})
    assert formula.latex == r"\operatorname{damp}(x)"
    assert np.allclose(evaluate(formula, {"x": np.array([3.0])}), [6.0])


def test_eaten_backslashes_ask_for_a_raw_string():
    samples = [
        "\frac{1}{x}",
        "\theta",
        "\beta",
        "\alpha",
        "\nu",
        "\right)",
        "\vec",
        "\times",
        "\tan x",
    ]
    for sample in samples:
        with pytest.raises(ExprError, match=r'Use a raw string: r"\\frac\{1\}\{x\}"'):
            parse_formula(sample)
    kept = parse_formula("""
    y = 2x
    """)
    assert np.allclose(evaluate(kept, {"x": np.array([1.0])}), [2.0])


def test_unsupported_command_names_itself():
    with pytest.raises(ExprError, match=r"\\int isn't supported in geom_function"):
        parse_formula(r"\int_0^1 x")
    with pytest.raises(ExprError, match=r"\\sum isn't supported in geom_function"):
        parse_formula(r"\sum x")
    with pytest.raises(ExprError, match=r"\\lim isn't supported in geom_function"):
        parse_formula(r"\lim_{x \to 0} x")
    with pytest.raises(ExprError, match=r"expected &"):
        parse_formula(r"\begin{cases} x \end{cases}")
    with pytest.raises(ExprError, match=r"\\begin\{matrix\} isn't supported"):
        parse_formula(r"\begin{matrix} x \end{matrix}")


def test_dollars_and_a_supplied_e_keep_their_meaning():
    wrapped = parse_formula(r"$$y = x^2$$")
    assert wrapped.latex == "y = x^2"
    assert _canon(_evaluated_tree(r"$$y = x^2$$")) == _canon(_evaluated_tree("y = x**2"))
    supplied = parse_formula(r"y = e^{x}", {"e": 2})
    assert np.allclose(evaluate(supplied, {"x": np.array([3.0])}), [8.0])


def test_latex_parameter_stays_in_the_caption():
    formula = parse_formula(r"$y = ax^{2}$", {"a": 2})
    assert formula.latex == r"y = ax^{2}"
    assert formula.legend_latex == r"y = ax^{2}"
    assert r"a = 2" in formula.caption_latex
    assert np.allclose(evaluate(formula, {"x": np.array([3.0])}), [18.0])
