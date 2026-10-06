"""Formula parser for geom_function."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plot3.expr import ExprError, evaluate, parse_formula


def test_explicit_line_and_power():
    formula = parse_formula("y = 2x + 2")
    assert formula.mode == "explicit"
    assert formula.dependent == "y"
    assert formula.variables == ("x",)
    xs = np.array([0.0, 1.0, 3.0])
    ys = evaluate(formula, {"x": xs})
    assert np.allclose(ys, [2.0, 4.0, 8.0])

    quad = parse_formula("y = x^2 + 1")
    assert np.allclose(evaluate(quad, {"x": xs}), xs**2 + 1)

    bare = parse_formula("x^2 + 1")
    assert bare.dependent == "y"
    assert bare.label.startswith("y =")
    assert np.allclose(evaluate(bare, {"x": xs}), xs**2 + 1)


def test_implicit_multiplication_and_parentheses():
    formula = parse_formula("y = 3(x - 1)^2")
    xs = np.array([1.0, 2.0])
    assert np.allclose(evaluate(formula, {"x": xs}), [0.0, 3.0])
    product = parse_formula("z = sin(x) cos(y)")
    assert product.mode == "explicit"
    assert product.variables == ("x", "y")
    assert product.dependent == "z"


def test_parameters_and_constants():
    formula = parse_formula("y = a x^2 + b x + c", {"a": 2, "b": -3, "c": 1})
    xs = np.array([0.0, 1.0])
    assert np.allclose(evaluate(formula, {"x": xs}), [1.0, 0.0])
    wave = parse_formula("y = A sin(k x + phi)", {"A": 3, "k": 2, "phi": math.pi / 2})
    assert np.allclose(evaluate(wave, {"x": np.array([0.0])}), [3.0])
    overridden = parse_formula("y = e x", {"e": 0.5})
    assert np.allclose(evaluate(overridden, {"x": np.array([4.0])}), [2.0])


def test_user_callable_and_vectorize():
    def damp(t):
        return math.exp(-t / 5)

    formula = parse_formula("y = damp(x) sin(3x)", {"damp": damp})
    xs = np.linspace(0.0, 1.0, 5)
    with pytest.warns(UserWarning, match="vectorize"):
        got = evaluate(formula, {"x": xs})
    assert np.allclose(got, np.exp(-xs / 5) * np.sin(3 * xs))


def test_own_names_and_sideways():
    speed = parse_formula("v = 9.8 t")
    assert speed.dependent == "v"
    assert speed.variables == ("t",)
    side = parse_formula("x = y^2")
    assert side.mode == "explicit"
    assert side.dependent == "x"
    assert side.variables == ("y",)


def test_implicit_classification():
    both = parse_formula("y = 2x + 2y")
    assert both.mode == "implicit"
    assert both.variables == ("x", "y")
    circle = parse_formula("x^2 + y^2 = 1")
    assert circle.mode == "implicit"
    hyperbola = parse_formula("x y = 1")
    assert hyperbola.mode == "implicit"


def test_constant_and_callable():
    flat = parse_formula("y = 4")
    assert flat.variables == ()
    assert float(evaluate(flat, {})) == 4.0
    fn = parse_formula(lambda x: x + 1)
    assert fn.mode == "callable"
    assert fn.fn_args == ("x",)
    surface = parse_formula(lambda x, y: x + y)
    assert surface.fn_args == ("x", "y")


def test_error_messages():
    with pytest.raises(ExprError, match=r"'a' has no value. Pass it at the end"):
        parse_formula("y = a x^2")
    with pytest.raises(ExprError, match=r"did you mean x\*y"):
        parse_formula("y = 2xy")
    with pytest.raises(ExprError, match=r"unknown function 'foo'"):
        parse_formula("y = foo(x)")
    with pytest.raises(ExprError, match=r"too many free variables \(x, t, s\)"):
        parse_formula("y = x + t + s")
    with pytest.raises(ExprError, match=r"syntax error at '\+' \(column (\d+)\)") as exc:
        parse_formula("y = 2x +")
    column = int(exc.value.args[0].rsplit("column ", 1)[1].rstrip(")"))
    assert "y = 2x +"[column - 1] == "+"


def test_coefficient_times_parentheses():
    formula = parse_formula("y = a(x + 1)", {"a": 3})
    assert np.allclose(evaluate(formula, {"x": np.array([1.0])}), [6.0])


def test_notebook_value_is_only_a_hint(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("IPython")
    fake.get_ipython = lambda: types.SimpleNamespace(user_ns={"a": 2.5})
    monkeypatch.setitem(sys.modules, "IPython", fake)
    with pytest.raises(ExprError, match=r"Your notebook has a = 2.5"):
        parse_formula("y = a*x")


def test_rejects_attribute_access():
    with pytest.raises(ExprError, match="not allowed"):
        parse_formula("y = x.real")
