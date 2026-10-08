"""Special functions, R-style densities, probability areas, and swept symbols."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plot3 import area, geom_function, ggplot, slider, transition_time
from plot3.build import build_spec
from plot3.expr import ExprError, evaluate, parse_formula
from plot3 import special


def _value(formula: str, x: float, **params) -> float:
    parsed = parse_formula(formula, params)
    return float(evaluate(parsed, {"x": np.array([x])})[0])


@pytest.mark.parametrize(
    "formula, x, expected",
    [
        ("y = gamma(x)", 5.0, 24.0),
        ("y = gamma(x)", 0.5, math.sqrt(math.pi)),
        ("y = lgamma(x)", 10.0, math.log(362880.0)),
        ("y = beta(2, x)", 5.0, 1.0 / 30.0),
        ("y = erf(x)", 1.0, 0.8427007929497149),
        ("y = erfc(x)", 1.0, 0.15729920705028513),
        ("y = dnorm(x)", 0.0, 1.0 / math.sqrt(2.0 * math.pi)),
        ("y = dnorm(x, 100, 15)", 100.0, 1.0 / (15.0 * math.sqrt(2.0 * math.pi))),
        ("y = pnorm(x)", 1.96, 0.9750021048517795),
        ("y = qnorm(x)", 0.975, 1.959963984540054),
        ("y = qnorm(x)", 1e-10, -6.361340902404056),
        ("y = dbeta(x, 2, 5)", 0.2, 2.4576),
        ("y = dbeta(x, 2, 5)", 1.5, 0.0),
        ("y = dbeta(x, 1, 1)", 0.0, 1.0),
        ("y = dt(x, 3)", 0.0, 0.36755259694786135),
        ("y = dchisq(x, 2)", 2.0, math.exp(-1.0) / 2.0),
        ("y = dgamma(x, 2, 3)", 1.0, 9.0 * math.exp(-3.0)),
        ("y = dexp(x, 2)", 1.0, 2.0 * math.exp(-2.0)),
        ("y = dexp(x, 2)", -1.0, 0.0),
        ("y = dunif(x, 0, 4)", 1.0, 0.25),
        ("y = dlnorm(x)", 1.0, 1.0 / math.sqrt(2.0 * math.pi)),
    ],
)
def test_special_function_values(formula, x, expected):
    assert _value(formula, x) == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize(
    "call",
    ["dnorm(x)", "dbeta(x, 2, 5)", "dt(x, 4)", "dgamma(x, 3, 2)", "dchisq(x, 5)", "dexp(x, 1.5)", "dlnorm(x, 0, 0.5)"],
)
def test_densities_integrate_to_one(call):
    xs = np.linspace(-60.0, 60.0, 600_001)
    ys = evaluate(parse_formula(f"y = {call}"), {"x": xs})
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2 has only trapz
    assert float(trapezoid(ys, xs)) == pytest.approx(1.0, abs=2e-3)


def test_beta_function_typesets_as_capital_b():
    formula = parse_formula("y = x^(a-1)(1-x)^(b-1)/beta(a, b)", {"a": 2, "b": 5})
    assert formula.pretty == "y = xᵃ⁻¹(1 − x)ᵇ⁻¹/B(a, b)"
    assert r"\mathrm{B}" in formula.latex
    assert parse_formula("y = gamma(x)").pretty == "y = Γ(x)"


def test_greek_coefficients_named_beta_and_gamma_still_work():
    formula = parse_formula("y = alpha + beta x", {"alpha": 1, "beta": 2})
    assert float(evaluate(formula, {"x": np.array([3.0])})[0]) == 7.0
    assert formula.pretty == "y = α + βx"
    with pytest.raises(ExprError, match="beta=2 at the end"):
        parse_formula("y = beta x")


@pytest.mark.parametrize(
    "formula, kwargs, lo, hi",
    [
        ("y = dbeta(x, 2, 5)", {}, 0.0, 1.0),
        ("y = dnorm(x, 100, 15)", {}, 40.0, 160.0),
        ("y = dnorm(x, mu, s)", {"mu": 5, "s": 2}, -3.0, 13.0),
        ("y = 0.3 dnorm(x) + 0.7 dnorm(x, 4, 0.5)", {}, -4.0, 6.0),
        ("y = dexp(x, 2)", {}, 0.0, 3.0),
    ],
)
def test_density_opens_on_its_support(formula, kwargs, lo, hi):
    spec, _ = build_spec(ggplot() + geom_function(formula, **kwargs))
    assert spec["scales"]["x"]["lo"] == pytest.approx(lo)
    assert spec["scales"]["x"]["hi"] == pytest.approx(hi)


def test_explicit_xlim_beats_the_density_support():
    spec, _ = build_spec(ggplot() + geom_function("y = dnorm(x)", xlim=(-2, 2)))
    assert spec["scales"]["x"]["lo"] == pytest.approx(-2.0)


def _labels(figure) -> list[str]:
    spec, _ = build_spec(figure)
    return [ann["text"] for ann in spec.get("ann") or []]


def test_area_under_a_density_is_a_probability():
    assert _labels(ggplot() + geom_function("y = dbeta(x, 2, 5)") + area(0.2, 0.5)) == [
        "P(0.2 ≤ X ≤ 0.5) = 0.546"
    ]


def test_two_sided_tails_with_infinite_limits():
    figure = (
        ggplot()
        + geom_function("y = dnorm(x)")
        + area(-math.inf, -1.96)
        + area(1.96, math.inf)
    )
    assert _labels(figure) == ["P(X ≤ −1.96) = 0.025", "P(X ≥ 1.96) = 0.025"]


@pytest.mark.parametrize(
    "formula, critical",
    [("y = dt(x, 3)", 2.353), ("y = dchisq(x, 3)", 7.815)],
)
def test_heavy_tails_beyond_the_view_are_counted(formula, critical):
    # Textbook 5% critical values: the tail past the drawn edge matters.
    assert _labels(ggplot() + geom_function(formula) + area(critical, math.inf)) == [
        f"P(X ≥ {critical:.4g}) = 0.05"
    ]


def test_area_under_an_ordinary_curve_stays_an_integral():
    assert _labels(ggplot() + geom_function("y = x^2") + area(0, 2)) == ["∫ = 8/3"]
    approx = _labels(ggplot() + geom_function("y = sin(x)") + area(0, math.inf))
    assert approx[0].startswith("∫ ≈ ")


def test_area_rejects_nan_and_equal_limits():
    with pytest.raises(ValueError):
        area(float("nan"), 1)
    with pytest.raises(ValueError):
        area(1, 1)


@pytest.mark.parametrize(
    "formula, sweep",
    [
        ("y = sin(x - t)", transition_time(t=(0, 6))),
        ("y = sin(x - phase)", transition_time(phase=(0, 6))),
        ("y = dnorm(x, mu, 1)", transition_time(mu=(-3, 3))),
        ("y = dbeta(x, alpha, b)", slider(alpha=(0.5, 5), b=(0.5, 5))),
    ],
)
def test_swept_symbols_are_coefficients_not_axes(formula, sweep):
    # A travelling wave stays a 2D curve instead of becoming a static surface.
    spec, _ = build_spec(ggplot() + geom_function(formula) + sweep)
    assert spec["is3d"] is False


def test_animated_density_covers_every_frame():
    spec, _ = build_spec(
        ggplot() + geom_function("y = dnorm(x, mu, 1)") + transition_time(mu=(-3, 3))
    )
    assert spec["scales"]["x"]["lo"] == pytest.approx(-7.0)
    assert spec["scales"]["x"]["hi"] == pytest.approx(7.0)


def test_support_hint_ignores_non_density_formulas():
    parsed = parse_formula("y = sin(x)")
    assert special.support_hint(parsed.body, "x", parsed.namespace) is None


def test_thin_tail_label_sits_above_the_shading():
    spec, _ = build_spec(ggplot() + geom_function("y = dnorm(x)") + area(1.96, math.inf))
    label = spec["ann"][0]
    tail_top = float(special.dnorm(1.96))
    assert label["y"] > tail_top
