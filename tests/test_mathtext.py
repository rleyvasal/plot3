"""LaTeX and Unicode rendered from the parsed formula, not the source text."""

from __future__ import annotations

import pandas as pd
import pytest

from plot3 import aes, geom_function, geom_point, ggplot, labs
from plot3.build import build_spec
from plot3.expr import parse_formula
from plot3.mathtext import split_math


# The conversion table. Parentheses follow precedence; nothing is simplified.
_CASES = [
    ("t = 2x^3 + 3y^3", r"t = 2x^{3} + 3y^{3}", "t = 2x³ + 3y³"),
    ("y = 2*x**3", r"y = 2x^{3}", "y = 2x³"),
    ("y = 2*3", r"y = 2 \cdot 3", "y = 2 · 3"),
    ("y = 1/x", r"y = \frac{1}{x}", "y = 1/x"),
    ("y = (x+1)/(x-1)", r"y = \frac{x + 1}{x - 1}", "y = (x + 1)/(x − 1)"),
    ("z = x^(2y)", r"z = x^{2y}", "z = x²ʸ"),
    ("y = sqrt(x)", r"y = \sqrt{x}", "y = √x"),
    ("y = cbrt(x)", r"y = \sqrt[3]{x}", "y = ∛x"),
    ("y = abs(x)", r"y = \left\lvert x \right\rvert", "y = |x|"),
    ("y = exp(x)", r"y = e^{x}", "y = eˣ"),
    ("y = exp(x + 1)", r"y = \exp\left(x + 1\right)", "y = exp(x + 1)"),
    ("y = log(x)", r"y = \ln x", "y = ln x"),
    ("y = ln(x)", r"y = \ln x", "y = ln x"),
    ("y = log10(x)", r"y = \log_{10} x", "y = log₁₀ x"),
    ("y = sin(x)^2", r"y = \sin^{2} x", "y = sin² x"),
    ("y = theta + phi + pi", r"y = \theta + \phi + \pi", "y = θ + φ + π"),
    ("y = a_1 + x0", r"y = a_{1} + x_{0}", "y = a₁ + x₀"),
    ("x^2 + y^2 = 1", r"x^{2} + y^{2} = 1", "x² + y² = 1"),
    ("x^2 + 1", r"y = x^{2} + 1", "y = x² + 1"),
    ("y = (x+1)^2", r"y = (x + 1)^{2}", "y = (x + 1)²"),
    ("y = -x^2", r"y = -(x^{2})", "y = −(x²)"),
    ("y = 2x^3 + 3x^3", r"y = 2x^{3} + 3x^{3}", "y = 2x³ + 3x³"),
]


@pytest.mark.parametrize("source, latex, pretty", _CASES)
def test_formula_latex_and_pretty(source, latex, pretty):
    formula = parse_formula(source)
    assert formula.latex == latex
    assert formula.pretty == pretty
    assert formula._repr_latex_() == f"${latex}$"


def test_user_function_and_lambda_have_no_tree_to_simplify():
    def damp(t):
        return t

    formula = parse_formula("y = damp(x)", {"damp": damp})
    assert formula.latex == r"y = \operatorname{damp}(x)"
    assert formula.pretty == "y = damp(x)"

    curve = parse_formula(lambda x: x)
    assert curve.latex == "f(x)"
    assert curve.pretty == "f(x)"
    surface = parse_formula(lambda x, y: x)
    assert surface.latex == "f(x, y)"
    assert surface._repr_latex_() == "$f(x, y)$"


def test_legend_substitutes_and_caption_keeps_the_symbol():
    formula = parse_formula("y = a x^2", {"a": 2})
    assert formula.latex == r"y = ax^{2}"
    assert formula.pretty == "y = ax²"
    assert formula.legend_latex == r"y = 2x^{2}"
    assert formula.legend_pretty == "y = 2x²"
    assert formula.caption_latex == r"y = ax^{2} \quad (a = 2)"
    assert formula.caption_pretty == "y = ax²  (a = 2)"

    both = parse_formula("y = a x + b", {"a": 2, "b": -3})
    assert both.legend_latex == "y = 2x - 3"
    assert both.caption_latex == r"y = ax + b \quad (a = 2,\ b = -3)"
    assert both.caption_pretty == "y = ax + b  (a = 2, b = −3)"


def test_dollar_segments_leave_the_surrounding_text_plain():
    plain, parts = split_math(r"Cubic: $t = 2x^3 + 3y^3$")
    assert plain == "Cubic: t = 2x³ + 3y³"
    assert parts[0] == {"text": "Cubic: "}
    assert parts[1]["latex"] == "t = 2x^3 + 3y^3"
    assert parts[1]["text"] == "t = 2x³ + 3y³"

    axis, axis_parts = split_math(r"$\theta$ (rad)")
    assert axis == "θ (rad)"
    assert axis_parts[0]["latex"] == r"\theta"


def test_single_function_formula_is_the_title():
    spec, _ = build_spec(
        ggplot() + geom_function("t = 2x^3 + 3y^3", n=6, xlim=(-1, 1), ylim=(-1, 1))
    )
    assert spec["legend"] is None
    assert spec["labs"]["title"] == "t = 2x³ + 3y³"
    assert spec["labsMath"]["title"][0]["latex"] == r"t = 2x^{3} + 3y^{3}"
    assert spec["labs"]["x"] == "x"
    assert spec["labs"]["y"] == "y"
    assert spec["labs"]["z"] == "t"
    assert "2x" not in spec["labs"]["z"]
    tip = spec["layers"][0]["tip"]
    assert tip["latex"] == r"t = 2x^{3} + 3y^{3}"
    assert tip["pretty"] == "t = 2x³ + 3y³"
    assert spec["math"] is True


def test_user_title_keeps_the_legend():
    spec, _ = build_spec(
        ggplot()
        + geom_function("y = x^2", xlim=(0, 1), n=4)
        + labs(title="Mine")
    )
    assert spec["labs"]["title"] == "Mine"
    assert spec["legend"][0]["latex"] == r"y = x^{2}"
    assert spec["legend"][0]["label"] == "y = x²"
    assert "^{" not in spec["legend"][0]["label"]


def test_several_functions_keep_distinct_legend_entries():
    figure = ggplot()
    for value in (1, 2, 3):
        figure = figure + geom_function("y = a x^2", a=value, xlim=(-1, 1), n=4)
    spec, _ = build_spec(figure)
    assert spec["labs"]["title"] == ""
    assert [entry["latex"] for entry in spec["legend"]] == [
        r"y = 1x^{2}",
        r"y = 2x^{2}",
        r"y = 3x^{2}",
    ]
    assert spec["layers"][0]["tip"]["latex"] == r"y = ax^{2} \quad (a = 1)"


def test_label_and_labs_dollars():
    spec, _ = build_spec(
        ggplot()
        + geom_function(
            "t = 2x^3 + 3y^3",
            label=r"Cubic: $t = 2x^3 + 3y^3$",
            n=6,
            xlim=(-1, 1),
            ylim=(-1, 1),
        )
    )
    assert spec["legend"] is None
    assert spec["labs"]["title"] == "Cubic: t = 2x³ + 3y³"
    assert spec["labsMath"]["title"][1]["latex"] == "t = 2x^3 + 3y^3"
    # The tooltip still shows the parsed formula, not the caption text.
    assert spec["layers"][0]["tip"]["latex"] == r"t = 2x^{3} + 3y^{3}"

    frame = pd.DataFrame({"x": [1.0], "y": [2.0]})
    labelled, _ = build_spec(
        ggplot(frame, aes(x="x", y="y"))
        + geom_point()
        + labs(title=r"Surface of $t = f(x, y)$", x=r"$\theta$ (rad)")
    )
    assert labelled["labs"]["title"] == "Surface of t = f(x, y)"
    assert labelled["labs"]["x"] == "θ (rad)"
    assert labelled["labsMath"]["x"][0]["latex"] == r"\theta"
    assert labelled["math"] is True


def test_katex_is_loaded_only_for_formulas():
    formula = (ggplot() + geom_function("y = x^2", xlim=(0, 1), n=4)).html()
    assert "cdn.jsdelivr.net/npm/katex@0.16.22" in formula
    assert "throwOnError: false" in formula
    assert "trust: false" in formula
    # The visible fallback is Unicode. The braces live in the LaTeX attribute.
    assert "y = x²" in formula or "y = x\\u00b2" in formula

    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    plain = (ggplot(frame, aes(x="x", y="y")) + geom_point()).html()
    assert "cdn.jsdelivr.net/npm/katex" not in plain
    assert plain.count("__KATEX__") == 0


def test_clip_note_uses_the_variable_name_and_a_unicode_minus():
    spec, _ = build_spec(ggplot() + geom_function("y = 1/x", n=40))
    note = spec["notes"][0]
    assert note.startswith("y clipped to [")
    assert "−" in note
    assert "pass ylim= to change" in note
    assert "-" not in note.split("clipped to ", 1)[1].split("]", 1)[0]

    surface, _ = build_spec(
        ggplot() + geom_function("t = exp(x*y)", n=21, xlim=(-3, 3), ylim=(-3, 3))
    )
    assert surface["notes"]
    assert surface["notes"][0].startswith("t clipped to [")
    assert "pass zlim= to change" in surface["notes"][0]
    assert "−" in surface["notes"][0]
