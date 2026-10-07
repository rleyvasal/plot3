"""LaTeX and Unicode for a parsed formula.

The walker reads the syntax tree (``^``, implicit ``*``, and grouping already
resolved) and does not simplify. ``2x^3 + 3x^3`` stays two terms.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

# Higher binds tighter. Parentheses follow the same cuts as ast.unparse.
_ADD = 10
_MUL = 20
_UNARY = 30
_POW = 40
_ATOM = 50

_GREEK = {
    "alpha": ("\\alpha", "α"),
    "beta": ("\\beta", "β"),
    "gamma": ("\\gamma", "γ"),
    "delta": ("\\delta", "δ"),
    "epsilon": ("\\epsilon", "ε"),
    "zeta": ("\\zeta", "ζ"),
    "eta": ("\\eta", "η"),
    "theta": ("\\theta", "θ"),
    "iota": ("\\iota", "ι"),
    "kappa": ("\\kappa", "κ"),
    "lambda": ("\\lambda", "λ"),
    "mu": ("\\mu", "μ"),
    "nu": ("\\nu", "ν"),
    "xi": ("\\xi", "ξ"),
    "pi": ("\\pi", "π"),
    "rho": ("\\rho", "ρ"),
    "sigma": ("\\sigma", "σ"),
    "tau": ("\\tau", "τ"),
    "phi": ("\\phi", "φ"),
    "chi": ("\\chi", "χ"),
    "psi": ("\\psi", "ψ"),
    "omega": ("\\omega", "ω"),
}

# Textbook operators. The power sits on the name: sin(x)^2 -> \sin^{2} x.
_OPS = {
    "sin": ("\\sin", "sin"),
    "cos": ("\\cos", "cos"),
    "tan": ("\\tan", "tan"),
    "asin": ("\\arcsin", "arcsin"),
    "acos": ("\\arccos", "arccos"),
    "atan": ("\\arctan", "arctan"),
    "arcsin": ("\\arcsin", "arcsin"),
    "arccos": ("\\arccos", "arccos"),
    "arctan": ("\\arctan", "arctan"),
    "sinh": ("\\sinh", "sinh"),
    "cosh": ("\\cosh", "cosh"),
    "tanh": ("\\tanh", "tanh"),
    "log": ("\\ln", "ln"),
    "ln": ("\\ln", "ln"),
    "log10": ("\\log_{10}", "log₁₀"),
}

_SUP = str.maketrans(
    {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        "+": "⁺", "-": "⁻", "−": "⁻", "(": "⁽", ")": "⁾",
        "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ",
        "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ",
        "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ",
        "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ",
        "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
    }
)
_SUB = str.maketrans(
    {
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "+": "₊", "-": "₋", "−": "₋", "(": "₍", ")": "₎",
        "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
        "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
        "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
        "v": "ᵥ", "x": "ₓ",
    }
)


@dataclass(frozen=True)
class _Piece:
    latex: str
    pretty: str
    prec: int
    kind: str


def formula_texts(
    lhs: ast.AST | None,
    rhs: ast.AST,
    *,
    mode: str,
    dependent: str | None,
    parameters: dict[str, float] | None = None,
) -> dict[str, str]:
    """Symbolic and value-substituted LaTeX and Unicode for one formula."""
    params = parameters or {}
    latex, pretty = _equation(lhs, rhs, mode, dependent, None)
    if params:
        legend_latex, legend_pretty = _equation(lhs, rhs, mode, dependent, params)
    else:
        legend_latex, legend_pretty = latex, pretty
    caption_latex, caption_pretty = _caption(latex, pretty, lhs, rhs, params)
    return {
        "latex": latex,
        "pretty": pretty,
        "legend_latex": legend_latex,
        "legend_pretty": legend_pretty,
        "caption_latex": caption_latex,
        "caption_pretty": caption_pretty,
    }


def split_math(text: str) -> tuple[str, list[dict[str, str]] | None]:
    """Split a label on ``$...$``. Plain text is the Unicode fallback.

    An unmatched ``$`` stays as text. Segments are ``{"text": ...}`` or
    ``{"text": pretty, "latex": source}``.
    """
    if "$" not in text:
        return text, None
    parts: list[dict[str, str]] = []
    plain: list[str] = []
    index = 0
    while index < len(text):
        start = text.find("$", index)
        if start < 0:
            parts.append({"text": text[index:]})
            plain.append(text[index:])
            break
        end = text.find("$", start + 1)
        if end < 0:
            parts.append({"text": text[index:]})
            plain.append(text[index:])
            break
        if start > index:
            parts.append({"text": text[index:start]})
            plain.append(text[index:start])
        latex = text[start + 1 : end]
        pretty = latex_to_pretty(latex)
        parts.append({"text": pretty, "latex": latex})
        plain.append(pretty)
        index = end + 1
    if not any("latex" in part for part in parts):
        return text, None
    return "".join(plain), parts


def latex_to_pretty(src: str) -> str:
    """A small LaTeX subset for canvas labels when KaTeX cannot draw."""
    out: list[str] = []
    index = 0
    length = len(src)
    while index < length:
        char = src[index]
        if char == "\\":
            name, index = _command_name(src, index)
            text, index = _command_pretty(name, src, index)
            out.append(text)
            continue
        if char == "^":
            body, index = _script_body(src, index + 1)
            out.append(_translate(body, _SUP) or f"^({body})")
            continue
        if char == "_":
            body, index = _script_body(src, index + 1)
            out.append(_translate(body, _SUB) or f"_{body}")
            continue
        if char in "{}":
            index += 1
            continue
        out.append(char)
        index += 1
    return " ".join("".join(out).split())


def _equation(lhs, rhs, mode, dependent, substitute):
    if mode == "implicit" and lhs is not None:
        left = render(lhs, 0, substitute)
    else:
        left = _name(dependent or "y", None)
    right = render(rhs, 0, substitute)
    return f"{left.latex} = {right.latex}", f"{left.pretty} = {right.pretty}"


def _caption(latex, pretty, lhs, rhs, parameters):
    names = _param_names(lhs, rhs, parameters)
    if not names:
        return latex, pretty
    bits_l = []
    bits_p = []
    for name in names:
        shown = _name(name, None)
        value = _number(parameters[name])
        bits_l.append(f"{shown.latex} = {value.latex}")
        bits_p.append(f"{shown.pretty} = {value.pretty}")
    return (
        latex + " \\quad (" + ",\\ ".join(bits_l) + ")",
        pretty + "  (" + ", ".join(bits_p) + ")",
    )


def _param_names(lhs, rhs, parameters) -> list[str]:
    """Parameter names in source order. ``ast.walk`` is breadth-first."""
    if not parameters:
        return []
    found: list[str] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            for arg in node.args:
                walk(arg)
            return
        if isinstance(node, ast.Name):
            if node.id in parameters and node.id not in found:
                found.append(node.id)
            return
        for child in ast.iter_child_nodes(node):
            walk(child)

    for tree in (rhs, lhs):
        if tree is not None:
            walk(tree)
    return found


def render(node: ast.AST, min_prec: int, substitute: dict[str, float] | None) -> _Piece:
    piece = _render(node, substitute)
    if piece.prec < min_prec:
        return _Piece(f"({piece.latex})", f"({piece.pretty})", _ATOM, "group")
    return piece


def _render(node: ast.AST, substitute: dict[str, float] | None) -> _Piece:
    if isinstance(node, ast.Name):
        return _name(node.id, substitute)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _number(float(node.value))
    if isinstance(node, ast.UnaryOp):
        return _unary(node, substitute)
    if isinstance(node, ast.BinOp):
        return _binop(node, substitute)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return _call(node, substitute)
    return _Piece("?", "?", _ATOM, "other")


def _unary(node: ast.UnaryOp, substitute) -> _Piece:
    sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
    if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
        return _number(sign * float(node.operand.value))
    # -x^2 is one power under the minus. LaTeX needs the parentheses that
    # Python's printer drops, because -x^{2} is easy to misread.
    if (
        isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.BinOp)
        and isinstance(node.operand.op, ast.Pow)
    ):
        inner = render(node.operand, 0, substitute)
        return _Piece(f"-({inner.latex})", f"−({inner.pretty})", _UNARY, "other")
    mark_l = "-" if isinstance(node.op, ast.USub) else "+"
    mark_p = "−" if isinstance(node.op, ast.USub) else "+"
    inner = render(node.operand, _UNARY + 1, substitute)
    return _Piece(mark_l + inner.latex, mark_p + inner.pretty, _UNARY, "other")


def _binop(node: ast.BinOp, substitute) -> _Piece:
    if isinstance(node.op, ast.Pow):
        if (
            isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id in _OPS
        ):
            return _op_call(
                node.left.func.id, node.left.args, substitute, power=node.right
            )
        base = render(node.left, _POW + 1, substitute)
        exp = render(node.right, 0, substitute)
        sup = _translate(exp.pretty, _SUP)
        pretty = base.pretty + (sup if sup else f"^({exp.pretty})")
        return _Piece(f"{base.latex}^{{{exp.latex}}}", pretty, _POW, "pow")
    if isinstance(node.op, ast.Div):
        left = render(node.left, 0, substitute)
        right = render(node.right, 0, substitute)
        return _Piece(
            f"\\frac{{{left.latex}}}{{{right.latex}}}",
            f"{_frac(left)}/{_frac(right)}",
            _MUL,
            "frac",
        )
    if isinstance(node.op, (ast.Add, ast.Sub)):
        return _add(node, substitute)
    if isinstance(node.op, ast.Mult):
        return _mul(node.left, node.right, substitute)
    if isinstance(node.op, ast.FloorDiv):
        return _spaced(node, "//", "//", substitute)
    if isinstance(node.op, ast.Mod):
        return _spaced(node, "\\bmod", "%", substitute)
    return _Piece("?", "?", _ATOM, "other")


def _add(node: ast.BinOp, substitute) -> _Piece:
    left = render(node.left, _ADD, substitute)
    right = render(node.right, _ADD + 1, substitute)
    if isinstance(node.op, ast.Add) and right.kind == "num" and right.latex.startswith("-"):
        return _Piece(
            f"{left.latex} - {right.latex[1:]}",
            f"{left.pretty} − {right.pretty[1:]}",
            _ADD,
            "other",
        )
    mark_l = "+" if isinstance(node.op, ast.Add) else "-"
    mark_p = "+" if isinstance(node.op, ast.Add) else "−"
    return _Piece(
        f"{left.latex} {mark_l} {right.latex}",
        f"{left.pretty} {mark_p} {right.pretty}",
        _ADD,
        "other",
    )


def _mul(left_node, right_node, substitute) -> _Piece:
    left = render(left_node, _MUL, substitute)
    right = render(right_node, _MUL + 1, substitute)
    if _juxtapose(left, right):
        pretty = left.pretty + right.pretty
        if left.kind == "call" or right.kind == "call":
            pretty = f"{left.pretty} {right.pretty}"
        kind = right.kind if right.kind in {"name", "pow", "call"} else "other"
        return _Piece(_join_command(left.latex, right.latex), pretty, _MUL, kind)
    if right.kind == "num" and right.latex.startswith("-"):
        right = _Piece(f"({right.latex})", f"({right.pretty})", _ATOM, "group")
    return _Piece(
        f"{left.latex} \\cdot {right.latex}",
        f"{left.pretty} · {right.pretty}",
        _MUL,
        "other",
    )


def _join_command(left: str, right: str) -> str:
    """``\\pi`` beside ``x`` is ``\\pi x``. ``\\pix`` would be a different command."""
    if right and right[0].isalpha() and _command_tail(left):
        return left + " " + right
    return left + right


def _command_tail(text: str) -> bool:
    if len(text) < 2 or not text[-1].isalpha():
        return False
    index = len(text) - 1
    while index >= 0 and text[index].isalpha():
        index -= 1
    return index >= 0 and text[index] == "\\"


def _juxtapose(left: _Piece, right: _Piece) -> bool:
    """Number or name beside a factor needs no dot. Number beside number does."""
    if right.kind == "num":
        return False
    if left.kind == "num":
        return True
    if left.kind in {"name", "pow"} and right.kind in {"name", "call", "group", "frac", "pow"}:
        return True
    if left.kind == "call" and right.kind in {"call", "group", "frac", "pow"}:
        return True
    if left.kind == "group" and right.kind in {"name", "call", "group", "frac", "pow"}:
        return True
    return False


def _spaced(node: ast.BinOp, latex_op: str, pretty_op: str, substitute) -> _Piece:
    left = render(node.left, _MUL, substitute)
    right = render(node.right, _MUL + 1, substitute)
    return _Piece(
        f"{left.latex} {latex_op} {right.latex}",
        f"{left.pretty} {pretty_op} {right.pretty}",
        _MUL,
        "other",
    )


def _call(node: ast.Call, substitute) -> _Piece:
    name = node.func.id
    if name == "sqrt":
        return _sqrt(node.args, substitute)
    if name == "cbrt":
        return _cbrt(node.args, substitute)
    if name == "abs":
        return _abs(node.args, substitute)
    if name == "exp" and len(node.args) == 1:
        return _exp(node.args[0], substitute)
    if name == "floor":
        return _wrap_call(node.args, substitute, "\\lfloor ", " \\rfloor", "⌊", "⌋")
    if name == "ceil":
        return _wrap_call(node.args, substitute, "\\lceil ", " \\rceil", "⌈", "⌉")
    if name in _OPS:
        return _op_call(name, node.args, substitute)
    return _user_call(name, node.args, substitute)


def _exp(arg_node, substitute) -> _Piece:
    arg = render(arg_node, 0, substitute)
    long = (
        " + " in arg.latex
        or " - " in arg.latex
        or "\\cdot" in arg.latex
        or "\\frac" in arg.latex
        or "\\left" in arg.latex
        or len(arg.latex) > 12
    )
    if long:
        return _Piece(
            f"\\exp\\left({arg.latex}\\right)",
            f"exp({arg.pretty})",
            _ATOM,
            "call",
        )
    sup = _translate(arg.pretty, _SUP)
    return _Piece(
        f"e^{{{arg.latex}}}",
        "e" + (sup if sup else f"^({arg.pretty})"),
        _ATOM,
        "pow",
    )


def _sqrt(args, substitute) -> _Piece:
    arg = render(args[0], 0, substitute) if args else _Piece("", "", _ATOM, "name")
    if _bare(arg) and arg.kind in {"name", "num"}:
        pretty = "√" + arg.pretty
    else:
        pretty = f"√({arg.pretty})"
    return _Piece(f"\\sqrt{{{arg.latex}}}", pretty, _ATOM, "call")


def _cbrt(args, substitute) -> _Piece:
    arg = render(args[0], 0, substitute) if args else _Piece("", "", _ATOM, "name")
    if _bare(arg) and arg.kind in {"name", "num"}:
        pretty = "∛" + arg.pretty
    else:
        pretty = f"∛({arg.pretty})"
    return _Piece(f"\\sqrt[3]{{{arg.latex}}}", pretty, _ATOM, "call")


def _abs(args, substitute) -> _Piece:
    arg = render(args[0], 0, substitute) if args else _Piece("", "", _ATOM, "name")
    return _Piece(
        f"\\left\\lvert {arg.latex} \\right\\rvert",
        f"|{arg.pretty}|",
        _ATOM,
        "call",
    )


def _wrap_call(args, substitute, latex_l, latex_r, pretty_l, pretty_r) -> _Piece:
    arg = render(args[0], 0, substitute) if args else _Piece("", "", _ATOM, "name")
    return _Piece(
        f"{latex_l}{arg.latex}{latex_r}",
        f"{pretty_l}{arg.pretty}{pretty_r}",
        _ATOM,
        "call",
    )


def _op_call(name, args, substitute, power=None) -> _Piece:
    latex_op, pretty_op = _OPS[name]
    rendered = [render(arg, 0, substitute) for arg in args]
    if len(rendered) == 1 and _bare(rendered[0]):
        suffix_l, suffix_p = f" {rendered[0].latex}", f" {rendered[0].pretty}"
    else:
        inner_l = ", ".join(piece.latex for piece in rendered)
        inner_p = ", ".join(piece.pretty for piece in rendered)
        suffix_l, suffix_p = f"\\left({inner_l}\\right)", f"({inner_p})"
    if power is not None:
        exp = render(power, 0, substitute)
        sup = _translate(exp.pretty, _SUP) or f"^({exp.pretty})"
        return _Piece(
            f"{latex_op}^{{{exp.latex}}}{suffix_l}",
            f"{pretty_op}{sup}{suffix_p}",
            _ATOM,
            "call",
        )
    return _Piece(latex_op + suffix_l, pretty_op + suffix_p, _ATOM, "call")


def _user_call(name, args, substitute) -> _Piece:
    rendered = [render(arg, 0, substitute) for arg in args]
    inner_l = ", ".join(piece.latex for piece in rendered)
    inner_p = ", ".join(piece.pretty for piece in rendered)
    # Below an atom so damp(x)^2 gains parentheses: the power must cover the call.
    return _Piece(
        f"\\operatorname{{{name}}}({inner_l})",
        f"{name}({inner_p})",
        _POW,
        "call",
    )


def _bare(piece: _Piece) -> bool:
    if piece.kind in {"group", "frac"}:
        return False
    return not any(token in piece.latex for token in (" ", "\\frac", "\\left", "\\cdot"))


def _frac(piece: _Piece) -> str:
    if piece.prec <= _ADD or piece.kind == "frac" or "/" in piece.pretty:
        return f"({piece.pretty})"
    return piece.pretty


def _name(name: str, substitute: dict[str, float] | None) -> _Piece:
    if substitute and name in substitute:
        return _number(substitute[name])
    base, sub = _split_ident(name)
    if base in _GREEK:
        latex_base, pretty_base = _GREEK[base]
    else:
        latex_base, pretty_base = base, base
    if not sub:
        return _Piece(latex_base, pretty_base, _ATOM, "name")
    mapped = _translate(sub, _SUB)
    pretty = pretty_base + (mapped if mapped else f"_{sub}")
    return _Piece(f"{latex_base}_{{{sub}}}", pretty, _ATOM, "name")


def _split_ident(name: str) -> tuple[str, str | None]:
    if "_" in name:
        base, sub = name.split("_", 1)
        if base and sub:
            return base, sub
    index = len(name)
    while index > 0 and name[index - 1].isdigit():
        index -= 1
    if 0 < index < len(name) and name[:index].isalpha():
        return name[:index], name[index:]
    return name, None


def _number(value: float) -> _Piece:
    number = float(value)
    if math.isfinite(number) and number == int(number) and abs(number) < 1e15:
        text = str(int(number))
    elif math.isfinite(number):
        text = f"{number:.12g}"
    else:
        text = str(number)
    pretty = text.replace("-", "−")
    prec = _UNARY if text.startswith("-") else _ATOM
    return _Piece(text, pretty, prec, "num")


def _translate(text: str, table) -> str | None:
    # maketrans keys are code points, not characters.
    if not text or any(ord(char) not in table for char in text):
        return None
    return text.translate(table)


def _command_name(src: str, index: int) -> tuple[str, int]:
    cursor = index + 1
    if cursor < len(src) and not src[cursor].isalpha():
        return src[cursor], cursor + 1
    while cursor < len(src) and src[cursor].isalpha():
        cursor += 1
    return src[index + 1 : cursor], cursor


def _command_pretty(name: str, src: str, index: int) -> tuple[str, int]:
    if name in {"left", "right"}:
        return "", index
    if name in {"lvert", "rvert", "vert", "|"}:
        return "|", index
    if name == "cdot":
        return "·", index
    if name in {"quad", ",,", ","}:
        return " ", index
    if name == "frac":
        num, index = _read_group(src, index)
        den, index = _read_group(src, index)
        return f"({latex_to_pretty(num)})/({latex_to_pretty(den)})", index
    if name == "sqrt":
        if index < len(src) and src[index] == "[":
            end = src.find("]", index)
            index = end + 1 if end >= 0 else index
        body, index = _read_group(src, index)
        inner = latex_to_pretty(body)
        shown = inner if inner.isalnum() else f"({inner})"
        return "√" + shown, index
    if name == "operatorname":
        body, index = _read_group(src, index)
        return body, index
    greek = {key: pretty for key, (_latex, pretty) in _GREEK.items()}
    if name in greek:
        return greek[name], index
    # User LaTeX keeps \log as log. The formula walker maps log() to \ln.
    words = {
        "sin": "sin", "cos": "cos", "tan": "tan",
        "ln": "ln", "log": "log", "exp": "exp",
        "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
        "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    }
    if name in words:
        return words[name], index
    return name, index


def _read_group(src: str, index: int) -> tuple[str, int]:
    while index < len(src) and src[index].isspace():
        index += 1
    if index >= len(src):
        return "", index
    if src[index] != "{":
        return src[index], index + 1
    depth = 0
    for cursor in range(index, len(src)):
        if src[cursor] == "{":
            depth += 1
        elif src[cursor] == "}":
            depth -= 1
            if depth == 0:
                return src[index + 1 : cursor], cursor + 1
    return src[index + 1 :], len(src)


def _script_body(src: str, index: int) -> tuple[str, int]:
    while index < len(src) and src[index].isspace():
        index += 1
    if index < len(src) and src[index] == "{":
        body, index = _read_group(src, index)
        return latex_to_pretty(body), index
    if index < len(src) and src[index] == "\\":
        name, index = _command_name(src, index)
        return _command_pretty(name, src, index)
    if index < len(src):
        return src[index], index + 1
    return "", index
