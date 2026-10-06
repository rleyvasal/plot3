"""Parse math formulas for ``geom_function``.

The quoted string is the main form (``"y = 2x + 2"``). A small normaliser
turns math notation into Python, ``ast`` parses it, and a whitelist walk
rejects anything that is not arithmetic or a known function call. Evaluation
uses that checked tree with NumPy functions — not a raw ``eval`` of the
user string.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# Single letters that may be plot variables without a keyword value.
# Other single letters (a, b, k, …) are coefficients and must be passed in.
_PLOT_LETTERS = set("xyztuvwrs")

_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_UNARY = (ast.UAdd, ast.USub)


class ExprError(ValueError):
    """A formula plot3 will not draw, with a message aimed at the call site."""


def _is_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _fmt_num(value: Any) -> str:
    number = float(value)
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.12g}"


def math_namespace() -> dict[str, Any]:
    """NumPy callables and constants a formula may use by name."""
    return {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "ln": np.log,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "floor": np.floor,
        "ceil": np.ceil,
        "pi": float(np.pi),
        "e": float(np.e),
    }


_MATH = math_namespace()
_MATH_FUNCS = {name for name, value in _MATH.items() if callable(value)}
_MATH_CONSTS = {name for name, value in _MATH.items() if not callable(value)}


@dataclass(frozen=True)
class Formula:
    """A checked formula ready to sample.

    ``mode`` is ``explicit`` (``y = f(x)``), ``implicit`` (``F = 0``), or
    ``callable`` (a Python function). ``variables`` are the free plot
    variables in appearance order. ``namespace`` holds constants, numeric
    parameters, and callables — not the sampled arrays.
    """

    label: str
    mode: str
    code: Any
    dependent: str | None
    variables: tuple[str, ...]
    namespace: dict[str, Any]
    fn: Callable[..., Any] | None = None
    fn_args: tuple[str, ...] = ()


def parse_formula(expr: Any, params: dict[str, Any] | None = None) -> Formula:
    """Turn a string, callable, or sympy-like object into a :class:`Formula`."""
    bound = dict(params or {})
    if isinstance(expr, str):
        return _parse_math(expr, bound)
    if callable(expr):
        return _parse_callable(expr)
    if getattr(expr, "free_symbols", None) is not None:
        return _parse_sympyish(expr, bound)
    raise TypeError(
        "geom_function() expects a formula string or a callable, "
        f"got {type(expr).__name__}"
    )


def evaluate(formula: Formula, variables: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate ``formula`` on array values for its free variables."""
    # Invalid samples (log of a negative, divide by zero) become NaN quietly.
    # A warning there pulls in ``__import__``, which the empty builtins hide.
    with np.errstate(all="ignore"):
        if formula.mode == "callable":
            if formula.fn is None:
                raise ExprError("geom_function() callable is missing")
            args = [variables[name] for name in formula.fn_args]
            value = formula.fn(*args)
        else:
            if formula.code is None:
                raise ExprError("geom_function() formula is missing")
            env = {"__builtins__": {}}
            env.update(formula.namespace)
            env.update(variables)
            try:
                value = eval(formula.code, env)  # noqa: S307
            except ExprError:
                raise
            except Exception as exc:
                raise ExprError(f"could not evaluate formula: {exc}") from exc
    return np.asarray(value, dtype=np.float64)


def _parse_callable(fn: Callable[..., Any]) -> Formula:
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        arg_names = ("x",)
    else:
        arg_names = []
        for param in signature.parameters.values():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            if param.default is not inspect.Parameter.empty:
                continue
            arg_names.append(param.name)
        if not arg_names:
            arg_names = ["x"]
    if len(arg_names) > 2:
        listed = ", ".join(arg_names)
        raise ExprError(
            f"too many free variables ({listed}): at most 2. "
            "A function plot takes one argument (a curve) or two (a surface)"
        )
    label = f"f({', '.join(arg_names)})"
    return Formula(
        label=label,
        mode="callable",
        code=None,
        dependent=None,
        variables=tuple(arg_names),
        namespace={},
        fn=_wrap_user_callable(label, fn),
        fn_args=tuple(arg_names),
    )


def _parse_sympyish(expr: Any, params: dict[str, Any]) -> Formula:
    """Duck-type sympy: ``Equality`` via our parser, other exprs via lambdify."""
    lhs = getattr(expr, "lhs", None)
    rhs = getattr(expr, "rhs", None)
    if lhs is not None and rhs is not None and type(expr).__name__ == "Equality":
        return _parse_math(f"({lhs}) = ({rhs})", params)

    try:
        import sympy
    except ImportError:
        sympy = None
    if sympy is not None and isinstance(expr, sympy.Basic):
        symbols = list(expr.free_symbols)
        symbols.sort(
            key=lambda sym: (
                sym.name not in _PLOT_LETTERS,
                sym.name,
            )
        )
        fn = sympy.lambdify(symbols, expr, modules="numpy")
        names = tuple(sym.name for sym in symbols)
        if len(names) > 2:
            listed = ", ".join(names)
            raise ExprError(
                f"too many free variables ({listed}): at most 2. "
                f"Pass parameters as keywords, e.g. {names[-1]}=1"
            )
        return Formula(
            label=str(expr),
            mode="callable",
            code=None,
            dependent=None,
            variables=names,
            namespace={},
            fn=_wrap_user_callable(str(expr), fn),
            fn_args=names,
        )
    return _parse_math(str(expr), params)


def _parse_math(source: str, params: dict[str, Any]) -> Formula:
    text = source.strip()
    if not text:
        raise ExprError(
            'geom_function() needs a formula, for example geom_function("y = 2x")'
        )
    numbers, callables = _split_bindings(params)
    func_names = set(_MATH_FUNCS)
    func_names.update(callables)
    # Names that are values, so ``a(x+1)`` means ``a*(x+1)`` rather than a call.
    value_names = set(numbers) | set(_MATH_CONSTS)
    lhs_src, rhs_src, label = _normalize(text, func_names, value_names)
    rhs_tree = _parse_side(rhs_src, text)
    lhs_tree = _parse_side(lhs_src, text) if lhs_src is not None else None
    _validate(rhs_tree, func_names)
    if lhs_tree is not None:
        _validate(lhs_tree, func_names)

    rhs_values = _value_names(rhs_tree)
    lhs_values = _value_names(lhs_tree) if lhs_tree is not None else []
    # Appearance order, rhs first so "y = x + t" lists x before anything on the left.
    ordered: list[str] = []
    for name in rhs_values + lhs_values:
        if name not in ordered:
            ordered.append(name)

    namespace: dict[str, Any] = {}
    for name, fn in _MATH.items():
        if name in _MATH_FUNCS:
            namespace[name] = fn
    for name, value in numbers.items():
        if name in _MATH_FUNCS:
            raise ExprError(
                f"'{name}' is a math function; pass a callable to replace it"
            )
        namespace[name] = float(value)
    for name, value in _MATH.items():
        if name in _MATH_CONSTS and name not in numbers:
            namespace[name] = value
    for name, fn in callables.items():
        namespace[name] = _wrap_user_callable(name, fn)

    lone_lhs = isinstance(lhs_tree, ast.Name) and lhs_tree.id not in rhs_values
    # The output name on the left (``v = 9.8 t``) is an axis label, not a
    # coefficient that still needs a value.
    skip_names = {lhs_tree.id} if lone_lhs else set()

    free: list[str] = []
    for name in ordered:
        if name in skip_names:
            continue
        if name in numbers or name in _MATH_CONSTS:
            continue
        if name in callables or name in _MATH_FUNCS:
            raise ExprError(f"'{name}' is a function; call it as {name}(...)")
        hint = _juxtaposition_hint(name)
        if hint is not None:
            raise ExprError(f"unknown name {name!r}: did you mean {hint}?")
        if len(name) == 1 and name not in _PLOT_LETTERS:
            raise ExprError(_missing_param_message(name))
        free.append(name)

    if lhs_tree is None:
        mode = "explicit"
        if len(free) == 0:
            dependent = "y"
        elif len(free) == 1:
            dependent = "x" if free[0] == "y" else "y"
        else:
            dependent = "z" if "z" not in free else "f"
        code_tree = rhs_tree
    elif lone_lhs:
        mode = "explicit"
        dependent = lhs_tree.id
        code_tree = rhs_tree
    else:
        mode = "implicit"
        dependent = None
        code_tree = ast.BinOp(left=lhs_tree, op=ast.Sub(), right=rhs_tree)
        # Names only on the left still count (already in ``free``).

    if len(free) > 2:
        listed = ", ".join(free)
        raise ExprError(
            f"too many free variables ({listed}): at most 2. "
            f"Pass parameters as keywords, e.g. {free[-1]}=1"
        )
    if mode == "implicit" and len(free) != 2:
        listed = ", ".join(free) if free else "none"
        raise ExprError(
            f"implicit equation needs 2 free variables, got ({listed})"
        )

    if lhs_tree is None:
        shown = text
        if "=" not in text:
            shown = f"{dependent} = {text}"
        label = shown
    else:
        label = text

    expr_node = ast.fix_missing_locations(ast.Expression(body=code_tree))
    code = compile(expr_node, "<geom_function>", "eval")
    return Formula(
        label=label,
        mode=mode,
        code=code,
        dependent=dependent,
        variables=tuple(free),
        namespace=namespace,
    )


def _split_bindings(
    params: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Callable[..., Any]]]:
    numbers: dict[str, float] = {}
    callables: dict[str, Callable[..., Any]] = {}
    for name, value in params.items():
        if _is_number(value):
            numbers[name] = float(value)
        elif callable(value):
            callables[name] = value
        else:
            raise TypeError(
                f"geom_function() keyword {name}={value!r} must be a number "
                "or a function"
            )
    return numbers, callables


def _missing_param_message(name: str) -> str:
    found = _notebook_number(name)
    if found is not None:
        return (
            f"'{name}' has no value. Your notebook has {name} = {_fmt_num(found)}. "
            f"Use geom_function(..., {name}={name})"
        )
    return (
        f"'{name}' has no value. Pass it at the end: "
        f"geom_function(..., {name}=2)"
    )


def _notebook_number(name: str) -> Any:
    """A numeric value from the notebook namespace, for a better error only."""
    try:
        from IPython import get_ipython
    except Exception:
        return None
    shell = get_ipython()
    if shell is None:
        return None
    user_ns = getattr(shell, "user_ns", None)
    if not isinstance(user_ns, dict) or name not in user_ns:
        return None
    value = user_ns[name]
    if _is_number(value):
        return value
    return None


def _juxtaposition_hint(name: str) -> str | None:
    """``xy`` → ``x*y`` when every letter is itself a plot variable."""
    if len(name) < 2 or not name.isalpha():
        return None
    if all(ch in _PLOT_LETTERS for ch in name):
        return "*".join(name)
    return None


def _value_names(tree: ast.AST | None) -> list[str]:
    """Names used as values, in source order. Call targets are skipped."""
    if tree is None:
        return []
    found: list[str] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            for arg in node.args:
                walk(arg)
            return
        if isinstance(node, ast.Name):
            found.append(node.id)
            return
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(tree)
    return found


def _validate(tree: ast.AST, func_names: set[str]) -> None:
    allowed_ops = _BINOPS + _UNARY
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load)):
            continue
        if isinstance(node, allowed_ops):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINOPS):
                raise ExprError("only + - * / // % and power are allowed")
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARY):
                raise ExprError("only + and - are allowed as signs")
            continue
        if isinstance(node, ast.Name):
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExprError("only numbers are allowed as constants")
            continue
        if isinstance(node, ast.Call):
            if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
                raise ExprError("function keyword arguments are not allowed")
            if not isinstance(node.func, ast.Name):
                raise ExprError("only plain function calls are allowed")
            fname = node.func.id
            if fname not in func_names:
                raise ExprError(
                    f"unknown function {fname!r}: pass it as "
                    f"geom_function(..., {fname}={fname})"
                )
            continue
        raise ExprError(f"not allowed in a formula: {type(node).__name__}")


def _parse_side(source: str, original: str) -> ast.AST:
    try:
        return ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        column = exc.offset or 1
        token = ""
        if exc.text and exc.offset:
            token = exc.text[exc.offset - 1 : exc.offset]
        if not token:
            token = original.strip()[-1:] or "?"
        raise ExprError(
            f"syntax error at {token!r} (column {column})"
        ) from exc


def _normalize(
    source: str, func_names: set[str], value_names: set[str]
) -> tuple[str | None, str, str]:
    """Return ``(lhs, rhs, label)`` with implicit ``*`` and ``^`` → ``**``.

    ``lhs`` is ``None`` when the formula has no ``=``.
    """
    try:
        raw = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenizeError as exc:
        raise ExprError(f"syntax error at {exc}") from exc

    tokens = []
    for tok in raw:
        if tok.type in (
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.COMMENT,
            tokenize.INDENT,
            tokenize.DEDENT,
        ):
            continue
        if tok.type == tokenize.ERRORTOKEN:
            raise ExprError(
                f"syntax error at {tok.string!r} (column {tok.start[1] + 1})"
            )
        if tok.string in {"==", "!=", "<=", ">=", "<", ">"}:
            raise ExprError(
                f"syntax error at {tok.string!r} (column {tok.start[1] + 1})"
            )
        tokens.append(tok)
    if not tokens:
        raise ExprError(
            'geom_function() needs a formula, for example geom_function("y = 2x")'
        )

    last = tokens[-1]
    if last.type == tokenize.OP and last.string not in {")", "}"}:
        raise ExprError(
            f"syntax error at {last.string!r} (column {last.start[1] + 1})"
        )

    depth = 0
    eq_at: int | None = None
    for index, tok in enumerate(tokens):
        if tok.string == "(":
            depth += 1
        elif tok.string == ")":
            depth -= 1
            if depth < 0:
                raise ExprError(
                    f"syntax error at ')' (column {tok.start[1] + 1})"
                )
        elif tok.string == "=" and depth == 0:
            if eq_at is not None:
                raise ExprError(
                    f"syntax error at '=' (column {tok.start[1] + 1})"
                )
            eq_at = index
    if depth != 0:
        raise ExprError("syntax error at '(' (column 1)")

    if eq_at is not None:
        lhs_tokens = tokens[:eq_at]
        rhs_tokens = tokens[eq_at + 1 :]
        if not lhs_tokens or not rhs_tokens:
            bad = tokens[eq_at]
            raise ExprError(
                f"syntax error at '=' (column {bad.start[1] + 1})"
            )
        return (
            _render(lhs_tokens, func_names, value_names),
            _render(rhs_tokens, func_names, value_names),
            source,
        )
    return None, _render(tokens, func_names, value_names), source


def _star_before_paren(name: str, func_names: set[str], value_names: set[str]) -> bool:
    """True when ``name(`` is a product, not a function call.

    Known functions stay calls. An unknown word (``foo(x)``) stays a call so
    the validator can say it is an unknown function. A coefficient or plot
    variable (``a(x+1)``, ``x(y+1)``) is multiplication.
    """
    if name in func_names:
        return False
    if len(name) == 1 or name in value_names:
        return True
    return False


def _render(
    tokens: list[tokenize.TokenInfo],
    func_names: set[str],
    value_names: set[str],
) -> str:
    """Join tokens, inserting ``*`` where math writes juxtaposition."""
    parts: list[str] = []
    prev_end = False
    prev_name: str | None = None
    for tok in tokens:
        is_name = tok.type == tokenize.NAME
        is_num = tok.type == tokenize.NUMBER
        is_lpar = tok.string == "("
        starts_value = is_name or is_num or is_lpar
        if prev_end and starts_value:
            keep_call = (
                prev_name is not None
                and is_lpar
                and not _star_before_paren(prev_name, func_names, value_names)
            )
            if not keep_call:
                parts.append("*")
        parts.append("**" if tok.string == "^" else tok.string)
        prev_end = is_name or is_num or tok.string == ")"
        prev_name = tok.string if is_name else None
    return "".join(parts)


def _wrap_user_callable(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Call ``fn`` on arrays; fall back to ``np.vectorize`` once if it cannot."""
    state: dict[str, Any] = {"warned": False, "vectorized": None}

    def vectorized(*args: Any) -> Any:
        import warnings

        if not state["warned"]:
            warnings.warn(
                f"geom_function: {name}() does not accept arrays; "
                "using np.vectorize (slower)",
                UserWarning,
                stacklevel=4,
            )
            state["warned"] = True
        if state["vectorized"] is None:
            state["vectorized"] = np.vectorize(fn, otypes=[np.float64])
        return state["vectorized"](*args)

    def wrapped(*args: Any) -> Any:
        if state["vectorized"] is not None:
            return state["vectorized"](*args)
        try:
            out = fn(*args)
        except Exception:
            return vectorized(*args)
        if args and np.ndim(args[0]) > 0:
            arr = np.asarray(out)
            if getattr(arr, "shape", None) != np.shape(args[0]):
                return vectorized(*args)
        return out

    return wrapped
