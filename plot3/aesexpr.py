"""Expressions in aes(): ``aes(ymin="mean - se")``, ``aes(colour="factor(cyl)")``.

ggplot2 evaluates ``aes()`` in the data, so R code maps computed values
directly. plot3 reads such a string as an expression over the data's
columns: arithmetic, comparisons, and a short list of functions, evaluated
with NumPy. Nothing else runs, so a string from a file cannot execute code.
The computed values become a column named by the expression, which then
titles the axis or legend, as in ggplot2.
"""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np
import pandas as pd

from plot3.table import ColumnNotFound, detect_backend, get_columns, has_column


def _factor(values):
    from plot3.scales import ordered_levels

    array = np.asarray(values)
    if array.dtype.kind == "f" and np.all(np.isnan(array) | (array == np.round(array))):
        # factor(cyl) reads 4, 6, 8, not 4.0, 6.0, 8.0.
        array = pd.array(np.where(np.isnan(array), 0, array).astype(np.int64), dtype="Int64")
        array[np.isnan(np.asarray(values, dtype=np.float64))] = pd.NA
    series = pd.Series(array)
    levels = ordered_levels([v for v in pd.unique(series) if not pd.isna(v)])
    return pd.Categorical(series, categories=levels)


def _round(values, digits=0):
    return np.round(np.asarray(values, dtype=np.float64), int(digits))


def _ifelse(cond, yes, no):
    return np.where(np.asarray(cond, dtype=bool), yes, no)


def _scalar(fn):
    return lambda values: fn(np.asarray(values, dtype=np.float64))


_FUNCTIONS = {
    "log": np.log, "log10": np.log10, "log2": np.log2, "log1p": np.log1p,
    "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs, "floor": np.floor,
    "ceiling": np.ceil, "ceil": np.ceil, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "round": _round, "ifelse": _ifelse, "pmin": np.fmin, "pmax": np.fmax,
    "factor": _factor, "as_factor": _factor,
    "as_numeric": lambda v: pd.to_numeric(pd.Series(np.asarray(v)), errors="coerce").to_numpy(),
    "as_character": lambda v: np.asarray(v).astype(str),
    "mean": _scalar(np.nanmean), "median": _scalar(np.nanmedian),
    "sd": _scalar(lambda a: np.nanstd(a, ddof=1)), "min": _scalar(np.nanmin),
    "max": _scalar(np.nanmax), "sum": _scalar(np.nansum),
}
_CONSTANTS = {"True": True, "False": False, "pi": np.pi}
_BINARY = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.true_divide,
    ast.Pow: np.power, ast.Mod: np.mod, ast.FloorDiv: np.floor_divide,
    ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or,
}
_COMPARE = {
    ast.Eq: np.equal, ast.NotEq: np.not_equal, ast.Lt: np.less, ast.LtE: np.less_equal,
    ast.Gt: np.greater, ast.GtE: np.greater_equal,
}
_STAT_FORMS = re.compile(r"^\s*(after_stat|stat|after_scale|stage)\s*\(|^\.\.\w+\.\.$")


def _translate(text: str) -> tuple[str, dict[str, str]]:
    """R spellings to Python ones; backtick names to placeholders."""
    names: dict[str, str] = {}

    def tick(match):
        key = f"__col{len(names)}"
        names[key] = match.group(1)
        return key

    out = re.sub(r"`([^`]+)`", tick, text)
    out = out.replace("^", "**")
    out = re.sub(r"\bas\.(factor|numeric|character)\s*\(", r"as_\1(", out)
    out = re.sub(r"\bTRUE\b", "True", out)
    out = re.sub(r"\bFALSE\b", "False", out)
    out = re.sub(r"!(?!=)", " not ", out)
    return out, names


def parse(text: str):
    """(tree, backtick names) when ``text`` is an expression, else None.

    A bare name or number is not an expression: it is a column (or a typo
    the missing-column message explains)."""
    if not isinstance(text, str) or _STAT_FORMS.search(text):
        return None
    source, names = _translate(text)
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError:
        return None
    if isinstance(tree.body, (ast.Name, ast.Constant)):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                return None
        elif isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.ListComp,
                               ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.Starred)):
            return None
    return tree, names


def columns_used(tree, names: dict[str, str]) -> list[str]:
    """Names read as values. A column may share a function's name (mean):
    only a name being called is the function."""
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    used = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in called and node.id not in _CONSTANTS:
            used.append(names.get(node.id, node.id))
    return list(dict.fromkeys(used))


def _evaluate(node, frame: pd.DataFrame, names: dict[str, str]):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, frame, names)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        column = frame[names.get(node.id, node.id)]
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            return column.to_numpy(dtype=np.float64, na_value=np.nan)
        return column.to_numpy()
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_evaluate(node.left, frame, names), _evaluate(node.right, frame, names))
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, frame, names)
        if isinstance(node.op, ast.USub):
            return np.negative(value)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return np.logical_not(value)
    if isinstance(node, ast.BoolOp):
        values = [_evaluate(v, frame, names) for v in node.values]
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        out = values[0]
        for value in values[1:]:
            out = combine(out, value)
        return out
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, frame, names)
        out = None
        for op, right_node in zip(node.ops, node.comparators):
            if type(op) not in _COMPARE:
                raise ValueError(f"aes() expressions do not support {type(op).__name__}")
            right = _evaluate(right_node, frame, names)
            step = _COMPARE[type(op)](left, right)
            out = step if out is None else np.logical_and(out, step)
            left = right
        return out
    if isinstance(node, ast.Call):
        args = [_evaluate(a, frame, names) for a in node.args]
        kwargs = {k.arg: _evaluate(k.value, frame, names) for k in node.keywords}
        return _FUNCTIONS[node.func.id](*args, **kwargs)
    raise ValueError(f"aes() expressions do not support {type(node).__name__}")


def _rows(data: Any, cols: list[str]) -> pd.DataFrame:
    """The columns as pandas, every row kept (missing values too)."""
    backend = detect_backend(data)
    if backend == "pandas":
        return data.loc[:, cols]
    from plot3.table import _polars_to_pandas, resolve_polars

    import polars as pl

    frame = resolve_polars(data) if backend == "tidy" else data
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    return _polars_to_pandas(frame.select(cols))


def with_expressions(data: Any, texts) -> Any:
    """``data`` plus one column per expression in ``texts`` that is not
    already a column. Other backends than pandas and polars are returned
    unchanged (their columns are positions)."""
    if data is None:
        return data
    wanted = []
    for text in dict.fromkeys(t for t in texts if isinstance(t, str)):
        if has_column(data, text):
            continue
        parsed = parse(text)
        if parsed is not None:
            wanted.append((text, parsed))
    if not wanted:
        return data
    backend = detect_backend(data)
    if backend not in {"pandas", "polars", "tidy"}:
        return data
    computed: dict[str, Any] = {}
    for text, (tree, names) in wanted:
        used = columns_used(tree, names)
        missing = [c for c in used if not has_column(data, c)]
        if missing:
            raise ColumnNotFound(missing, data)
        frame = _rows(data, used)
        value = _evaluate(tree, frame, names)
        if np.ndim(value) == 0:
            value = np.full(len(frame), value)
        computed[text] = value
    if backend == "pandas":
        out = data.copy()
        for text, value in computed.items():
            out[text] = value
        return out
    import polars as pl

    frame = data
    if backend == "tidy":
        from plot3.table import resolve_polars

        frame = resolve_polars(data)
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    series = []
    for text, value in computed.items():
        if isinstance(value, pd.Categorical):
            # polars keeps the levels' order as an Enum.
            levels = [str(c) for c in value.categories]
            series.append(pl.Series(text, np.asarray(value.astype(str)), dtype=pl.Enum(levels)))
        else:
            series.append(pl.Series(text, np.asarray(value)))
    return frame.with_columns(series)


def add_expression_columns(g):
    """A copy of the figure whose data carry a column for each aes()
    expression its layers use."""
    import copy

    def texts(mapping) -> list:
        return [v for k, v in dict(mapping or {}).items()]

    shared = []
    layers = []
    changed = False
    for layer in g.layers:
        mapping = texts(g.mapping) + texts(getattr(layer, "mapping", None))
        own = getattr(layer, "layer_data", None)
        if own is not None:
            new = with_expressions(own, mapping)
            if new is not own:
                layer = copy.copy(layer)
                layer.layer_data = new
                changed = True
        elif getattr(layer, "kind", None) not in {"function", "vector"}:
            shared.extend(mapping)
        layers.append(layer)
    data = with_expressions(g.data, shared) if g.data is not None else None
    if data is g.data and not changed:
        return g
    out = copy.copy(g)
    out.data = data
    out.layers = layers
    return out
