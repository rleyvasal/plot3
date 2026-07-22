"""R-style bare-name / backtick column masking for plot3 (Jupyter layer).

Mirrors tidy3 so ggplot2-style aesthetics work without string quotes::

    aes(x=wt, y=mpg, colour=cyl)
    aes(x=`First Name`, y=`Age (%)`)
    facet_wrap(cyl)

Two-phase design (SolveIt / IPython)::

1. **Source preparser** turns backticks into a sentinel::

       `First Name`  →  __plot3_bt__("First Name")

2. **AST transformer** resolves the sentinel (and bare names) inside
   :func:`aes` / :func:`facet_wrap` as **column-name strings**::

       aes(x=wt, y=mpg)              → aes(x="wt", y="mpg")
       aes(x=`First Name`, y=mpg)    → aes(x="First Name", y="mpg")
       facet_wrap(cyl)               → facet_wrap("cyl")

Plain ``.py`` files are unchanged — keep ``aes(x="wt", y="mpg")`` there.

When tidy3 is also loaded, its backtick preparser may emit
``__tidy3_bt__("name")``; this transformer treats that sentinel the same way
inside ``aes`` / ``facet_wrap`` so dual-stack notebooks keep working.
"""

from __future__ import annotations

import ast
import builtins
import re
from typing import Any, Iterable

# Sentinels (plot3 own + tidy3 compatibility).
BT_NAME = "__plot3_bt__"
TIDY3_BT_NAME = "__tidy3_bt__"
_BT_SENTINELS = frozenset({BT_NAME, TIDY3_BT_NAME})

# Calls whose arguments / keywords are column *selectors* (→ strings).
_SELECTOR_FUNCS = frozenset(
    {
        "aes",
        "facet_wrap",
    }
)

# Keywords that are plain labels / options — never rewrite as column names.
_PASSTHROUGH_KW = frozenset(
    {
        "title",
        "subtitle",
        "caption",
        "label",
        "labels",
        "trans",
        "limits",
        "palette",
        "option",
        "scales",
        "ncol",
        "nrow",
        "bins",
        "binwidth",
        "width",
        "size",
        "alpha",
        "color",  # const color on geoms is a colour code, not a column
        "colour",
        "linewidth",
        "wireframe",
        "levels",
        "n",
        "bw",
        "kernel",
        "trim",
        "coef",
        "varwidth",
        "outlier",
        "na_rm",
        "hide",
        "height",
        "theme",
        "kind",
        "max_points",
        "size_mode",
        "fov",
        "near",
        "far",
        "target",
        "up",
        "position",
    }
)

_BT_RE = re.compile(r"`([^`\n]+)`")


def rewrite_backticks(source: str) -> str:
    """Preparse backticks: `` `col name` `` → ``__plot3_bt__("col name")``."""
    if "`" not in source:
        return source
    return _BT_RE.sub(lambda m: f"{BT_NAME}({m.group(1)!r})", source)


def plot3_backtick_transform(lines: list[str]) -> list[str]:
    """IPython input transformer: backtick preparser."""
    if not lines:
        return lines
    src = "".join(lines)
    if "`" not in src:
        return lines
    out = rewrite_backticks(src)
    if out == src:
        return lines
    if out.endswith("\n"):
        return out.splitlines(keepends=True)
    parts = [ln + "\n" for ln in out.splitlines()]
    return parts or [out]


def _is_bt_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _BT_SENTINELS
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    )


def _string_const(name: str, old: ast.AST) -> ast.AST:
    return ast.copy_location(ast.Constant(value=name), old)


class MaskSelectors(ast.NodeTransformer):
    """Rewrite bare names / backtick sentinels to string constants."""

    def __init__(self, known: set[str]):
        self.known = known

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id not in self.known:
            return _string_const(node.id, node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if _is_bt_call(node):
            return _string_const(str(node.args[0].value), node)

        # Nested calls (e.g. unlikely helpers): keep func name; mask args.
        if not isinstance(node.func, ast.Name):
            node.func = self.visit(node.func)
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [
            ast.keyword(arg=kw.arg, value=self.visit(kw.value))
            for kw in node.keywords
        ]
        return node


def default_known_names(extra: Iterable[str] | None = None) -> set[str]:
    """Names that must not become column strings (funcs, builtins, API)."""
    known = set(dir(builtins))
    known.update(_BT_SENTINELS)
    known.update({"True", "False", "None"})
    try:
        import plot3 as p3

        for name in getattr(p3, "__all__", ()):
            if not str(name).startswith("_"):
                known.add(name)
    except Exception:
        pass
    # Common frame / helper names left alone when present in the user ns
    # are added via extra from IPython.user_ns.
    if extra:
        known.update(extra)
    return known


class Plot3MaskTransformer(ast.NodeTransformer):
    """AST pass: bare names / backticks → column strings inside aes / facet_wrap."""

    def __init__(self, known: set[str] | None = None):
        self._known_static = known

    def _known(self) -> set[str]:
        if self._known_static is not None:
            return set(self._known_static)
        extra: set[str] = set()
        try:
            from IPython import get_ipython

            ip = get_ipython()
            if ip is not None and getattr(ip, "user_ns", None) is not None:
                extra.update(ip.user_ns.keys())
        except Exception:
            pass
        return default_known_names(extra)

    def _mask_selector(self, node: ast.AST) -> ast.AST:
        return MaskSelectors(self._known()).visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Always recurse so nested aes(...) inside geom_point(aes(...)) is seen.
        if not isinstance(node.func, ast.Name):
            return self.generic_visit(node)

        name = node.func.id
        if name not in _SELECTOR_FUNCS:
            return self.generic_visit(node)

        if name == "aes":
            # All positional + keyword values are column selectors.
            node.args = [self._mask_selector(a) for a in node.args]
            node.keywords = [
                ast.keyword(arg=kw.arg, value=self._mask_selector(kw.value))
                for kw in node.keywords
            ]
            return node

        if name == "facet_wrap":
            # facets (positional or facets=) is a column selector; other kwargs
            # (ncol, nrow, scales) stay as-is.
            node.args = [self._mask_selector(a) for a in node.args]
            new_kws: list[ast.keyword] = []
            for kw in node.keywords:
                if kw.arg in (None, "facets") or (
                    kw.arg is not None and kw.arg not in _PASSTHROUGH_KW
                ):
                    new_kws.append(
                        ast.keyword(arg=kw.arg, value=self._mask_selector(kw.value))
                    )
                else:
                    new_kws.append(kw)
            node.keywords = new_kws
            return node

        return self.generic_visit(node)


def apply_masking(
    source: str,
    *,
    known: set[str] | None = None,
    backticks: bool = True,
) -> str:
    """Apply backtick rewrite + AST masking; return unparsed source (tests)."""
    text = rewrite_backticks(source) if backticks else source
    tree = ast.parse(text)
    tree = Plot3MaskTransformer(known=known or default_known_names()).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def is_mask_transformer(obj: Any) -> bool:
    return isinstance(obj, Plot3MaskTransformer) or (
        type(obj).__name__ == "Plot3MaskTransformer"
        and getattr(type(obj), "__module__", "").startswith("plot3")
    )


def is_backtick_transformer(obj: Any) -> bool:
    return (
        getattr(obj, "__module__", "") in {"plot3.masking", "plot3.jupyter"}
        and getattr(obj, "__name__", "") == "plot3_backtick_transform"
    )


def is_tidy3_backtick_transformer(obj: Any) -> bool:
    """True if tidy3 already installed a backtick preparser (share it)."""
    mod = getattr(obj, "__module__", "") or ""
    name = getattr(obj, "__name__", "") or ""
    return name.endswith("backtick_transform") and (
        mod.startswith("tidy3") or name == "tidy3_backtick_transform"
    )
