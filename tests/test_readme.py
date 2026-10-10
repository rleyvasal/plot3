"""Every ```python block in README.md and docs/reference.md runs, and every
figure it makes builds.

Blocks tagged ```python notest are skipped (notebook magics, CRAFT, files
the reader supplies). Blocks share one namespace, in order, like a notebook.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs" / "reference.md"]


def _blocks(doc: Path) -> list[tuple[int, str]]:
    text = doc.read_text(encoding="utf-8")
    out = []
    for match in re.finditer(r"^```(python[^\n]*)\n(.*?)^```", text, re.M | re.S):
        info, body = match.group(1), match.group(2)
        if "notest" in info:
            continue
        line = text[: match.start()].count("\n") + 1
        out.append((line, body))
    return out


def _check(value) -> None:
    from plot3.build import build_doc, build_spec
    from plot3.compose import Composition
    from plot3.ggplot import ggplot

    if isinstance(value, Composition):
        assert "<iframe" in value.html()
    elif isinstance(value, ggplot):
        if value.facet is not None:
            assert "<iframe" in build_doc(value)
        else:
            build_spec(value)


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_doc_has_examples(doc):
    assert len(_blocks(doc)) >= 5


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_doc_examples_run_and_build(doc, capsys):
    namespace: dict = {}
    for line, body in _blocks(doc):
        tree = ast.parse(body)
        for node in tree.body:
            code = compile(ast.Module(body=[node], type_ignores=[]), f"{doc.name}:{line}", "exec")
            if isinstance(node, ast.Expr):
                value = eval(compile(ast.Expression(node.value), f"{doc.name}:{line}", "eval"), namespace)
                _check(value)
                continue
            exec(code, namespace)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        _check(namespace[target.id])
    capsys.readouterr()  # the build prints sizes; keep the test output quiet
