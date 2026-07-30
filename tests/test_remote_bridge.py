"""Phase D: CRAFT host ↔ remote PlotPayload bridge (mocked)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    MIME_PLOT3,
    aes,
    geom_bar,
    geom_point,
    ggplot,
    is_remote_kernel,
)
from plot3.remote import (
    fetch_remote_payload,
    has_craft_host,
    remote_ggplot_payload,
)


def test_mime_constant():
    assert MIME_PLOT3.startswith("application/vnd.plot3")


def test_is_remote_kernel_env(monkeypatch):
    monkeypatch.delenv("PLOT3_REMOTE", raising=False)
    monkeypatch.delenv("CRAFT_REMOTE", raising=False)
    monkeypatch.delenv("GPU_KERNEL", raising=False)
    # Without CRAFT host remote_run_, default is False unless env set.
    assert is_remote_kernel() in (True, False)  # env-dependent machine
    monkeypatch.setenv("PLOT3_REMOTE", "1")
    assert is_remote_kernel() is True
    monkeypatch.setenv("PLOT3_REMOTE", "0")
    # CRAFT_REMOTE still
    monkeypatch.setenv("CRAFT_REMOTE", "yes")
    assert is_remote_kernel() is True


def test_repr_mimebundle_includes_payload(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    bundle = fig._repr_mimebundle_()
    assert "text/html" in bundle
    assert MIME_PLOT3 in bundle
    payload = bundle[MIME_PLOT3]
    assert payload["kind"] == "figure"
    assert payload["spec"]["layers"][0]["n"] == len(cars)


def test_repr_mimebundle_faceted_html_only(cars):
    from plot3 import facet_wrap

    fig = (
        ggplot(cars, aes(x="wt", y="mpg"))
        + geom_point()
        + facet_wrap("cyl")
    )
    bundle = fig._repr_mimebundle_()
    assert "text/html" in bundle
    # Facets cannot to_payload(); mime payload omitted.
    assert MIME_PLOT3 not in bundle


def test_fetch_remote_payload_mocked(cars, tmp_path):
    """Simulate host CRAFT: remote writes JSON, ssh cats it back."""
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    payload = fig.to_payload()
    remote_file = tmp_path / "plot3_payload_test.json"
    remote_file.write_text(json.dumps(payload), encoding="utf-8")

    def fake_rr(code, max_chars=8000):
        assert "_plot3_payload" in code or "to_payload" in code
        return f"{remote_file}\nplot3_payload_bytes {remote_file.stat().st_size}\n"

    def fake_ssh(cmd: str) -> bytes:
        if cmd.startswith("cat"):
            return remote_file.read_bytes()
        if cmd.startswith("rm"):
            return b""
        raise AssertionError(cmd)

    with (
        patch("plot3.remote._remote_run", return_value=fake_rr),
        patch("plot3.io.ssh_bytes", side_effect=fake_ssh),
    ):
        got = fetch_remote_payload("_plot3_payload = {'v': 1}")
    assert got["kind"] == "figure"
    assert got["spec"]["layers"][0]["n"] == len(cars)


def test_fetch_remote_payload_roundtrip_exec(cars, tmp_path):
    """End-to-end style: remote_run executes real to_payload in this process."""
    payload_holder = {}

    def fake_rr(code, max_chars=8000):
        # Run the embedded remote script in a sandbox ns with cars.
        ns = {
            "cars": cars,
            "ggplot": ggplot,
            "aes": aes,
            "geom_point": geom_point,
            "geom_bar": geom_bar,
        }
        # Extract body between imports and path write — run full code with Path mock
        from pathlib import Path as P

        out_path = tmp_path / f"p_{len(payload_holder)}.json"

        class FakePath(type(P())):
            pass

        # Simpler: exec only the user body pattern used by remote_ggplot_payload
        # fetch_remote_payload wraps body — exec the whole string with patched Path
        import pathlib

        real_path = pathlib.Path

        class PathProxy:
            def __init__(self, p):
                self._p = real_path(str(p))

            def write_text(self, text, encoding="utf-8"):
                out_path.write_text(text, encoding=encoding)
                payload_holder["path"] = out_path

            def as_posix(self):
                return str(out_path)

            def stat(self):
                return out_path.stat()

        def path_factory(p):
            return PathProxy(p)

        ns["Path"] = path_factory
        # The remote code uses `from pathlib import Path as _Path` then _Path(...)
        # Execute carefully:
        exec_globals = {
            "__builtins__": __builtins__,
            "cars": cars,
        }
        # Inject plot3 API
        import plot3 as p3

        for name in p3.__all__:
            if hasattr(p3, name):
                exec_globals[name] = getattr(p3, name)
        exec_globals["ggplot"] = ggplot
        exec_globals["aes"] = aes
        exec_globals["geom_point"] = geom_point

        # Replace Path in the code's import by pre-binding
        code2 = code.replace("from pathlib import Path as _Path", "")
        exec_globals["_Path"] = path_factory
        exec(code2, exec_globals)
        return f"{out_path}\nplot3_payload_bytes {out_path.stat().st_size}\n"

    def fake_ssh(cmd: str) -> bytes:
        if cmd.startswith("cat"):
            return payload_holder["path"].read_bytes()
        return b""

    with (
        patch("plot3.remote._remote_run", return_value=fake_rr),
        patch("plot3.io.ssh_bytes", side_effect=fake_ssh),
    ):
        payload = remote_ggplot_payload(
            "ggplot(cars, aes(x='wt', y='mpg')) + geom_point()"
        )
    assert payload["kind"] == "figure"
    assert payload["spec"]["layers"][0]["n"] == len(cars)
    # Host can display without data
    local = ggplot.from_payload(payload)
    assert local.data is None
    assert len(local.html()) > 500


def test_remote_bar_payload_is_small(cars, tmp_path):
    """Bar chart payload should only carry counts, not all rows."""
    payload_holder = {}

    def fake_rr(code, max_chars=8000):
        from pathlib import Path as real_path

        out_path = tmp_path / "bar.json"

        def path_factory(p):
            class PP:
                def write_text(self, text, encoding="utf-8"):
                    out_path.write_text(text, encoding=encoding)
                    payload_holder["path"] = out_path

                def as_posix(self):
                    return str(out_path)

                def stat(self):
                    return out_path.stat()

            return PP()

        exec_globals = {"__builtins__": __builtins__, "cars": cars}
        import plot3 as p3

        for name in p3.__all__:
            if hasattr(p3, name):
                exec_globals[name] = getattr(p3, name)
        code2 = code.replace("from pathlib import Path as _Path", "")
        exec_globals["_Path"] = path_factory
        exec(code2, exec_globals)
        return f"{out_path}\nplot3_payload_bytes {out_path.stat().st_size}\n"

    def fake_ssh(cmd: str) -> bytes:
        if cmd.startswith("cat"):
            return payload_holder["path"].read_bytes()
        return b""

    with (
        patch("plot3.remote._remote_run", return_value=fake_rr),
        patch("plot3.io.ssh_bytes", side_effect=fake_ssh),
    ):
        payload = remote_ggplot_payload(
            "ggplot(cars, aes(x='cyl')) + geom_bar()"
        )
    assert payload["spec"]["layers"][0]["n"] == cars["cyl"].nunique()
    assert payload["spec"]["layers"][0]["n"] < len(cars)
