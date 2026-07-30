"""CRAFT / GPU bridge: stats on the remote, render on the host.

Phase D of the remote/local split.

Host (SolveIt + CRAFT connected)
    ``remote_run_`` executes Python on the GPU kernel.
    :func:`fetch_remote_payload` runs a ggplot expression there, pulls only
    the compact PlotPayload (via temp file + SSH), and the host calls
    :func:`plot3.payload.display_payload` / :func:`render_payload`.

Remote kernel (``%gpu`` cell body)
    ``ggplot(...)._repr_mimebundle_`` includes
    ``application/vnd.plot3.v1+json`` so a future host can intercept the
    payload; HTML remains the fallback for display.

Environment
    ``PLOT3_REMOTE=1`` forces remote-kernel display behaviour (payload mime
    preferred, lighter show path).
"""

from __future__ import annotations

import json
import os
import shlex
import uuid
from typing import Any, Callable

# Custom Jupyter mime type for PlotPayload handoff (host may ignore → HTML).
MIME_PLOT3 = "application/vnd.plot3.v1+json"

__all__ = [
    "MIME_PLOT3",
    "is_remote_kernel",
    "has_craft_host",
    "fetch_remote_payload",
    "remote_ggplot_payload",
    "show_remote",
]


def is_remote_kernel() -> bool:
    """True when this process should act as the GPU/remote side of the bridge.

    Heuristics (any match):

    * ``PLOT3_REMOTE=1`` / ``true`` / ``yes``
    * ``CRAFT_REMOTE=1`` or ``GPU_KERNEL=1``
    * IPython ns has CRAFT remote markers but **not** host ``remote_run_``
    """
    for key in ("PLOT3_REMOTE", "CRAFT_REMOTE", "GPU_KERNEL"):
        raw = (os.environ.get(key) or "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
    try:
        from IPython import get_ipython

        ip = get_ipython()
        ns = (getattr(ip, "user_ns", None) or {}) if ip is not None else {}
    except Exception:
        ns = {}
    # Host CRAFT always exposes remote_run_ for talking *to* the GPU.
    if callable(ns.get("remote_run_")):
        return False
    # Remote kernel often has GPU-oriented markers without host helpers.
    for key in ("_craft_remote", "IS_GPU_KERNEL", "cuda", "CUDA_VISIBLE_DEVICES"):
        if key in ns:
            return True
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None and not callable(
        ns.get("remote_run_")
    ):
        # Weak signal — only if explicitly on a GPU box without host CRAFT.
        if (os.environ.get("PLOT3_ASSUME_REMOTE") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return True
    return False


def has_craft_host() -> bool:
    """True when the host IPython session can call ``remote_run_``."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        ns = (getattr(ip, "user_ns", None) or {}) if ip is not None else {}
    except Exception:
        return False
    return callable(ns.get("remote_run_"))


def _remote_run() -> Callable[..., str]:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        raise RuntimeError("fetch_remote_payload requires IPython")
    rr = (ip.user_ns or {}).get("remote_run_")
    if not callable(rr):
        raise RuntimeError(
            "remote_run_ missing — load CRAFT and run %gpu on the host first"
        )
    return rr


def fetch_remote_payload(
    remote_source: str,
    *,
    max_chars: int = 8000,
) -> dict[str, Any]:
    """Execute *remote_source* on the GPU kernel and return a PlotPayload.

    *remote_source* must leave a dict named ``_plot3_payload`` in scope
    (or assign the final expression result). Implementation writes JSON to a
    temp file on the remote and streams it back with SSH (same pattern as
    :func:`plot3.jupyter.remote_df`), so large blobs are not limited by
    ``remote_run_`` stdout caps.
    """
    from plot3.io import ssh_bytes
    from plot3.payload import validate_payload

    rr = _remote_run()
    tmp = f"/tmp/plot3_payload_{uuid.uuid4().hex}.json"
    # Indent remote body for the embedded script.
    body = remote_source.strip()
    if not body.endswith("\n"):
        body += "\n"
    code = f"""
import json as _json
from pathlib import Path as _Path
{body}
if "_plot3_payload" not in dir() and "_plot3_payload" not in locals():
    raise RuntimeError(
        "remote plot source must set _plot3_payload = fig.to_payload()"
    )
_p = _Path({tmp!r})
_p.write_text(_json.dumps(_plot3_payload, separators=(",", ":")), encoding="utf-8")
print(_p.as_posix())
print("plot3_payload_bytes", _p.stat().st_size)
"""
    out = (rr(code, max_chars=max_chars) or "").strip()
    # Path line: absolute or relative *.json written by the remote helper.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    path_line = None
    for ln in reversed(lines):
        if not ln.endswith(".json"):
            continue
        if "plot3_payload" in ln or ln.startswith("/") or ln[1:3] == ":\\":
            path_line = ln
            break
    if not path_line:
        for ln in reversed(lines):
            if ln.endswith(".json") and not ln.startswith("plot3_payload"):
                path_line = ln
                break
    if not path_line:
        raise RuntimeError(
            "remote payload path not found in remote_run_ output:\n"
            + (out[-800:] if out else "(empty)")
        )
    try:
        raw = ssh_bytes("cat -- " + shlex.quote(path_line))
    finally:
        try:
            ssh_bytes("rm -f -- " + shlex.quote(path_line))
        except Exception:
            pass
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"invalid remote PlotPayload JSON: {e}") from e
    return validate_payload(data)


def remote_ggplot_payload(
    expr: str,
    *,
    max_chars: int = 8000,
) -> dict[str, Any]:
    """Evaluate a ggplot expression on the remote kernel; return PlotPayload.

    Example::

        payload = remote_ggplot_payload(
            "ggplot(df, aes(x='wt', y='mpg')) + geom_point()"
        )
    """
    expr = expr.strip()
    source = f"""
_fig = ({expr})
if not hasattr(_fig, "to_payload"):
    raise TypeError(
        f"remote expression must produce a ggplot, got {{type(_fig).__name__}}"
    )
_plot3_payload = _fig.to_payload()
"""
    return fetch_remote_payload(source, max_chars=max_chars)


def show_remote(
    expr: str,
    *,
    browser: bool | None = None,
    height: str | int = "480px",
    hide: bool | None = None,
    max_chars: int = 8000,
):
    """Run *expr* on the remote kernel and display the figure on the host.

    Stats / encoding run where the data lives; only the PlotPayload crosses
    the wire. Viewer HTML is built locally.
    """
    from plot3.payload import display_payload

    payload = remote_ggplot_payload(expr, max_chars=max_chars)
    return display_payload(
        payload, browser=browser, height=height, hide=hide
    )
