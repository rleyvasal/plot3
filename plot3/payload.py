"""Serializable PlotPayload — the contract between stats and the viewer.

Remote/local split:

* **Data / stats side** (local or GPU) produces a ``PlotPayload`` via
  :func:`build_payload` / :meth:`ggplot.to_payload`.
* **Viewer side** turns that payload into HTML via :func:`render_payload`
  or a payload-backed figure via :meth:`ggplot.from_payload`.

A payload does not hold a DataFrame. It holds the already-encoded wire format
that the three.js template consumes (JSON spec + base64 binary blobs).

Schema (``kind == "figure"``)::

    {
      "v": 1,
      "kind": "figure",
      "spec": { ... },          # same dict as build_spec()[0]
      "blobs": { "p0x": "<b64>", ... },
    }

Faceted figures are still assembled by :func:`plot3.build.build_doc` as a
grid of independent figure payloads (each panel is a normal figure payload).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Wire-format version for PlotPayload (independent of spec["v"]).
PAYLOAD_VERSION = 1

__all__ = [
    "PAYLOAD_VERSION",
    "build_payload",
    "render_payload",
    "validate_payload",
    "payload_from_spec",
    "payload_blobs_list",
    "save_payload",
    "load_payload",
    "display_payload",
]


def payload_from_spec(
    spec: dict[str, Any],
    payloads: list[tuple[str, str]],
) -> dict[str, Any]:
    """Build a figure PlotPayload from :func:`plot3.build.build_spec` output."""
    blobs = {str(pid): str(b64) for pid, b64 in payloads}
    return {
        "v": PAYLOAD_VERSION,
        "kind": "figure",
        "spec": spec,
        "blobs": blobs,
    }


def payload_blobs_list(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Restore ``build_spec``-style ``[(id, b64), ...]`` from a payload."""
    blobs = payload.get("blobs") or {}
    if not isinstance(blobs, dict):
        raise TypeError("payload['blobs'] must be a dict")
    return [(str(pid), str(b64)) for pid, b64 in blobs.items()]


def validate_payload(payload: Any) -> dict[str, Any]:
    """Validate a PlotPayload dict; return it on success.

    Raises ``TypeError`` / ``ValueError`` with actionable messages.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"PlotPayload must be a dict, got {type(payload).__name__}")
    if payload.get("v") != PAYLOAD_VERSION:
        raise ValueError(
            f"unsupported PlotPayload version {payload.get('v')!r}; "
            f"expected v={PAYLOAD_VERSION}"
        )
    kind = payload.get("kind")
    if kind != "figure":
        raise ValueError(
            f"unsupported PlotPayload kind {kind!r}; expected 'figure'"
        )
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        raise TypeError("PlotPayload['spec'] must be a dict")
    for key in ("v", "is3d", "theme", "labs", "scales", "layers"):
        if key not in spec:
            raise ValueError(f"PlotPayload spec missing required key {key!r}")
    if not isinstance(spec["layers"], list):
        raise TypeError("PlotPayload spec['layers'] must be a list")
    blobs = payload.get("blobs")
    if not isinstance(blobs, dict):
        raise TypeError("PlotPayload['blobs'] must be a dict")
    for pid, b64 in blobs.items():
        if not isinstance(pid, str) or not isinstance(b64, str):
            raise TypeError("PlotPayload blobs must map str id -> str base64")
    return payload


def build_payload(g) -> dict[str, Any]:
    """Compute a PlotPayload for a single-panel (non-faceted) figure.

    Runs stats / encoding where the data lives. Does not produce HTML.

    Faceted figures (``facet_wrap``) should use :func:`plot3.build.build_doc`,
    which builds one payload per panel. Calling this on a faceted ggplot
    raises ``ValueError``.
    """
    if getattr(g, "facet", None) is not None:
        raise ValueError(
            "to_payload() / build_payload() do not support facet_wrap(); "
            "use build_doc() / fig.html() which assembles panel payloads"
        )
    from plot3.build import build_spec

    spec, payloads = build_spec(g)
    return payload_from_spec(spec, payloads)


def render_payload(
    payload: dict[str, Any],
    *,
    log: bool = True,
) -> str:
    """Turn a validated figure PlotPayload into a standalone HTML document.

    This is the **viewer** half: no DataFrame access, only ``spec`` + ``blobs``.
    """
    from plot3.viewer import _DOC_TEMPLATE as DOC_TEMPLATE
    from plot3.viewer import _KATEX_BOOT

    payload = validate_payload(payload)
    spec = payload["spec"]
    blobs = payload["blobs"]

    # Preserve insertion order from build_spec (viewer looks up by id).
    blocks = "\n".join(
        f'<script type="text/plain" id="{pid}">{b64}</script>'
        for pid, b64 in blobs.items()
    )
    doc = (
        DOC_TEMPLATE
        .replace("__SPEC__", json.dumps(spec, separators=(",", ":")))
        .replace("__PAYLOADS__", blocks)
        .replace("__KATEX__", _KATEX_BOOT if spec.get("math") else "")
    )
    if log:
        kb = len(doc) // 1024
        # The size line is the one worth seeing unprompted. The row count is
        # opt-in so a notebook cell is just the plot.
        if os.environ.get("PLOT3_VERBOSE", "").strip() == "1":
            rows = sum(int(sp.get("n", 0)) for sp in spec.get("layers", []))
            n_layers = len(spec.get("layers", []))
            print(
                f"plot3: {n_layers} layer(s), {rows:,} rows -> {kb:,} KB "
                f"portable HTML{' (3D)' if spec.get('is3d') else ''}"
            )
        if kb > 1500:
            print(
                "plot3: warning — figure may exceed sslive's ~1.8 MB in-slide cap"
            )
    return doc


def save_payload(payload: dict[str, Any], path: str | Path) -> str:
    """Write a PlotPayload to a JSON file (remote → local handoff artifact)."""
    payload = validate_payload(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    return str(path.resolve())


def load_payload(path: str | Path) -> dict[str, Any]:
    """Load and validate a PlotPayload JSON file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate_payload(data)


def display_payload(
    payload: dict[str, Any],
    *,
    browser: bool | None = None,
    path: str | Path | None = None,
    height: str | int = "480px",
    hide: bool | None = None,
):
    """Show a PlotPayload without a live DataFrame (local viewer half).

    Builds a payload-backed :class:`~plot3.ggplot.ggplot` and calls
    :meth:`~plot3.ggplot.ggplot.show`.
    """
    from plot3.ggplot import ggplot

    fig = ggplot.from_payload(payload, height=height, hide=hide)
    return fig.show(browser=browser, path=path)
