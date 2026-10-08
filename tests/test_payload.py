"""Phase A: PlotPayload contract — serializable, renderable, no DataFrame."""

from __future__ import annotations

import json

import pytest

from plot3 import (
    PAYLOAD_VERSION,
    aes,
    build_payload,
    facet_wrap,
    geom_bar,
    geom_point,
    ggplot,
    render_payload,
    validate_payload,
)
from plot3.build import build_doc, build_spec
from plot3.payload import payload_blobs_list, payload_from_spec


def test_to_payload_matches_build_spec(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg", colour="cyl")) + geom_point()
    spec, payloads = build_spec(fig)
    payload = fig.to_payload()
    assert payload["v"] == PAYLOAD_VERSION
    assert payload["kind"] == "figure"
    assert payload["spec"] == spec
    assert payload["blobs"] == {pid: b64 for pid, b64 in payloads}


def test_build_payload_function_matches_method(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    assert build_payload(fig) == fig.to_payload()


def test_render_payload_matches_build_doc(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    # Suppress duplicate log noise by comparing structure, not print side effects.
    html_doc = build_doc(fig)
    payload = fig.to_payload()
    html_payload = render_payload(payload, log=False)
    assert html_doc == html_payload
    assert "three" in html_doc.lower() or "THREE" in html_doc
    assert "__SPEC__" not in html_doc
    assert len(html_doc) > 500


def test_payload_json_roundtrip(cars):
    """Payload must survive JSON serialize/deserialize (remote wire path)."""
    fig = ggplot(cars, aes(x="wt", y="mpg", colour="cyl")) + geom_point()
    payload = fig.to_payload()
    raw = json.dumps(payload, separators=(",", ":"))
    restored = json.loads(raw)
    validate_payload(restored)
    html = render_payload(restored, log=False)
    assert len(html) > 500
    assert restored["spec"]["layers"][0]["n"] == payload["spec"]["layers"][0]["n"]


def test_payload_has_no_dataframe(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    payload = fig.to_payload()
    # Walk the tree: no pandas objects, only JSON-friendly types.
    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        else:
            assert isinstance(obj, (str, int, float, bool, type(None)))

    walk(payload)


def test_validate_payload_rejects_bad_input(cars):
    with pytest.raises(TypeError, match="dict"):
        validate_payload([])
    with pytest.raises(ValueError, match="version"):
        validate_payload({"v": 99, "kind": "figure", "spec": {}, "blobs": {}})
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    good = fig.to_payload()
    bad = dict(good)
    bad["kind"] = "nope"
    with pytest.raises(ValueError, match="kind"):
        validate_payload(bad)
    bad2 = dict(good)
    bad2["spec"] = dict(good["spec"])
    del bad2["spec"]["layers"]
    with pytest.raises(ValueError, match="layers"):
        validate_payload(bad2)


def test_payload_from_spec_and_blobs_list(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    spec, payloads = build_spec(fig)
    payload = payload_from_spec(spec, payloads)
    assert payload_blobs_list(payload) == list(payloads)


def test_bar_payload_discrete(cars):
    fig = ggplot(cars, aes(x="cyl")) + geom_bar()
    payload = fig.to_payload()
    assert payload["spec"]["scales"]["x"]["kind"] == "cat"
    assert payload["spec"]["layers"][0]["kind"] == "col"
    html = render_payload(payload, log=False)
    assert len(html) > 500


def test_faceted_to_payload_raises(cars):
    fig = (
        ggplot(cars, aes(x="wt", y="mpg"))
        + geom_point()
        + facet_wrap("cyl")
    )
    with pytest.raises(ValueError, match="facet"):
        fig.to_payload()
    # Full HTML path still works (panel payloads under the hood).
    html = fig.html()
    assert "grid-template-columns" in html
    assert html.count("<iframe") == cars["cyl"].nunique()


def test_render_without_source_data(cars):
    """Viewer only needs the payload — drop the figure data after encode."""
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    payload = fig.to_payload()
    fig.data = None
    fig.backend = None
    html = render_payload(payload, log=False)
    assert len(html) > 500


def test_text_from_the_data_cannot_break_out_of_the_page():
    """A title or category with </script> or a tag stays text: the figure's
    JSON is escaped for <script>, and the viewer escapes tick labels."""
    import pandas as pd

    from plot3 import aes, geom_col, ggplot, labs
    from plot3.viewer import _DOC_TEMPLATE

    d = pd.DataFrame({"g": ["<img src=x onerror=alert(1)>", "a & b"], "n": [1.0, 2.0]})
    doc = (ggplot(d, aes("g", "n")) + geom_col()
           + labs(title="__PAYLOADS__ </script><script>alert(1)</script>")).html()
    assert "<img src=x" not in doc
    assert "</script><script>alert(1)" not in doc
    assert "\\u003c/script\\u003e" in doc
    assert "__PAYLOADS__" in doc  # the title's text, not swapped for the data blocks
    assert "${lab}</text>" not in _DOC_TEMPLATE  # tick labels go through plot3Esc
