"""Phase C: payload-only local display path (no live DataFrame)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plot3 import (
    aes,
    coord_3d,
    geom_bar,
    geom_histogram,
    geom_point,
    geom_point3d,
    ggplot,
    load_payload,
    render_payload,
    save_payload,
)
from plot3.build import build_doc


def test_from_payload_html_matches_original(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg", colour="cyl")) + geom_point()
    payload = fig.to_payload()
    html_live = render_payload(payload, log=False)

    frozen = ggplot.from_payload(payload)
    assert frozen.data is None
    assert frozen._payload is not None
    html_frozen = frozen.html()
    # log=True on html() may print but document body matches
    assert html_frozen == render_payload(payload, log=True)
    assert len(html_frozen) == len(html_live) or "three" in html_frozen.lower()
    assert frozen.to_payload() is payload or frozen.to_payload() == payload


def test_freeze_drops_data_and_still_renders(cars):
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    n_before = build_doc(fig)  # ensure live path works
    assert len(n_before) > 500

    fig.freeze()
    assert fig.data is None
    assert fig.backend is None
    assert fig._payload is not None
    html = fig.html()
    assert len(html) > 500
    assert "three" in html.lower() or "THREE" in html


def test_freeze_iframe_without_data(cars):
    fig = (ggplot(cars, aes(x="wt", y="mpg")) + geom_point()).freeze()
    iframe = fig._iframe()
    assert "<iframe" in iframe
    assert "srcdoc=" in iframe


def test_from_payload_3d_array():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(80, 3))
    fig = ggplot(pts, aes(x=0, y=1, z=2)) + geom_point3d() + coord_3d()
    payload = fig.to_payload()
    # Simulate remote → local: only JSON survives
    payload = json.loads(json.dumps(payload))
    local = ggplot.from_payload(payload)
    assert local.data is None
    html = local.html()
    assert len(html) > 500
    assert local.to_payload()["spec"]["is3d"] is True


def test_from_payload_bar_and_hist(cars):
    bar_p = (ggplot(cars, aes(x="cyl")) + geom_bar()).to_payload()
    hist_p = (ggplot(cars, aes(x="mpg")) + geom_histogram(bins=5)).to_payload()
    assert ggplot.from_payload(bar_p).html()
    assert ggplot.from_payload(hist_p).to_payload()["spec"]["layers"][0]["n"] == 5


def test_save_load_payload_roundtrip(cars, tmp_path: Path):
    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point()
    path = tmp_path / "fig.plot3.json"
    save_payload(fig.to_payload(), path)
    loaded = load_payload(path)
    html = ggplot.from_payload(loaded).html()
    assert len(html) > 500


def test_add_invalidates_frozen_payload(cars):
    fig = (ggplot(cars, aes(x="wt", y="mpg")) + geom_point()).freeze()
    assert fig._payload is not None
    # Grammar change clears freeze; needs data again to build
    with_labs = fig + __import__("plot3", fromlist=["labs"]).labs(title="t")
    assert with_labs._payload is None
    assert with_labs.data is None
    with pytest.raises(ValueError, match="no data"):
        with_labs.html()


def test_pipe_into_payload_backed_raises(cars):
    fig = ggplot.from_payload(
        (ggplot(cars, aes(x="wt", y="mpg")) + geom_point()).to_payload()
    )
    with pytest.raises(TypeError, match="payload-backed"):
        _ = cars >> fig


def test_freeze_faceted_raises(cars):
    from plot3 import facet_wrap

    fig = ggplot(cars, aes(x="wt", y="mpg")) + geom_point() + facet_wrap("cyl")
    with pytest.raises(ValueError, match="facet"):
        fig.freeze()
