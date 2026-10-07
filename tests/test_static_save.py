"""Static PNG and SVG export."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plot3 import (
    aes,
    coord_3d,
    facet_wrap,
    geom_boxplot,
    geom_col,
    geom_density,
    geom_function,
    geom_histogram,
    geom_point,
    geom_violin,
    ggplot,
    ggsave,
    labs,
    slider,
    theme_light,
    transition_time,
)


def _read_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    ihdr = None
    idat = []
    while pos + 8 <= len(data):
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + size]
        if tag == b"IHDR":
            ihdr = chunk
        elif tag == b"IDAT":
            idat.append(chunk)
        elif tag == b"IEND":
            break
        pos += 12 + size
    assert ihdr is not None
    width, height, depth, color, _comp, _filt, inter = struct.unpack(">IIBBBBB", ihdr)
    assert (depth, color, inter) == (8, 2, 0)
    raw = zlib.decompress(b"".join(idat))
    stride = width * 3
    rows = []
    cursor = 0
    for _ in range(height):
        assert raw[cursor] == 0
        cursor += 1
        rows.append(np.frombuffer(raw[cursor:cursor + stride], dtype=np.uint8).copy())
        cursor += stride
    return np.vstack(rows).reshape(height, width, 3)


def _polyline_points(svg: str) -> list[tuple[float, float]]:
    start = svg.index("<polyline")
    points = svg[start:].split('points="', 1)[1].split('"', 1)[0]
    out = []
    for pair in points.split():
        x, y = pair.split(",")
        out.append((float(x), float(y)))
    return out


def test_html_save_is_unchanged_and_other_suffixes_stay_html(tmp_path, capsys):
    fig = ggplot() + geom_function("y = x", xlim=(0, 1), n=5)
    html_path = tmp_path / "fig.html"
    ggsave(html_path, fig)
    text = html_path.read_text(encoding="utf-8")
    assert text.lower().startswith("<!doctype html>")
    assert "saved" in capsys.readouterr().out

    other = tmp_path / "fig.pdf"
    fig.save(other)
    assert other.read_text(encoding="utf-8").lower().startswith("<!doctype html>")


def test_png_has_size_background_and_marks(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 3.0, 2.0]})
    fig = ggplot(df, aes(x="x", y="y")) + geom_point(alpha=1)
    path = tmp_path / "fig.png"
    ggsave(str(path), fig, width=640, height=400)
    rgb = _read_png(path)
    assert rgb.shape == (400, 640, 3)
    assert tuple(rgb[0, 0]) == (0x0B, 0x10, 0x20)
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 4
    # Default dark categorical blue is on the page.
    assert np.any(np.all(rgb == (0x39, 0x87, 0xE5), axis=-1))


def test_svg_scatter_has_circles_axes_and_theme(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 4.0], "g": ["a", "b"]})
    fig = ggplot(df, aes(x="x", y="y", color="g")) + geom_point() + theme_light()
    path = tmp_path / "fig.svg"
    ggsave(path, fig, width=500, height=320)
    svg = path.read_text(encoding="utf-8")
    assert svg.startswith("<?xml")
    assert "<svg " in svg
    assert svg.count("<circle") >= 2
    assert 'fill="#fcfcfb"' in svg
    assert ">a</text>" in svg
    assert ">b</text>" in svg
    assert ">x</text>" in svg
    assert ">y</text>" in svg


def test_function_curve_is_a_parabola(tmp_path):
    fig = ggplot() + geom_function("y = x^2", xlim=(-2, 2), n=21)
    path = tmp_path / "curve.svg"
    fig.save(path, width=480, height=360)
    pts = _polyline_points(path.read_text(encoding="utf-8"))
    assert len(pts) == 21
    mid = min(pts, key=lambda p: abs(p[0] - np.median([q[0] for q in pts])))
    ends = (pts[0][1] + pts[-1][1]) / 2
    # y = x^2 is lowest at the vertex, which is lower on the page.
    assert mid[1] > ends + 40


def test_bars_are_rects(tmp_path):
    df = pd.DataFrame({"cat": ["a", "b", "c"], "y": [1.0, 3.0, 2.0]})
    fig = ggplot(df, aes(x="cat", y="y")) + geom_col()
    path = tmp_path / "bars.svg"
    ggsave(path, fig)
    svg = path.read_text(encoding="utf-8")
    filled = [
        line for line in svg.split("<rect ")
        if 'fill="#' in line and 'fill="none"' not in line and 'width="800"' not in line
    ]
    assert len(filled) >= 3


def test_slider_opens_flat_and_transition_opens_on_the_last_frame(tmp_path):
    slider_fig = (
        ggplot()
        + geom_function("y = a x^2", n=11)
        + slider(a=(0, 3), steps=3)
        + labs(title="a = {frame_time}")
    )
    slider_svg = tmp_path / "slider.svg"
    ggsave(slider_svg, slider_fig, width=480, height=320)
    text = slider_svg.read_text(encoding="utf-8")
    assert "a = 0.00" in text
    assert "{frame_time}" not in text
    pts = _polyline_points(text)
    assert max(p[1] for p in pts) - min(p[1] for p in pts) < 1.5

    swept = (
        ggplot()
        + geom_function("y = a x^2", n=11)
        + transition_time(a=(0, 3), frames=3)
        + labs(title="a = {frame_time}")
    )
    swept_svg = (tmp_path / "sweep.svg")
    ggsave(swept_svg, swept)
    text = swept_svg.read_text(encoding="utf-8")
    assert "a = 3.00" in text
    pts = _polyline_points(text)
    assert max(p[1] for p in pts) - min(p[1] for p in pts) > 80


def test_year_title_uses_the_last_frame(tmp_path):
    df = pd.DataFrame({
        "x": [1.0, 1.0],
        "y": [1.0, 2.0],
        "year": [2000, 2001],
        "country": ["A", "A"],
    })
    fig = (
        ggplot(df, aes(x="x", y="y", group="country"))
        + geom_point()
        + transition_time("year")
        + labs(title="Year {frame_time}")
    )
    path = tmp_path / "year.svg"
    ggsave(path, fig)
    assert ">Year 2001</text>" in path.read_text(encoding="utf-8")


def test_stats_geoms_draw(tmp_path):
    rng = np.random.default_rng(0)
    box = pd.DataFrame({"g": ["a"] * 8, "y": [1, 2, 3, 4, 5, 6, 7, 30]})
    hist = pd.DataFrame({"x": rng.normal(size=40)})
    density = ggplot(hist, aes(x="x")) + geom_density(n=32)
    boxes = ggplot(box, aes(x="g", y="y")) + geom_boxplot()
    violins = ggplot(box, aes(x="g", y="y")) + geom_violin(n=16)
    bars = ggplot(hist, aes(x="x")) + geom_histogram(bins=5)
    for fig, name, needle in (
        (density, "density.svg", "<polygon"),
        (boxes, "box.svg", "<line"),
        (violins, "violin.svg", "<path"),
        (bars, "hist.svg", "<rect"),
    ):
        path = tmp_path / name
        ggsave(path, fig, width=420, height=280)
        assert needle in path.read_text(encoding="utf-8")


def test_facets_tile_panels(tmp_path):
    df = pd.DataFrame({
        "x": [1, 2, 3, 1, 2, 3],
        "y": [1, 2, 3, 3, 2, 1],
        "g": ["left", "left", "left", "right", "right", "right"],
    })
    fig = (
        ggplot(df, aes(x="x", y="y"))
        + geom_point()
        + facet_wrap("g", ncol=2)
        + labs(title="panels")
    )
    path = tmp_path / "nested" / "facets.svg"
    ggsave(path, fig, width=640, height=320)
    text = path.read_text(encoding="utf-8")
    assert ">panels</text>" in text
    assert "panels — left" in text
    assert "panels — right" in text
    assert path.parent.is_dir()


def test_3d_center_point_projects_near_the_middle(tmp_path):
    df = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]})
    fig = ggplot(df, aes(x="x", y="y", z="z")) + geom_point(size=0.08) + coord_3d()
    path = tmp_path / "cloud.png"
    ggsave(path, fig, width=400, height=400)
    rgb = _read_png(path)
    mask = np.all(rgb == (0x39, 0x87, 0xE5), axis=-1)
    ys, xs = np.nonzero(mask)
    assert len(xs) >= 8
    assert abs(float(xs.mean()) - 200) < 8
    assert abs(float(ys.mean()) - 200) < 8


def test_3d_surface_saves(tmp_path):
    fig = ggplot() + geom_function("z = sin(x) cos(y)", n=8)
    path = tmp_path / "surface.svg"
    ggsave(path, fig, width=360, height=280)
    svg = path.read_text(encoding="utf-8")
    assert "<polygon" in svg
    assert "<svg " in svg


def test_ggsave_accepts_swapped_arguments_and_case(tmp_path):
    fig = ggplot() + geom_function("y = x", xlim=(0, 1), n=4)
    svg = tmp_path / "swap.svg"
    png = tmp_path / "SWAP.PNG"
    assert ggsave(fig, svg) == str(svg)
    assert ggsave(fig, png) == str(png)
    assert svg.read_text(encoding="utf-8").startswith("<?xml")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_unsupported_layer_kind():
    from plot3 import static

    spec = {
        "is3d": False,
        "theme": {
            "surface": "#ffffff", "ink": "#000000", "ink2": "#333333",
            "muted": "#888888", "grid": "#eeeeee", "axis": "#cccccc",
            "cat": ["#3987e5"], "seq": ["#000000", "#ffffff"],
        },
        "labs": {"title": "", "x": "x", "y": "y", "z": "", "color": ""},
        "scales": {
            "x": {"kind": "num", "lo": 0, "hi": 1, "ticks": [[0, "0"], [1, "1"]]},
            "y": {"kind": "num", "lo": 0, "hi": 1, "ticks": [[0, "0"], [1, "1"]]},
        },
        "layers": [{"kind": "ribbon", "n": 1, "alpha": 1}],
        "color": {"kind": "none"},
        "notes": [],
        "gz": 0,
    }
    with pytest.raises(ValueError, match="ribbon"):
        static._draw_spec(spec, {}, 0, 0, 200, 150, [], border=False)


def test_viewer_html_has_save_menu():
    df = pd.DataFrame({"x": [1.0], "y": [1.0]})
    html = (ggplot(df, aes(x="x", y="y")) + geom_point()).html()
    fig = html.split('id="fig"', 1)[1].split('id="player"', 1)[0]
    assert 'id="modebar"' in fig
    assert 'id="save-btn"' in fig
    assert 'aria-haspopup="menu"' in fig
    assert 'aria-expanded="false"' in fig
    assert 'role="menu"' in fig
    assert 'role="menuitem"' in fig
    assert 'data-act="html"' in fig
    assert 'data-act="svg"' in fig
    assert 'data-act="png"' in fig
    assert 'data-act="video"' in fig
    assert 'data-act="copy"' in fig
    assert "Interactive figure" in fig
    assert "Vector, for papers" in fig
    assert "Image (2× sharp)" in fig
    assert "Copy PNG to clipboard" in fig
    assert 'id="save-png"' not in html
    assert 'id="save-svg"' not in html
    assert 'id="play-rec"' not in html
    assert "Vector axes" in html
    assert "showSaveFilePicker" in html
    assert "Downloads are blocked here" in html
    assert "preserveDrawingBuffer: true" in html
    assert "#fig:hover #modebar" in html
    assert "@media (hover:none)" in html
    assert "const PRISTINE" in html


def test_width_is_pixels_not_inches():
    from plot3.static import _pixels

    assert _pixels(20, "width") == 20
    with pytest.raises(ValueError, match="at least 1"):
        _pixels(0, "width")
    with pytest.raises(TypeError, match="pixel count"):
        _pixels("7in", "height")
