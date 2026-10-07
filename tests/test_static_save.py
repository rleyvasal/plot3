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
    theme_bw,
    theme_classic,
    theme_light,
    theme_minimal,
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
    prev = bytearray(stride)
    for _ in range(height):
        filt = raw[cursor]
        cursor += 1
        row = bytearray(raw[cursor:cursor + stride])
        cursor += stride
        if filt == 1:
            for i in range(stride):
                left = row[i - 3] if i >= 3 else 0
                row[i] = (row[i] + left) & 255
        elif filt == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 255
        elif filt == 3:
            for i in range(stride):
                left = row[i - 3] if i >= 3 else 0
                row[i] = (row[i] + ((left + prev[i]) // 2)) & 255
        elif filt == 4:
            for i in range(stride):
                left = row[i - 3] if i >= 3 else 0
                up = prev[i]
                upper_left = prev[i - 3] if i >= 3 else 0
                row[i] = (row[i] + _paeth(left, up, upper_left)) & 255
        else:
            assert filt == 0
        prev = row
        rows.append(np.frombuffer(row, dtype=np.uint8).copy())
    return np.vstack(rows).reshape(height, width, 3)


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    dl = abs(estimate - left)
    du = abs(estimate - up)
    dul = abs(estimate - upper_left)
    if dl <= du and dl <= dul:
        return left
    if du <= dul:
        return up
    return upper_left


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

    other = tmp_path / "fig.txt"
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
    assert 'font-family="Helvetica, Arial, sans-serif"' in svg


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


def _curve():
    return ggplot() + geom_function("y = x", xlim=(0, 1), n=4)


def test_publication_themes_are_white_and_the_default_stays_dark(tmp_path):
    dark = _curve()
    assert dark.theme_name == "dark"
    bw = dark + theme_bw()
    classic = dark + theme_classic()
    minimal = dark + theme_minimal()
    assert bw.theme_name == "bw"
    assert classic.theme_name == "classic"
    assert minimal.theme_name == "minimal"
    # Adding another theme replaces the font choice from the previous one.
    sized = bw + theme_bw(base_size=11, base_family="Arial")
    assert sized.theme_base_size == 11
    assert sized.theme_family == "Arial"
    assert (sized + theme_classic()).theme_family is None
    sized_svg = tmp_path / "sized.svg"
    ggsave(sized_svg, sized, width=420, height=280)
    sized_text = sized_svg.read_text(encoding="utf-8")
    # 11 pt base: axis text is 0.8× and the title is 1.2×, in CSS pixels.
    assert 'font-family="Arial"' in sized_text
    assert 'font-size="18"' in sized_text
    assert 'font-size="12"' in sized_text

    bw_svg = tmp_path / "bw.svg"
    classic_svg = tmp_path / "classic.svg"
    minimal_svg = tmp_path / "minimal.svg"
    ggsave(bw_svg, bw, width=420, height=280)
    ggsave(classic_svg, classic, width=420, height=280)
    ggsave(minimal_svg, minimal, width=420, height=280)
    bw_text = bw_svg.read_text(encoding="utf-8")
    classic_text = classic_svg.read_text(encoding="utf-8")
    minimal_text = minimal_svg.read_text(encoding="utf-8")
    assert 'fill="#ffffff"' in bw_text
    assert 'stroke="#ebebeb"' in bw_text
    assert 'stroke="#333333"' in bw_text
    # Classic keeps the two axis lines and drops the grey grid.
    assert classic_text.count("<line ") == 2
    assert 'stroke="#000000"' in classic_text
    assert 'stroke="#ebebeb"' not in classic_text
    assert 'stroke="#ebebeb"' in minimal_text
    assert 'stroke="#000000"' not in minimal_text
    assert 'stroke="#333333"' not in minimal_text


def test_inches_and_dpi_set_the_png_page(tmp_path, monkeypatch, capsys):
    from plot3 import static

    monkeypatch.setattr(static, "_load_cairosvg", lambda: None)
    fig = _curve() + theme_bw()
    path = tmp_path / "journal.png"
    ggsave(path, fig, width=7, height=4, units="in", dpi=300)
    rgb = _read_png(path)
    assert rgb.shape == (1200, 2100, 3)
    assert tuple(rgb[0, 0]) == (255, 255, 255)
    assert "plot3[export]" in capsys.readouterr().out

    cm = tmp_path / "cm.png"
    ggsave(cm, fig, width=2.54, height=2.54, units="cm", dpi=100)
    assert _read_png(cm).shape == (100, 100, 3)


def test_svg_uses_physical_size_family_and_point_size(tmp_path):
    fig = _curve() + theme_bw(base_family="Helvetica") + labs(y="$x^2$")
    path = tmp_path / "page.svg"
    ggsave(path, fig, width=7, height=4, units="in", dpi=300, fontsize=20, family="Arial")
    svg = path.read_text(encoding="utf-8")
    assert 'width="7in"' in svg
    assert 'height="4in"' in svg
    assert 'viewBox="0 0 672 384"' in svg
    assert 'font-family="Arial"' in svg
    # 20 pt → 20 * 96/72 px; title is 1.2× that, axis text is 0.8×.
    assert 'font-size="32"' in svg
    assert 'font-size="21"' in svg
    assert "x²" in svg
    assert "$" not in svg


def test_pdf_without_cairosvg_names_the_extra(tmp_path, monkeypatch):
    from plot3 import static

    monkeypatch.setattr(static, "_load_cairosvg", lambda: None)
    with pytest.raises(RuntimeError, match=r"plot3\[export\]"):
        ggsave(tmp_path / "fig.pdf", _curve())
    assert not (tmp_path / "fig.pdf").exists()


def test_units_and_dpi_are_not_silently_dropped():
    fig = _curve()
    with pytest.raises(ValueError, match="units"):
        ggsave("fig.png", fig, units="pt")
    with pytest.raises(TypeError, match="units='in'"):
        ggsave("fig.png", fig, width="7in", units="in")
    with pytest.raises(ValueError, match="dpi applies"):
        ggsave("fig.png", fig, width=800, dpi=300)


def test_cairosvg_png_and_pdf_match_the_svg(tmp_path):
    pytest.importorskip("cairosvg")
    fig = _curve() + theme_bw() + labs(title="Hello", y="y")
    png = tmp_path / "fig.png"
    pdf = tmp_path / "fig.pdf"
    svg = tmp_path / "fig.svg"
    ggsave(png, fig, width=3.5, height=2.5, units="in", dpi=300)
    ggsave(pdf, fig, width=3.5, height=2.5, units="in")
    ggsave(svg, fig, width=3.5, height=2.5, units="in")
    rgb = _read_png(png)
    assert rgb.shape == (750, 1050, 3)
    assert tuple(int(v) for v in rgb[0, 0]) == (255, 255, 255)
    # Antialiased Helvetica has greys. The bitmap fallback has only flat colours.
    title = rgb[8:70, 20:280]
    assert len(np.unique(title.reshape(-1, 3), axis=0)) > 4
    assert pdf.read_bytes().startswith(b"%PDF")
    text = svg.read_text(encoding="utf-8")
    assert 'width="3.5in"' in text
    assert 'viewBox="0 0 336 240"' in text
    assert ">Hello</text>" in text
    assert ">y</text>" in text
