"""Static PNG and SVG export."""

from __future__ import annotations

import math
import re
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
    theme,
    theme_bw,
    theme_dark,
    theme_classic,
    theme_light,
    theme_minimal,
    transition_time,
)
from plot3.__version__ import __version__
from plot3.build import build_spec


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
    # Saved files default to theme_bw: a white page, and points with no
    # colour of their own in black, as ggplot2 draws them.
    assert tuple(rgb[0, 0]) == (0xFF, 0xFF, 0xFF)
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 4
    assert np.any(np.all(rgb == (0, 0, 0), axis=-1))
    assert not np.any(np.all(rgb == (0x2A, 0x78, 0xD6), axis=-1))
    # A theme you add wins over that default.
    dark = tmp_path / "dark.png"
    ggsave(str(dark), fig + theme_dark(), width=640, height=400)
    assert tuple(_read_png(dark)[0, 0]) == (0x0B, 0x10, 0x20)


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
    # ggplot2 layout: the title once, a strip label on each panel.
    assert text.count(">panels</text>") == 1
    assert ">left</text>" in text
    assert ">right</text>" in text
    assert "panels — left" not in text
    assert path.parent.is_dir()


def test_3d_center_point_projects_near_the_middle(tmp_path):
    df = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]})
    fig = (
        ggplot(df, aes(x="x", y="y", z="z")) + geom_point(size=0.08, colour="#3987e5")
        + coord_3d() + theme_dark()
    )
    path = tmp_path / "cloud.png"
    ggsave(path, fig, width=400, height=400)
    rgb = _read_png(path)
    mask = np.all(rgb == (0x39, 0x87, 0xE5), axis=-1)
    ys, xs = np.nonzero(mask)
    assert len(xs) >= 8
    # The cube's outline is centred in the room left above the axis labels,
    # and perspective (near corners larger) sets its centre a little above
    # the middle of that outline.
    assert abs(float(xs.mean()) - 200) < 16
    assert 120 < float(ys.mean()) < 200


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
    assert f"<metadata>plot3 {__version__}</metadata>" in svg
    assert 'font-family="Arial"' in svg
    # 20 pt → 20 * 96/72 px; title is 1.2× that, axis text is 0.8×.
    assert 'font-size="32"' in svg
    assert 'font-size="21"' in svg
    assert "x²" in svg
    assert "$" not in svg


def test_pdf_without_cairosvg_names_the_extra(tmp_path, monkeypatch):
    from plot3 import static

    monkeypatch.setattr(static, "_load_cairosvg", lambda: None)
    with pytest.raises(RuntimeError, match=r"plot3\[export\]") as caught:
        ggsave(tmp_path / "fig.pdf", _curve())
    message = str(caught.value)
    assert "Cairo" in message
    assert ".svg" in message
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
    raw = pdf.read_bytes()
    assert raw.startswith(b"%PDF")
    assert f"/Creator (plot3 {__version__})".encode() in raw
    start = raw.rfind(b"startxref")
    offset = int(raw[start + len(b"startxref"):].split()[0])
    assert raw[offset:offset + 4] == b"xref"
    assert _png_phys(png) == (11811, 11811, 1)
    text = svg.read_text(encoding="utf-8")
    assert 'width="3.5in"' in text
    assert 'viewBox="0 0 336 240"' in text
    assert ">Hello</text>" in text
    assert ">y</text>" in text


def _png_chunks(path: Path) -> list[tuple[bytes, bytes]]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    chunks = []
    while pos + 8 <= len(data):
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + size]
        chunks.append((tag, payload))
        pos += 12 + size
        if tag == b"IEND":
            break
    return chunks


def _png_phys(path: Path) -> tuple[int, int, int]:
    chunks = _png_chunks(path)
    phys = [payload for tag, payload in chunks if tag == b"pHYs"]
    assert len(phys) == 1
    assert [tag for tag, _payload in chunks[:2]] == [b"IHDR", b"pHYs"]
    return struct.unpack(">IIB", phys[0])


def _svg_texts(svg: str) -> list[tuple[float, float, str]]:
    return [
        (float(x), float(y), text)
        for x, y, text in re.findall(
            r'<text x="([^"]+)" y="([^"]+)"[^>]*>([^<]*)</text>',
            svg,
        )
    ]


def _svg_circles(svg: str) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in re.findall(r'<circle cx="([^"]+)" cy="([^"]+)"', svg)
    ]


def _scatter():
    frame = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 3.0, 2.0],
        "g": ["a", "b", "a"],
    })
    return ggplot(frame, aes(x="x", y="y", color="g")) + geom_point()


def test_png_records_pixels_per_metre(tmp_path, monkeypatch):
    from plot3 import static

    monkeypatch.setattr(static, "_load_cairosvg", lambda: None)
    fig = _curve() + theme_bw()
    physical = tmp_path / "journal.png"
    ggsave(physical, fig, width=2, height=2, units="in", dpi=300)
    assert _png_phys(physical) == (11811, 11811, 1)
    assert _read_png(physical).shape == (600, 600, 3)

    pixels = tmp_path / "screen.png"
    ggsave(pixels, fig, width=200, height=120)
    # 96 CSS dpi. Word would otherwise treat the file as 72 dpi.
    assert _png_phys(pixels) == (3780, 3780, 1)


def test_legend_sits_outside_unless_placed(tmp_path):
    right = tmp_path / "right.svg"
    bottom = tmp_path / "bottom.svg"
    hidden = tmp_path / "none.svg"
    inside = tmp_path / "inside.svg"
    ggsave(right, _scatter(), width=500, height=320)
    ggsave(bottom, _scatter() + theme(legend_position="bottom"), width=500, height=320)
    ggsave(hidden, _scatter() + theme(legend_position="none"), width=500, height=320)
    ggsave(inside, _scatter() + theme(legend_position=(0.1, 0.9)), width=500, height=320)

    right_svg = right.read_text(encoding="utf-8")
    # The legend box is opaque. Point alpha is separate and still fades marks.
    boxes = re.findall(
        r'<rect [^>]*fill="#[0-9A-Fa-f]{6}"[^>]*stroke="#[0-9A-Fa-f]{6}"[^>]*/>',
        right_svg,
    )
    assert boxes
    assert all("fill-opacity" not in box for box in boxes)
    circles = _svg_circles(right_svg)
    labels = [item for item in _svg_texts(right_svg) if item[2] == "a"]
    assert labels
    assert min(item[0] for item in labels) > max(cx for cx, _cy in circles)

    bottom_svg = bottom.read_text(encoding="utf-8")
    axis = [item for item in _svg_texts(bottom_svg) if item[2] == "x"]
    key = [item for item in _svg_texts(bottom_svg) if item[2] == "a"]
    assert axis and key
    assert min(item[1] for item in key) > max(item[1] for item in axis)

    assert ">a</text>" not in hidden.read_text(encoding="utf-8")
    assert ">b</text>" not in hidden.read_text(encoding="utf-8")

    inside_svg = inside.read_text(encoding="utf-8")
    placed = [item for item in _svg_texts(inside_svg) if item[2] == "a"]
    data = _svg_circles(inside_svg)
    assert placed
    assert min(item[0] for item in placed) < max(cx for cx, _cy in data)
    assert min(item[1] for item in placed) < 160

    spec, _payloads = build_spec(_scatter())
    assert spec["legendPosition"] == "right"
    moved, _payloads = build_spec(_scatter() + theme(legend_position=(0.1, 0.9)))
    assert moved["legendPosition"] == [0.1, 0.9]
    dropped, _payloads = build_spec(_scatter() + theme(legend_position="none"))
    assert dropped["legendPosition"] == "none"


def test_legend_position_keeps_the_colour_theme(tmp_path):
    fig = _scatter() + theme_bw() + theme(legend_position="bottom")
    assert fig.theme_name == "bw"
    assert fig.legend_position == "bottom"
    path = tmp_path / "bw.svg"
    ggsave(path, fig, width=500, height=320)
    svg = path.read_text(encoding="utf-8")
    assert 'stroke="#333333"' in svg
    assert 'stroke="#2e3a5c"' not in svg
    key = [item for item in _svg_texts(svg) if item[2] == "a"]
    axis = [item for item in _svg_texts(svg) if item[2] == "x"]
    assert min(item[1] for item in key) > max(item[1] for item in axis)


def test_legend_position_rejects_unknown_values():
    with pytest.raises(ValueError, match="legend_position"):
        theme(legend_position="left")
    with pytest.raises(ValueError, match="0 to 1"):
        theme(legend_position=(1.2, 0.0))
    with pytest.raises(ValueError, match="legend_position"):
        theme(legend_position=0)


def test_static_labels_keep_symbols_and_function_parentheses(tmp_path):
    fig = (
        ggplot()
        + geom_function("y = sin(x)/x", xlim=(0.2, 8), n=12)
        + labs(x=r"$\Delta$ ($^\circ$C)", y=r"$\mathrm{mg}$")
    )
    path = tmp_path / "labels.svg"
    ggsave(path, fig, width=480, height=320)
    svg = path.read_text(encoding="utf-8")
    assert ">y = sin(x)/x</text>" in svg
    assert "sin x/x" not in svg
    assert ">Δ (°C)</text>" in svg
    assert ">mg</text>" in svg


def _point_in_poly(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i, (x1, y1) in enumerate(poly):
        x0, y0 = poly[j]
        if (y1 > py) != (y0 > py):
            cross = (x0 - x1) * (py - y1) / ((y0 - y1) or 1e-9) + x1
            if px < cross:
                inside = not inside
        j = i
    return inside


def _cube_hull(svg: str) -> list[tuple[float, float]]:
    """Convex hull of the long cube edges. Tick marks are the short lines."""
    points = []
    for x1, y1, x2, y2 in re.findall(
        r'<line x1="([^"]+)" y1="([^"]+)" x2="([^"]+)" y2="([^"]+)"',
        svg,
    ):
        a = (float(x1), float(y1))
        b = (float(x2), float(y2))
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 40:
            continue
        points.extend((a, b))
    ordered = sorted(set(points))
    if len(ordered) < 3:
        return ordered

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def test_3d_box_fills_the_panel_and_labels_sit_outside(tmp_path):
    # One colour, so no colour bar takes width from the cube.
    fig = ggplot() + geom_function("z = sin(x) cos(y)", n=8, colour="steelblue") + theme_bw()
    path = tmp_path / "surface.svg"
    ggsave(path, fig, width=400, height=400)
    svg = path.read_text(encoding="utf-8")
    hull = _cube_hull(svg)
    assert len(hull) >= 4
    xs = [point[0] for point in hull]
    ys = [point[1] for point in hull]
    # The old camera left the cube near 40% of the canvas. The long side
    # now fills the panel; the short side is this view's aspect.
    long_side, short_side = sorted(
        (max(xs) - min(xs), max(ys) - min(ys)), reverse=True,
    )
    assert long_side > 0.64 * 400
    assert short_side > 0.55 * 400
    labels = {text: (x, y) for x, y, text in _svg_texts(svg) if text in {"x", "y", "z"}}
    assert set(labels) == {"x", "y", "z"}
    for name, (lx, ly) in labels.items():
        assert not _point_in_poly(lx, ly, hull), name


def test_3d_export_draws_tick_numbers(tmp_path):
    fig = ggplot() + geom_function("z = sin(x) cos(y)", n=6)
    path = tmp_path / "surface.svg"
    ggsave(path, fig, width=480, height=360)
    svg = path.read_text(encoding="utf-8")
    # x and y both carry tick numbers; where their edges meet, the label that
    # would sit on the other one is left out instead of drawn on top of it.
    assert ">-10</text>" in svg and ">10</text>" in svg
    assert ">-0.5</text>" in svg
    assert ">0.5</text>" in svg
    assert ">x</text>" in svg
    assert ">y</text>" in svg
    assert ">z</text>" in svg


def test_clip_notes_are_printed_not_drawn(tmp_path, capsys):
    figure = ggplot() + geom_function("y = 1/x")
    path = tmp_path / "pole.svg"
    ggsave(str(path), figure)
    svg = path.read_text()
    assert "clipped" not in svg
    out = capsys.readouterr().out
    assert "y clipped to [" in out
    assert "notes=True" in out


def test_clip_notes_can_be_drawn_on_request(tmp_path):
    figure = ggplot() + geom_function("y = 1/x")
    path = tmp_path / "pole_notes.svg"
    ggsave(str(path), figure, notes=True)
    assert "y clipped to [" in path.read_text()


def _legend_labels(svg: str) -> list[str]:
    return re.findall(r">(a = [^<]*)<", svg)


def test_legend_keeps_every_entry_in_a_small_figure(tmp_path):
    # Four curves of one formula at journal column width: none may be dropped
    # or cut to "...".
    figure = ggplot() + theme_bw()
    for a, b, k in [(0.5, 0.5, 3.142), (2, 2, 0.1667), (2, 5, 0.03333), (5, 1, 0.2)]:
        figure = figure + geom_function(
            "y = x^(a-1) (1-x)^(b-1) / K", a=a, b=b, K=k, xlim=(0, 1)
        )
    path = tmp_path / "beta.svg"
    ggsave(str(path), figure, width=3.5, height=2.6, units="in")
    labels = _legend_labels(path.read_text())
    assert len(labels) == 4
    assert not any(label.endswith("...") for label in labels)


def test_bottom_legend_puts_short_entries_side_by_side(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "g": ["ctrl", "drug A", "drug B"]})
    figure = (
        ggplot(df, aes(x="x", y="y", colour="g"))
        + geom_point()
        + theme(legend_position="bottom")
    )
    path = tmp_path / "bottom.svg"
    ggsave(str(path), figure, width=3.5, height=2.6, units="in")
    svg = path.read_text()
    ys = {
        label: float(m.group(1))
        for label in ("ctrl", "drug A", "drug B")
        for m in [re.search(r'<text[^>]* y="([-\d.]+)"[^>]*>' + label + "<", svg)]
    }
    assert len(set(ys.values())) == 1  # one row


def test_real_font_metrics_are_used_for_svg_layout():
    from plot3 import static

    label = "a = 0.5, b = 0.5, K = 3.142"
    bitmap = static._text_width(label, 11)
    token = static._REAL_FONT.set(True)
    try:
        real = static._text_width(label, 11)
    finally:
        static._REAL_FONT.reset(token)
    # Helvetica is about half as wide as the 5x7 bitmap font.
    assert real < 0.6 * bitmap
    assert 120 <= real <= 145


def test_formula_surface_is_coloured_by_height_with_a_colour_bar(tmp_path):
    from plot3.build import build_spec

    fig = ggplot() + geom_function("t = sin(x) cos(y)", n=8)
    spec, _ = build_spec(fig)
    assert spec["color"]["kind"] == "num"
    assert spec["color"]["ramp"][0] == "#440154"  # viridis
    assert spec["labs"]["color"] == "t"
    path = tmp_path / "height.svg"
    ggsave(str(path), fig, width=480, height=360)
    svg = path.read_text(encoding="utf-8")
    assert ">t</text>" in svg  # colour bar title
    # A colour you choose turns the height colouring off.
    plain, _ = build_spec(ggplot() + geom_function("z = x y", colour="grey40"))
    assert plain["color"]["kind"] == "none"


def test_3d_surface_triangles_are_stroked_to_close_seams(tmp_path):
    fig = ggplot() + geom_function("z = x y", n=4, colour="#336699")
    path = tmp_path / "seams.svg"
    ggsave(str(path), fig, width=300, height=300)
    svg = path.read_text(encoding="utf-8")
    polygons = re.findall(r"<polygon[^>]*>", svg)
    assert polygons and all('stroke="' in poly for poly in polygons)


def test_3d_axis_title_stays_on_the_canvas(tmp_path):
    import re

    from plot3 import geom_function, ggplot, ggsave, labs
    from plot3.static import _real_width

    out = tmp_path / "surface.svg"
    ggsave(
        str(out),
        ggplot() + geom_function("z = sin(x) cos(y)", xlim=(-3, 3), ylim=(-3, 3))
        + labs(z="a long height title"),
        width=4, height=3.5, units="in",
    )
    svg = out.read_text()
    tag = re.search(r"<text[^>]*>a long height title</text>", svg).group(0)
    x = float(re.search(r'\bx="([-0-9.]+)"', tag).group(1))
    size = float(re.search(r'font-size="([0-9.]+)', tag).group(1))
    if 'text-anchor="end"' in tag:
        x -= _real_width("a long height title", size)
    assert x >= 0


def test_equal_aspect_ticks_span_the_widened_window(tmp_path):
    # coord_equal shows x from about -8 to 13 for y^2 = x^3 (data 0 to 4.6):
    # the ticks cover what is visible instead of bunching in 0..4.
    from plot3 import geom_function, ggplot, ggsave

    out = tmp_path / "cusp.svg"
    ggsave(str(out), ggplot() + geom_function("y^2 = x^3"), width=3, height=3, units="in")
    svg = out.read_text()
    assert ">10<" in svg and (">−5<" in svg or ">-5<" in svg)
    assert ">1<" not in svg and ">3<" not in svg
