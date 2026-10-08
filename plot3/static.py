"""Offline PNG, SVG, and PDF export.

PNG, SVG, and PDF replay one list of drawing commands. SVG is text.
PNG and PDF are that same SVG rendered by cairosvg when the optional
``plot3[export]`` extra is installed, so the three files share one
drawing and one font. That extra needs the Cairo C library as well as
the Python package (``libcairo2``, or the GTK runtime on Windows).
Without cairosvg, PNG falls back to a zlib RGB file and a built-in
5×7 font, and PDF raises with an install hint. ``.svg`` needs nothing
extra.
The geometry is the static channel already stored on each layer (the
last frame of a transition, the low end of a slider).
"""

from __future__ import annotations

import base64
import contextvars
import gzip
import io
import math
import re
import struct
import zlib
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from plot3.__version__ import __version__

_MAX_PX = 8192
_CSS_DPI = 96.0
_PAD = 0.03  # same normalized camera pad as the 2D viewer
_DEFAULT_FAMILY = "Helvetica, Arial, sans-serif"
_UNIT_INCH = {
    "in": 1.0,
    "inch": 1.0,
    "inches": 1.0,
    "cm": 1.0 / 2.54,
    "mm": 1.0 / 25.4,
}
_EXPORT_HINT = (
    "ggsave() needs the Cairo C library to write PDF with a journal font. "
    "Install the Python extra with: pip install 'plot3[export]'. "
    "Windows and minimal Linux images also need the Cairo library itself "
    "(for example the libcairo2 package, or the GTK runtime). "
    "ggsave('fig.svg', plot) writes the same drawing with no extra dependencies."
)

# Column bitmasks, LSB = top row. 5x7, authored for axis labels and titles.
def _glyph(*rows: str) -> tuple[int, ...]:
    cols = [0, 0, 0, 0, 0]
    for r, row in enumerate(rows):
        for c, ch in enumerate(row[:5]):
            if ch == "#":
                cols[c] |= 1 << r
    return tuple(cols)


_FONT: dict[str, tuple[int, ...]] = {
    " ": (0, 0, 0, 0, 0),
    "!": _glyph("..#..", "..#..", "..#..", "..#..", ".....", "..#..", "..#.."),
    '"': _glyph(".#.#.", ".#.#.", ".....", ".....", ".....", ".....", "....."),
    "#": _glyph(".#.#.", ".#.#.", "#####", ".#.#.", "#####", ".#.#.", ".#.#."),
    "$": _glyph("..#..", ".####", "#.#..", ".###.", "..#.#", "####.", "..#.."),
    "%": _glyph("#...#", "#..#.", "...#.", "..#..", ".#...", ".#..#", "#...#"),
    "&": _glyph(".##..", "#..#.", ".#...", ".##..", "#.#.#", "#..#.", ".##.#"),
    "'": _glyph("..#..", "..#..", ".....", ".....", ".....", ".....", "....."),
    "(": _glyph("...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."),
    ")": _glyph(".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."),
    "*": _glyph(".....", ".#.#.", "..#..", "#####", "..#..", ".#.#.", "....."),
    "+": _glyph(".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    ",": _glyph(".....", ".....", ".....", ".....", "..#..", "..#..", ".#..."),
    "-": _glyph(".....", ".....", ".....", "#####", ".....", ".....", "....."),
    ".": _glyph(".....", ".....", ".....", ".....", ".....", "..#..", "..#.."),
    "/": _glyph("....#", "...#.", "...#.", "..#..", ".#...", ".#...", "#...."),
    "0": _glyph(".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": _glyph("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": _glyph(".###.", "#...#", "....#", "..##.", ".#...", "#....", "#####"),
    "3": _glyph(".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."),
    "4": _glyph("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": _glyph("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": _glyph(".##..", "#....", "#....", "####.", "#...#", "#...#", ".###."),
    "7": _glyph("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": _glyph(".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": _glyph(".###.", "#...#", "#...#", ".####", "....#", "....#", "..##."),
    ":": _glyph(".....", "..#..", "..#..", ".....", "..#..", "..#..", "....."),
    ";": _glyph(".....", "..#..", "..#..", ".....", "..#..", "..#..", ".#..."),
    "<": _glyph("...#.", "..#..", ".#...", "#....", ".#...", "..#..", "...#."),
    "=": _glyph(".....", ".....", "#####", ".....", "#####", ".....", "....."),
    ">": _glyph(".#...", "..#..", "...#.", "....#", "...#.", "..#..", ".#..."),
    "?": _glyph(".###.", "#...#", "....#", "..##.", "..#..", ".....", "..#.."),
    "@": _glyph(".###.", "#...#", "#.###", "#.#.#", "#.###", "#....", ".###."),
    "A": _glyph(".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": _glyph("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": _glyph(".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "D": _glyph("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": _glyph("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": _glyph("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": _glyph(".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."),
    "H": _glyph("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": _glyph(".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "J": _glyph("..###", "...#.", "...#.", "...#.", "#..#.", "#..#.", ".##.."),
    "K": _glyph("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": _glyph("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": _glyph("#...#", "##.##", "#.#.#", "#...#", "#...#", "#...#", "#...#"),
    "N": _glyph("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": _glyph(".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": _glyph("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": _glyph(".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": _glyph("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": _glyph(".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": _glyph("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": _glyph("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": _glyph("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": _glyph("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": _glyph("#...#", ".#.#.", "..#..", "..#..", "..#..", ".#.#.", "#...#"),
    "Y": _glyph("#...#", ".#.#.", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "Z": _glyph("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "[": _glyph(".###.", ".#...", ".#...", ".#...", ".#...", ".#...", ".###."),
    "\\": _glyph("#....", ".#...", ".#...", "..#..", "...#.", "...#.", "....#"),
    "]": _glyph(".###.", "...#.", "...#.", "...#.", "...#.", "...#.", ".###."),
    "^": _glyph("..#..", ".#.#.", "#...#", ".....", ".....", ".....", "....."),
    "_": _glyph(".....", ".....", ".....", ".....", ".....", ".....", "#####"),
    "`": _glyph(".#...", "..#..", ".....", ".....", ".....", ".....", "....."),
    "a": _glyph(".....", ".....", ".###.", "....#", ".####", "#...#", ".####"),
    "b": _glyph("#....", "#....", "####.", "#...#", "#...#", "#...#", "####."),
    "c": _glyph(".....", ".....", ".###.", "#....", "#....", "#....", ".###."),
    "d": _glyph("....#", "....#", ".####", "#...#", "#...#", "#...#", ".####"),
    "e": _glyph(".....", ".....", ".###.", "#...#", "#####", "#....", ".###."),
    "f": _glyph("..##.", ".#...", ".#...", "####.", ".#...", ".#...", ".#..."),
    "g": _glyph(".....", ".....", ".####", "#...#", ".####", "....#", ".###."),
    "h": _glyph("#....", "#....", "####.", "#...#", "#...#", "#...#", "#...#"),
    "i": _glyph("..#..", ".....", "..#..", "..#..", "..#..", "..#..", ".###."),
    "j": _glyph("...#.", ".....", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "k": _glyph("#....", "#....", "#..#.", "#.#..", "##...", "#.#..", "#..#."),
    "l": _glyph("..#..", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "m": _glyph(".....", ".....", "##.#.", "#.#.#", "#.#.#", "#...#", "#...#"),
    "n": _glyph(".....", ".....", "####.", "#...#", "#...#", "#...#", "#...#"),
    "o": _glyph(".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###."),
    "p": _glyph(".....", ".....", "####.", "#...#", "####.", "#....", "#...."),
    "q": _glyph(".....", ".....", ".####", "#...#", ".####", "....#", "....#"),
    "r": _glyph(".....", ".....", "#.##.", "##...", "#....", "#....", "#...."),
    "s": _glyph(".....", ".....", ".####", "#....", ".###.", "....#", "####."),
    "t": _glyph("..#..", "..#..", "####.", "..#..", "..#..", "..#..", "...##"),
    "u": _glyph(".....", ".....", "#...#", "#...#", "#...#", "#...#", ".####"),
    "v": _glyph(".....", ".....", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "w": _glyph(".....", ".....", "#...#", "#...#", "#.#.#", "##.##", "#...#"),
    "x": _glyph(".....", ".....", "#...#", ".#.#.", "..#..", ".#.#.", "#...#"),
    "y": _glyph(".....", ".....", "#...#", "#...#", ".####", "....#", ".###."),
    "z": _glyph(".....", ".....", "#####", "...#.", "..#..", ".#...", "#####"),
    "{": _glyph("...#.", "..#..", "..#..", "#....", "..#..", "..#..", "...#."),
    "|": _glyph("..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "}": _glyph(".#...", "..#..", "..#..", "....#", "..#..", "..#..", ".#..."),
    "~": _glyph(".....", ".....", ".#.#.", "#.#..", ".....", ".....", "....."),
    "π": _glyph(".....", "#####", ".#.#.", ".#.#.", ".#.#.", ".#.#.", ".#.#."),
    "α": _glyph(".....", ".....", ".###.", "#...#", ".###.", "#..#.", ".##.#"),
    "β": _glyph("###..", "#..#.", "#..#.", "###..", "#..#.", "#..#.", "###.."),
    "θ": _glyph("..#..", ".#.#.", "#...#", "#####", "#...#", ".#.#.", "..#.."),
    "μ": _glyph(".....", ".....", "#...#", "#...#", "#...#", "##..#", "#.##."),
    "σ": _glyph(".....", ".....", ".####", "#....", "#...#", "#...#", ".###."),
    "∫": _glyph("..##.", ".#...", ".#...", "..#..", "...#.", "...#.", ".##.."),
}

_FOLD = str.maketrans({
    "−": "-", "–": "-", "—": "-",
    "×": "x", "·": ".",
    "’": "'", "‘": "'",
    "“": '"', "”": '"',
    "\u00a0": " ",
})

_KINDS = frozenset({"point", "line", "col", "box", "area", "poly", "surface", "isosurface"})


def save_static(
    fig,
    path,
    *,
    width=None,
    height=None,
    units: str = "px",
    dpi: float | None = None,
    family: str | None = None,
    fontsize: float | None = None,
    notes: bool = False,
) -> str:
    """Write ``fig`` to a ``.png``, ``.svg``, or ``.pdf`` file.

    Notes such as ``y clipped to [...]`` are for whoever makes the plot,
    not for readers of a paper. They are printed, not drawn, unless
    ``notes=True``.

    Bare ``width`` and ``height`` are pixels. ``units="in"`` (also
    ``"cm"`` and ``"mm"``) with ``dpi`` (default 300) sets a physical
    page. Layout stays in CSS pixels (96 per inch) so type keeps its
    size, and cairosvg rasterizes that SVG at ``dpi``. PDF and a
    journal-font PNG need the Cairo C library (``pip install
    'plot3[export]'``). ``.svg`` needs nothing extra.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".png", ".svg", ".pdf"}:
        raise ValueError("save_static() writes .png, .svg, or .pdf")
    fig = _print_theme(fig)
    size = _figure_size(fig, width, height, units, dpi)
    layout_w, layout_h = size["layout"]
    base_pt = _resolve_base_pt(fig, fontsize)
    family_name = _resolve_family(fig, family)
    dropped: list[str] = []
    # Measure text the way it will be drawn. SVG and PDF always use real
    # fonts; a PNG does too when Cairo is available, else the bitmap font.
    real = suffix in {".svg", ".pdf"} or _load_cairosvg() is not None
    token = _REAL_FONT.set(real)
    try:
        commands = _figure_commands(
            fig, layout_w, layout_h, base_pt, notes=notes, dropped=dropped
        )
    finally:
        _REAL_FONT.reset(token)
    # Browsers fall back per character; Cairo (PNG, PDF) needs one font
    # that has the glyphs, chosen from what this machine has installed.
    browser_wide = f"{family_name}, {_WIDE_FALLBACKS}"
    svg = _svg_text(
        commands,
        layout_w,
        layout_h,
        svg_width=size["svg_width"],
        svg_height=size["svg_height"],
        family=family_name,
        wide=browser_wide if suffix == ".svg" else (_wide_font() or browser_wide),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fallback = False
    if suffix == ".svg":
        data = svg.encode("utf-8")
    elif suffix == ".pdf":
        data = _pdf_bytes(svg)
    else:
        data, fallback = _png_file_bytes(svg, commands, size)
    if suffix == ".pdf":
        data = _stamp_pdf(data, __version__)
    path.write_bytes(data)
    print(f"plot3: saved {path} ({len(data) // 1024} KB)")
    shown = list(dict.fromkeys(dropped))
    for note in shown:
        print(f"plot3: {note}")
    if shown:
        print("plot3: notes are not drawn in the file; ggsave(..., notes=True) adds them")
    if fallback:
        print(
            "plot3: PNG used the built-in font. "
            "pip install 'plot3[export]' for Helvetica."
        )
    return str(path)


def _print_theme(fig):
    """Saved files default to theme_bw (white page, grey grid); the
    interactive viewer keeps its own default. A theme you add always wins."""
    import copy

    from plot3.compose import Composition

    def lighten(plot):
        if getattr(plot, "theme_explicit", True) or getattr(plot, "_payload", None) is not None:
            return plot
        out = copy.copy(plot)
        out.theme_name = "bw"
        return out

    if isinstance(fig, Composition):
        def walk(node):
            if isinstance(node, Composition):
                clone = copy.copy(node)
                clone.items = [walk(item) for item in node.items]
                return clone
            return lighten(node)

        return walk(fig)
    return lighten(fig)


def _figure_size(fig, width, height, units, dpi) -> dict:
    unit = str(units or "px").strip().lower()
    if unit in {"px", "pixel", "pixels"}:
        if dpi is not None:
            raise ValueError(
                "ggsave() dpi applies when units is 'in', 'cm', or 'mm'. "
                "For a 7 by 4 inch figure: "
                "ggsave('fig.png', plot, width=7, height=4, units='in', dpi=300)"
            )
        w, h = _figure_pixels(fig, width, height)
        return {
            "layout": (w, h),
            "png": (w, h),
            "svg_width": str(w),
            "svg_height": str(h),
            "scale": (1.0, 1.0),
            "dpi": _CSS_DPI,
            "physical": False,
        }
    if unit not in _UNIT_INCH:
        raise ValueError("ggsave() units must be 'px', 'in', 'cm', or 'mm'")
    dpi_value = 300.0 if dpi is None else _plain_float(
        dpi,
        "dpi",
        "ggsave() dpi must be between 1 and 2400",
        minimum=1,
        maximum=2400,
        type_message="ggsave() dpi must be a number, for example dpi=300",
    )
    default_h = _css_px(getattr(fig, "height", None), 480)
    width_in = _as_inches(width, "width", unit, 800)
    height_in = _as_inches(height, "height", unit, default_h)
    layout = (_px_extent(width_in, _CSS_DPI, "width"), _px_extent(height_in, _CSS_DPI, "height"))
    png = (_px_extent(width_in, dpi_value, "width"), _px_extent(height_in, dpi_value, "height"))
    return {
        "layout": layout,
        "png": png,
        "svg_width": _inch_attr(width_in),
        "svg_height": _inch_attr(height_in),
        "scale": (png[0] / layout[0], png[1] / layout[1]),
        "dpi": dpi_value,
        "physical": True,
    }


def _figure_pixels(fig, width, height) -> tuple[int, int]:
    w = 800 if width is None else _pixels(width, "width")
    h = _css_px(getattr(fig, "height", None), 480) if height is None else _pixels(height, "height")
    return w, h


def _as_inches(value, name: str, unit: str, default_px: int) -> float:
    if value is None:
        return default_px / _CSS_DPI
    if isinstance(value, str):
        raise TypeError(
            f"ggsave() {name} is a number of {unit}. "
            f"For example {name}=7, units='in'."
        )
    number = _plain_float(
        value,
        name,
        f"ggsave() {name} must be greater than 0",
        minimum=1e-6,
        maximum=None,
        type_message=(
            f"ggsave() {name} is a number of {unit}. "
            f"For example {name}=7, units='in'."
        ),
    )
    return number * _UNIT_INCH[unit]


def _px_extent(inches: float, dpi: float, name: str) -> int:
    pixels = int(round(inches * dpi))
    if pixels < 1:
        raise ValueError(f"ggsave() {name} must be at least 1 pixel")
    if pixels > _MAX_PX:
        raise ValueError(
            f"ggsave() {name} is {pixels} pixels, and the maximum is {_MAX_PX}. "
            "Lower dpi or the size."
        )
    return pixels


def _inch_attr(inches: float) -> str:
    text = f"{inches:.6f}".rstrip("0").rstrip(".")
    return f"{text}in"


def _plain_float(value, name: str, bounds_message: str, *, minimum, maximum, type_message=None) -> float:
    if isinstance(value, bool):
        raise TypeError(type_message or bounds_message)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(type_message or bounds_message) from exc
    if not math.isfinite(number):
        raise ValueError(bounds_message)
    if minimum is not None and number < minimum:
        raise ValueError(bounds_message)
    if maximum is not None and number > maximum:
        raise ValueError(bounds_message)
    return number


def _resolve_family(fig, family) -> str:
    chosen = family if family not in (None, "") else getattr(fig, "theme_family", None)
    if chosen in (None, ""):
        return _DEFAULT_FAMILY
    return str(chosen)


def _resolve_base_pt(fig, fontsize) -> float | None:
    if fontsize is not None:
        return _font_points(fontsize, "fontsize")
    theme_size = getattr(fig, "theme_base_size", None)
    if theme_size is None:
        return None
    return _font_points(theme_size, "base_size")


def _font_points(value, name: str) -> float:
    number = _plain_float(
        value,
        name,
        f"ggsave() {name} must be between 1 and 96 points",
        minimum=1,
        maximum=96,
        type_message=f"ggsave() {name} must be a font size in points, for example {name}=11",
    )
    return number


def _load_cairosvg():
    try:
        import cairosvg
    except Exception:
        return None
    return cairosvg


def _cairo_bytes(method: str, svg: str, **kwargs) -> bytes | None:
    lib = _load_cairosvg()
    if lib is None:
        return None
    buf = io.BytesIO()
    try:
        getattr(lib, method)(bytestring=svg.encode("utf-8"), write_to=buf, **kwargs)
    except OSError:
        return None
    data = buf.getvalue()
    return data or None


def _pdf_bytes(svg: str) -> bytes:
    data = _cairo_bytes("svg2pdf", svg)
    if not data:
        raise RuntimeError(_EXPORT_HINT)
    return data


def _png_file_bytes(svg: str, commands, size: dict) -> tuple[bytes, bool]:
    dpi = size["dpi"]
    data = _cairo_bytes("svg2png", svg, dpi=dpi)
    if data:
        return _with_phys(data, dpi), False
    sx, sy = size["scale"]
    png_w, png_h = size["png"]
    if sx != 1.0 or sy != 1.0:
        commands = _scale_commands(commands, sx, sy)
    return _png_bytes(_raster(commands, png_w, png_h), dpi), True


def _scale_commands(commands, sx: float, sy: float) -> list:
    stroke = (sx + sy) / 2.0
    scaled = []
    for cmd in commands:
        op = cmd[0]
        if op == "rect":
            _op, x, y, w, h, fill, color, sw, alpha = cmd
            scaled.append((
                "rect", x * sx, y * sy, w * sx, h * sy, fill, color,
                (sw or 0) * stroke, alpha,
            ))
        elif op == "line":
            _op, x1, y1, x2, y2, color, sw, alpha = cmd
            scaled.append((
                "line", x1 * sx, y1 * sy, x2 * sx, y2 * sy, color,
                (sw or 0) * stroke, alpha,
            ))
        elif op == "polyline":
            _op, pts, color, sw, alpha = cmd
            scaled.append((
                "polyline", [(px * sx, py * sy) for px, py in pts], color,
                (sw or 0) * stroke, alpha,
            ))
        elif op == "polygon":
            _op, pts, fill, color, sw, alpha = cmd
            scaled.append((
                "polygon", [(px * sx, py * sy) for px, py in pts], fill, color,
                (sw or 0) * stroke, alpha,
            ))
        elif op == "polymask":
            _op, tris, color, alpha = cmd
            scaled.append((
                "polymask",
                [[(px * sx, py * sy) for px, py in tri] for tri in tris],
                color,
                alpha,
            ))
        elif op == "circle":
            _op, cx, cy, r, fill, color, sw, alpha = cmd
            scaled.append((
                "circle", cx * sx, cy * sy, r * stroke, fill, color,
                (sw or 0) * stroke, alpha,
            ))
        elif op == "text":
            _op, x, y, text, size, fill, anchor, baseline, rotate, weight = cmd
            scaled.append((
                "text", x * sx, y * sy, text, float(size) * stroke, fill,
                anchor, baseline, rotate, weight,
            ))
        elif op == "clip":
            _op, x, y, w, h = cmd
            scaled.append(("clip", x * sx, y * sy, w * sx, h * sy))
        else:
            scaled.append(cmd)
    return scaled


def _pixels(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"ggsave() {name} must be a pixel count, for example {name}=800")
    text = value
    if isinstance(text, str):
        text = text.strip().lower()
        if text.endswith("px"):
            text = text[:-2].strip()
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"ggsave() {name} must be a pixel count, for example {name}=800"
        ) from exc
    if not math.isfinite(number) or number < 1:
        raise ValueError(f"ggsave() {name} must be at least 1 pixel")
    if number > _MAX_PX:
        raise ValueError(f"ggsave() {name} must be at most {_MAX_PX} pixels")
    return int(round(number))


def _css_px(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().lower()
        if text.endswith("px"):
            text = text[:-2].strip()
        try:
            number = float(text)
        except ValueError:
            return default
    if not math.isfinite(number) or number < 1:
        return default
    return int(round(min(number, _MAX_PX)))


def _figure_commands(
    fig,
    width: int,
    height: int,
    base_pt: float | None = None,
    *,
    notes: bool = True,
    dropped: list[str] | None = None,
) -> list:
    from plot3.compose import Composition

    if isinstance(fig, Composition):
        return _composition_commands(fig, width, height, base_pt, notes=notes, dropped=dropped)
    panels, layout = _panels(fig)
    if not notes:
        kept = []
        for item in panels:
            if item is None:
                kept.append(None)
                continue
            spec, blobs = item
            if dropped is not None:
                dropped.extend(str(n) for n in spec.get("notes") or [])
            kept.append((dict(spec, notes=[]), blobs))
        panels = kept
    first = next(item for item in panels if item is not None)
    theme = first[0].get("theme") or {}
    surface = theme.get("surface") or "#0b1020"
    commands: list = [("rect", 0, 0, width, height, surface, None, 0, 1.0)]
    if layout is None:
        spec, blobs = panels[0]
        _draw_spec(spec, blobs, 0, 0, width, height, commands, border=False, base_pt=base_pt)
        return commands
    _draw_facets(panels, layout, first[0], theme, width, height, base_pt, commands)
    return commands


def _composition_commands(fig, width, height, base_pt, *, notes, dropped) -> list:
    """``p1 | p2`` and ``p1 / p2``: each plot in its rectangle, one type size."""
    from plot3.compose import _first_theme

    fig = fig.tagged()
    theme = _first_theme(fig)
    commands: list = [("rect", 0, 0, width, height, theme["surface"], None, 0, 1.0)]
    fonts = _font_sizes(width, base_pt)
    if base_pt is None:
        # The whole figure's type, so a narrow panel does not get smaller text.
        base_pt = fonts[0] / 0.8 * 72.0 / _CSS_DPI
    note = fig.annotation
    header = {
        key: str(getattr(note, key, "") or "")
        for key in ("title", "subtitle", "caption")
    }
    head, foot = _draw_header_footer({}, header, fonts, theme, 0, 0, width, height, commands)
    pad = 6.0
    for plot, x, y, w, h in fig.rects(pad, head + pad, width - 2 * pad, height - head - foot - 2 * pad):
        sub = _figure_commands(plot, w, h, base_pt, notes=notes, dropped=dropped)
        commands.extend(_offset_commands(sub[1:], x, y))  # sub[0] is its page fill
    return commands


def _offset_commands(commands, dx: float, dy: float) -> list:
    out = []
    for cmd in commands:
        op = cmd[0]
        if op in {"rect", "circle", "text", "clip"}:
            out.append((op, cmd[1] + dx, cmd[2] + dy, *cmd[3:]))
        elif op == "line":
            out.append((op, cmd[1] + dx, cmd[2] + dy, cmd[3] + dx, cmd[4] + dy, *cmd[5:]))
        elif op in {"polyline", "polygon"}:
            out.append((op, [(x + dx, y + dy) for x, y in cmd[1]], *cmd[2:]))
        elif op == "polymask":
            out.append((op, [[(x + dx, y + dy) for x, y in tri] for tri in cmd[1]], *cmd[2:]))
        else:
            out.append(cmd)
    return out


def _draw_facets(panels, layout, first_spec, theme, width, height, base_pt, commands) -> None:
    """Panels in a grid with ggplot2's furniture: title rows, strips, one
    x and one y title, and a single legend to the right."""
    ncol, nrow = layout["ncol"], layout["nrow"]
    fonts = _font_sizes(width, base_pt)
    tick = fonts[0]
    label = _static_label(first_spec)
    header = {k: _with_frame(str(v), label) for k, v in (layout.get("header") or {}).items()}
    for key in list(header):
        if "$" in header[key]:
            from plot3.mathtext import split_math

            header[key], _segments = split_math(header[key])
    head, foot = _draw_header_footer(
        {"themeOpts": first_spec.get("themeOpts")}, header, fonts, theme,
        0, 0, width, height, commands,
    )
    pad = 8.0
    strip_bg = theme.get("grid") or "#1c2742"
    ink2 = theme.get("ink2") or theme.get("ink") or "#ffffff"
    raw_labs = first_spec.get("labs") or {}
    x_title = _with_frame(raw_labs.get("x") or "", label)
    y_title = _with_frame(raw_labs.get("y") or "", label)

    # One legend for the figure, from the first panel (colour levels are
    # shared, so every panel has the same keys).
    legend_spec = dict(first_spec, legendPosition="right")
    metrics = _legend_metrics(
        legend_spec, theme, fonts, raw_labs.get("color") or "",
        max_box=(width * 0.3, height - head - foot),
    )
    legend_w = float(metrics["w"]) + 12.0 if metrics else 0.0

    col_strips = layout.get("col_strips")
    row_strips = layout.get("row_strips")
    wrap = layout.get("kind") != "grid"
    strip = tick + 10.0
    top_strip = strip if col_strips else 0.0
    side_strip = strip if row_strips else 0.0
    left_title = _line_height(tick) + 10.0 if y_title else 0.0
    bottom_title = _line_height(tick) + 10.0 if x_title else 0.0
    top = head + pad + top_strip
    grid_w = width - left_title - legend_w - side_strip - pad * (ncol + 1)
    grid_h = height - top - foot - bottom_title - pad * nrow
    cell_w, cell_h = grid_w / ncol, grid_h / nrow

    def origin(row, col):
        return left_title + pad + col * (cell_w + pad), top + row * (cell_h + pad)

    def strip_box(x, y, w, h, text, rotate=0):
        commands.append(("rect", x, y, w, h, strip_bg, None, 0, 1.0))
        commands.append(("text", x + w / 2, y + h / 2, text, tick, ink2, "middle", "middle", rotate, 600))

    if col_strips:
        for col, text in enumerate(col_strips):
            x, _y = origin(0, col)
            strip_box(x, top - top_strip, cell_w, top_strip - 2, text)
    if row_strips:
        sx = left_title + pad + ncol * (cell_w + pad) - pad + 2
        for row, text in enumerate(row_strips):
            _x, y = origin(row, 0)
            strip_box(sx, y, side_strip - 2, cell_h, text, rotate=90)
    for cell, item in zip(layout["cells"], panels):
        if item is None:
            continue
        spec, blobs = item
        x, y = origin(cell["row"], cell["col"])
        h = cell_h
        if wrap and cell.get("strip"):
            strip_box(x, y, cell_w, strip - 2, cell["strip"])
            y, h = y + strip, cell_h - strip
        _draw_spec(spec, blobs, x, y, cell_w, h, commands, border=False, base_pt=base_pt)
    grid_left = left_title + pad
    grid_right = left_title + pad + ncol * (cell_w + pad) - pad
    if x_title:
        commands.append((
            "text", (grid_left + grid_right) / 2, height - foot - bottom_title + 4,
            x_title, tick, ink2, "middle", "top", 0, 400,
        ))
    if y_title:
        commands.append((
            "text", 4 + _line_height(tick) / 2, top + grid_h / 2 + pad * (nrow - 1) / 2,
            y_title, tick, ink2, "middle", "middle", -90, 400,
        ))
    if metrics:
        lx = width - legend_w + 4
        _paint_legend(commands, (lx, top), metrics, theme, fonts)

def _panels(fig):
    """``(panels, layout)``: one panel and no layout, or facet cells.

    A facet_grid cell with no rows is ``None`` and draws nothing.
    """
    payload = getattr(fig, "_payload", None)
    if payload is not None:
        return [(payload["spec"], payload.get("blobs") or {})], None

    from plot3.build import build_spec, facet_cells

    if getattr(fig, "facet", None) is None:
        spec, pairs = build_spec(fig)
        return [(spec, dict(pairs))], None
    layout = facet_cells(fig)
    panels = []
    for cell in layout["cells"]:
        if cell["fig"] is None:
            panels.append(None)
            continue
        spec, pairs = build_spec(cell["fig"])
        panels.append((spec, dict(pairs)))
    return panels, layout


def _is_missing(level) -> bool:
    if level is None:
        return True
    try:
        if bool(np.isnan(level)):
            return True
    except (TypeError, ValueError):
        pass
    try:
        import pandas as pd

        return bool(pd.isna(level))
    except (TypeError, ValueError):
        return False


def _draw_spec(spec, blobs, x, y, w, h, commands, *, border: bool, base_pt: float | None = None) -> None:
    theme = spec.get("theme") or {}
    if border:
        commands.append((
            "rect", x, y, w, h, None, theme.get("axis") or "#2e3a5c", 1, 1.0,
        ))
    labs = _labs(spec)
    is3d = bool(spec.get("is3d"))
    fonts = _font_sizes(w, base_pt)
    if not is3d:
        spec = _fit_x_labels(spec, w, fonts[0])
    # Tag, title, subtitle above the panel and the caption below it take
    # their own rows; the panel layout then sees a smaller cell, no title.
    head, foot = _draw_header_footer(spec, labs, fonts, theme, x, y, w, h, commands)
    y, h = y + head, max(8.0, h - head - foot)
    labs = dict(labs, title="")
    position = _legend_position(spec)
    left = 16.0 if is3d else _y_gutter(spec, labs, fonts, w)
    # A right-hand key that cannot show its labels moves under the panel.
    # The spec keeps the user's legendPosition; only this painting changes.
    if position == "right" and _legend_prefers_bottom(spec, fonts, w, h, left):
        position = "bottom"
    max_w, max_h = _legend_budget(position, w, h, left)
    metrics = _legend_metrics(
        spec, theme, fonts, labs.get("color") or "", max_box=(max_w, max_h),
        columns=position == "bottom",
    )
    extra_right, extra_bottom = _legend_reserve(position, metrics, w, h, left)
    if is3d:
        _draw_3d(
            spec, blobs, x, y, w, h, commands, labs, theme, fonts,
            extra_right=extra_right, extra_bottom=extra_bottom,
        )
        box = _box_3d(x, y, w, h, labs, fonts, extra_right, extra_bottom)
    else:
        _draw_2d(
            spec, blobs, x, y, w, h, commands, labs, theme, fonts,
            extra_right=extra_right, extra_bottom=extra_bottom,
        )
        box = _box_2d(spec, x, y, w, h, labs, fonts, extra_right, extra_bottom)
    if metrics is not None and position != "none":
        origin = _legend_origin(position, box, metrics, x, y, w, h)
        _paint_legend(commands, origin, metrics, theme, fonts)
    note_lines = _note_lines(spec.get("notes") or [], fonts[2], w)
    if note_lines:
        line_h = _note_line_height(fonts[2])
        # One caption per line, stacked upward so a second curve does not
        # run off the right edge. The axis margin reserved this space.
        baseline = y + h - 4 - extra_bottom
        color = theme.get("muted") or "#898781"
        for index, line in enumerate(reversed(note_lines)):
            commands.append((
                "text", x + 12, baseline - index * line_h, line,
                fonts[2], color, "start", "alphabetic", 0, 400,
            ))


def _labs(spec) -> dict:
    label = _static_label(spec)
    raw = spec.get("labs") or {}
    keys = ("title", "x", "y", "z", "color", "subtitle", "caption", "tag")
    out = {key: _with_frame(raw.get(key) or "", label) for key in keys}
    if spec.get("facetChild"):
        # The facet layout draws one x and one y title for the whole figure.
        out["x"] = out["y"] = ""
    return out


def _fit_x_labels(spec, width: float, tick: float):
    """Turn or thin x labels that would overlap (30 categories at 3.5 in).

    Tries flat, then 45 degrees, then 90; if even upright labels collide,
    keeps every k-th. An explicit theme(axis_text_x_angle=) is left alone.
    """
    opts = dict(spec.get("themeOpts") or {})
    if "xAngle" in opts:
        return spec
    labels = [str(lab) for _v, lab in _ticks((spec.get("scales") or {}).get("x") or {})]
    if len(labels) < 2:
        return spec
    slot = 0.72 * float(width) / len(labels)  # panel is roughly 3/4 of the cell
    longest = max(_text_width(lab, tick) for lab in labels)
    line = 1.25 * tick
    if longest + 6 <= slot:
        return spec
    if line * 1.45 <= slot:
        opts["xAngle"] = 45.0
    else:
        opts["xAngle"] = 90.0
        is_cat = ((spec.get("scales") or {}).get("x") or {}).get("kind") == "cat"
        if line > slot and is_cat:
            opts["xThin"] = int(math.ceil(line / max(slot, 1e-6)))
    return dict(spec, themeOpts=opts)


def _thin_y_ticks(ticks, scale, height: float, tick: float):
    """Every k-th tick when labels would sit closer than 1.6 lines apart."""
    if len(ticks) < 3 or scale.get("kind") == "cat":
        return ticks
    span = abs(float(scale.get("hi", 1.0)) - float(scale.get("lo", 0.0))) or 1.0
    gap = abs(float(ticks[1][0]) - float(ticks[0][0])) / span * float(height)
    step = 1
    while gap * step < 1.6 * tick and step < len(ticks):
        step += 1
    return ticks[::step]


def _x_label_drop(spec, tick: float) -> float:
    """Extra height turned x labels need below the axis."""
    angle = float((spec.get("themeOpts") or {}).get("xAngle", 0.0))
    if angle <= 0:
        return 0.0
    labels = [str(lab) for _v, lab in _ticks((spec.get("scales") or {}).get("x") or {})]
    longest = max((_text_width(lab, tick) for lab in labels), default=0)
    rad = math.radians(angle)
    return max(0.0, longest * math.sin(rad) + tick * math.cos(rad) - _line_height(tick))


def _draw_header_footer(spec, labs, fonts, theme, x, y, w, h, commands) -> tuple[float, float]:
    """Tag, title, and subtitle rows on top; caption row at the bottom.

    Returns the heights used, so the panel can be laid out below them.
    """
    tick, title_size, note = fonts
    ink = theme.get("ink") or "#ffffff"
    ink2 = theme.get("ink2") or ink
    muted = theme.get("muted") or "#898781"
    opts = spec.get("themeOpts") or {}
    hjust = float(opts.get("titleHjust", 0.0))
    title, subtitle, tag = labs.get("title"), labs.get("subtitle"), labs.get("tag")
    caption = labs.get("caption")
    top = 6.0
    tag_w = 0.0
    if tag:
        commands.append(("text", x + 10, y + top, tag, title_size, ink, "start", "top", 0, 700))
        tag_w = _text_width(tag, title_size) + 10.0
    left, right = x + 12 + tag_w, x + w - 12
    anchor = "start" if hjust < 0.25 else "end" if hjust > 0.75 else "middle"
    def at(width_hint: float) -> float:
        if anchor == "start":
            return left
        if anchor == "end":
            return right
        return (left + right) / 2.0
    used = 0.0
    if title:
        commands.append(("text", at(0), y + top, title, title_size, ink, anchor, "top", 0, 600))
        used = top + title_size + 4
    if subtitle:
        sub_size = max(tick, title_size - 2)
        start = used if used else top
        commands.append(("text", at(0), y + start, subtitle, sub_size, ink2, anchor, "top", 0, 400))
        used = start + sub_size + 4
    if tag and not used:
        used = top + title_size + 4
    foot = 0.0
    if caption:
        size = max(8, note)
        commands.append((
            "text", x + w - 12, y + h - 6, caption, size, muted, "end", "alphabetic", 0, 400,
        ))
        foot = size + 8
    return (used + 2 if used else 0.0), foot


def _with_frame(text: str, label: str) -> str:
    if not text or "{frame_time}" not in text:
        return text
    return text.replace("{frame_time}", label)


def _fmt_param(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    magnitude = abs(number)
    if magnitude >= 1e6 or (magnitude > 0 and magnitude < 1e-3):
        return f"{number:.2e}"
    return f"{number:.2f}"


def _fmt_tick(value, integer: bool) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if integer:
        return str(int(round(number)))
    if number == 0:
        return "0"
    magnitude = abs(number)
    if magnitude >= 1e6 or magnitude < 1e-4:
        return f"{number:.3g}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _static_label(spec) -> str:
    slider = spec.get("slider") or None
    if slider and slider.get("params"):
        return ", ".join(
            f"{item['name']} = {_fmt_param(item['lo'])}" for item in slider["params"]
        )
    transition = spec.get("transition") or None
    if not transition:
        return ""
    params = transition.get("params") or []
    if params:
        return ", ".join(
            f"{item['name']} = {_fmt_param(item['hi'])}" for item in params
        )
    times = transition.get("times") or []
    if not times:
        return ""
    if transition.get("type") == "states":
        return str(times[-1])
    return _fmt_tick(times[-1], bool(transition.get("integer")))


def _font_sizes(width: float, base_pt: float | None = None) -> tuple[int, int, int]:
    """Tick, title, and note sizes in CSS pixels.

    ``base_pt`` is a ggplot2 base size in points. Axis text is 0.8× and
    the title is 1.2×. Without it, the size follows the panel width.
    """
    if base_pt is not None:
        px = float(base_pt) * _CSS_DPI / 72.0
        tick = max(6, int(round(px * 0.8)))
        title = max(8, int(round(px * 1.2)))
        note = max(6, int(round(px * 0.7)))
        return tick, title, note
    if width < 280:
        return 9, 11, 9
    if width < 480:
        return 11, 13, 10
    return 12, 14, 11


def _box_3d(x, y, w, h, labs, fonts, extra_right=0.0, extra_bottom=0.0):
    # The fit padding inside this box holds the axis names. The gutter
    # itself only keeps those names off the canvas edge.
    margin = 8
    top = margin + (fonts[1] + 8 if labs.get("title") else 0)
    right = margin + extra_right
    bottom = margin + extra_bottom
    return (
        x + margin,
        y + top,
        max(8, w - margin - right),
        max(8, h - top - bottom),
    )


def _box_2d(spec, x, y, w, h, labs, fonts, extra_right=0.0, extra_bottom=0.0):
    tick, title, note = fonts
    void = bool((spec.get("theme") or {}).get("void"))
    left = 10.0 if void else _y_gutter(spec, labs, fonts, w)
    x_name = labs.get("x") or ""
    bottom = 6 + _line_height(tick) + (4 + _line_height(tick) if x_name else 0) + 8
    bottom += _x_label_drop(spec, tick)
    if void:
        bottom = 10.0
    note_lines = _note_lines(spec.get("notes") or [], note, w)
    if note_lines:
        bottom += len(note_lines) * _note_line_height(note) + 4
    bottom += extra_bottom
    top = 10 + (title + 6 if labs.get("title") else 0)
    right = 14 + extra_right
    # A long legend must not squeeze the panel down to a sliver.
    min_panel = max(48.0, min(w * 0.42, w - left - 8.0))
    if w - left - right < min_panel:
        right = max(8.0, w - left - min_panel)
    # A bottom legend needs more than the usual axis margin.
    bottom_cap = 0.75 if extra_bottom else 0.38
    bottom = min(bottom, h * bottom_cap)
    top = min(top, h * 0.32)
    return (
        x + left,
        y + top,
        max(8, w - left - right),
        max(8, h - top - bottom),
    )


def _draw_2d(
    spec, blobs, x, y, w, h, commands, labs, theme, fonts,
    extra_right=0.0, extra_bottom=0.0,
) -> None:
    box = _box_2d(spec, x, y, w, h, labs, fonts, extra_right, extra_bottom)
    scales = spec.get("scales") or {}
    window = _view_window(spec, box[2], box[3])
    surface = theme.get("surface") or "#0b1020"
    grid = theme.get("grid") or "#1c2742"
    axis = theme.get("axis") or "#2e3a5c"
    muted = theme.get("muted") or "#898781"
    ink2 = theme.get("ink2") or "#c3c2b7"
    ink = theme.get("ink") or "#ffffff"
    frame = str(theme.get("frame") or "box")
    tick_size, title_size, _note = fonts

    def px(u, v):
        left, right, bottom, top = window
        sx = box[0] + (u - left) / (right - left) * box[2]
        sy = box[1] + (top - v) / (top - bottom) * box[3]
        return sx, sy

    # A grid painted in the page colour is invisible, and at print
    # resolution the antialiased edge still shows. Skip it.
    opts = spec.get("themeOpts") or {}
    void = bool(theme.get("void"))
    if not void and opts.get("panelGrid", True) and _rgb(grid) != _rgb(surface):
        for value, _lab in _ticks(scales.get("x") or {}):
            u = _unit(scales.get("x") or {}, value)
            if u < window[0] - 0.02 or u > window[1] + 0.02:
                continue
            x0, _y0 = px(u, 0)
            commands.append(("line", x0, box[1], x0, box[1] + box[3], grid, 1, 1.0))
        for value, _lab in _ticks(scales.get("y") or {}):
            v = _unit(scales.get("y") or {}, value)
            if v < window[2] - 0.02 or v > window[3] + 0.02:
                continue
            _x0, y0 = px(0, v)
            commands.append(("line", box[0], y0, box[0] + box[2], y0, grid, 1, 1.0))

    gz = bool(spec.get("gz"))
    # Marks stay inside the panel, as ggplot2 clips them (coord_cartesian
    # zooms past the data; a big point at the edge is cut, not spilled).
    commands.append(("clip", box[0], box[1], box[2], box[3]))
    for layer in spec.get("layers") or []:
        _draw_layer_2d(layer, spec, blobs, gz, px, commands)
    _draw_rugs(spec, commands, box, px, window)
    commands.append(("unclip",))

    _draw_frame(commands, box, axis, frame)
    for value, lab in (() if void else _ticks(scales.get("x") or {})):
        u = _unit(scales.get("x") or {}, value)
        if u < window[0] - 0.02 or u > window[1] + 0.02:
            continue
        sx, _sy = px(u, 0)
        if sx < box[0] - 1 or sx > box[0] + box[2] + 1:
            continue
        angle = float(opts.get("xAngle", 0.0))
        thin = int(opts.get("xThin", 1) or 1)
        if thin > 1 and int(round(float(value))) % thin:
            continue
        if angle > 0:
            # Turned labels end at their tick, as ggplot2's hjust = 1.
            commands.append((
                "text", sx, box[1] + box[3] + 6, str(lab), tick_size, muted,
                "end", "middle" if angle >= 60 else "top", -angle, 400,
            ))
        else:
            commands.append((
                "text", sx, box[1] + box[3] + 4, str(lab), tick_size, muted,
                "middle", "top", 0, 400,
            ))
    y_ticks = [] if void else _thin_y_ticks(_ticks(scales.get("y") or {}), scales.get("y") or {}, box[3], tick_size)
    for value, lab in y_ticks:
        v = _unit(scales.get("y") or {}, value)
        if v < window[2] - 0.02 or v > window[3] + 0.02:
            continue
        _sx, sy = px(0, v)
        if sy < box[1] - 1 or sy > box[1] + box[3] + 1:
            continue
        commands.append((
            "text", box[0] - 6, sy, str(lab), tick_size, muted,
            "end", "middle", 0, 400,
        ))
    if labs.get("x") and not void:
        commands.append((
            "text", box[0] + box[2] / 2,
            box[1] + box[3] + 6 + _line_height(tick_size) + _x_label_drop(spec, tick_size),
            labs["x"], tick_size, ink2, "middle", "top", 0, 400,
        ))
    if labs.get("y") and not void:
        commands.append((
            "text", x + 8 + _line_height(tick_size) / 2, box[1] + box[3] / 2,
            labs["y"], tick_size, ink2, "middle", "middle", -90, 400,
        ))
    _draw_refs(spec, commands, window, px)
    _draw_arrows(spec, commands, px)
    placed: list[tuple[float, float, float, float]] = []
    text_boxes: list[tuple[float, float, float, float]] = []
    for ann in spec.get("ann") or []:
        text = str(ann.get("text") or "")
        if not text:
            continue
        try:
            u = _unit(scales.get("x") or {}, float(ann["x"]))
            v = _unit(scales.get("y") or {}, float(ann["y"]))
        except (TypeError, ValueError, KeyError):
            continue
        if u < window[0] - 0.02 or u > window[1] + 0.02:
            continue
        if v < window[2] - 0.02 or v > window[3] + 0.02:
            continue
        sx, sy = px(u, v)
        if ann.get("style") in {"text", "label"}:
            # ggplot2: geom_text's default 3.88 mm equals the base font. Keep
            # that ratio to this figure's base (axis text is 0.8 of it).
            scale = (tick_size / 0.8) / (3.88 * 96.0 / 25.4)
            _draw_text_ann(commands, ann, text, sx, sy, surface, text_boxes, scale)
            continue
        size = tick_size
        half_w = _text_width(text, size) / 2.0 + 3.0
        half_h = 0.65 * size + 1.0
        # Keep the label inside the panel. A thin tail area puts its centroid
        # on the axis; lift it clear.
        sx = min(max(sx, box[0] + half_w + 2.0), box[0] + box[2] - half_w - 2.0)
        sy = min(max(sy, box[1] + half_h + 2.0), box[1] + box[3] - half_h - 4.0)
        for _try in range(len(placed) + 1):
            hit = any(
                abs(sx - px0) < half_w + pw and abs(sy - py0) < half_h + ph
                for px0, py0, pw, ph in placed
            )
            if not hit:
                break
            sy -= 2.0 * half_h + 2.0
        placed.append((sx, sy, half_w, half_h))
        # A backing in the page colour keeps the label readable where it
        # crosses the curve or the fill.
        commands.append((
            "rect", sx - half_w, sy - half_h, 2.0 * half_w, 2.0 * half_h,
            surface, None, 0, 0.82,
        ))
        commands.append((
            "text", sx, sy, text, size, ink,
            "middle", "middle", 0, 600,
        ))
    if labs.get("title"):
        commands.append((
            "text", x + 12, y + 6, labs["title"], title_size, ink,
            "start", "top", 0, 600,
        ))


def _draw_frame(commands, box, axis, frame: str) -> None:
    x, y, w, h = box
    if frame == "none" or w <= 0 or h <= 0:
        return
    if frame == "axes":
        commands.append(("line", x, y, x, y + h, axis, 1, 1.0))
        commands.append(("line", x, y + h, x + w, y + h, axis, 1, 1.0))
        return
    commands.append(("rect", x, y, w, h, None, axis, 1, 1.0))


def _draw_layer_2d(layer, spec, blobs, gz, px, commands) -> None:
    kind = layer.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"ggsave() cannot draw a {kind!r} layer")
    if kind in {"surface", "isosurface"}:
        raise ValueError(f"ggsave() cannot draw a {kind!r} layer on a 2D figure")
    n = int(layer.get("n") or 0)
    if n <= 0:
        return
    colors = _layer_colors(layer, spec, blobs, gz, n)
    alpha = float(layer.get("alpha") if layer.get("alpha") is not None else 1.0)
    xs = _norm_channel(layer, "x", blobs, gz, n)
    ys = _norm_channel(layer, "y", blobs, gz, n)
    if kind == "point":
        radii = _point_radii(layer, blobs, gz, n, scene=False, min_dim=1.0)
        shapes = _point_shapes(layer, blobs, gz, n)
        for i in range(n):
            cx, cy = px(float(xs[i]), float(ys[i]))
            _marker(commands, cx, cy, float(radii[i]), shapes[i], _hex(colors[i]), alpha)
        return
    if kind == "col":
        hw = float(layer.get("width") or 0.08) * 0.5
        y0 = float(layer["y0"]) if layer.get("y0") is not None else 0.0
        for i in range(n):
            x0, y_top = px(float(xs[i]) - hw, max(float(ys[i]), y0))
            x1, y_bot = px(float(xs[i]) + hw, min(float(ys[i]), y0))
            commands.append((
                "rect", min(x0, x1), min(y_top, y_bot), abs(x1 - x0), abs(y_bot - y_top),
                _hex(colors[i]), None, 0, alpha,
            ))
        return
    if kind == "box":
        _draw_boxes(layer, blobs, gz, n, colors, alpha, px, commands, spec)
        return
    if kind == "area":
        y0 = float(layer["y0"]) if layer.get("y0") is not None else 0.0
        width = float(layer.get("linewidth") or 1.5)
        for start, count in _groups(layer, n):
            if count < 2:
                continue
            color = _hex(colors[start])
            curve = [px(float(xs[i]), float(ys[i])) for i in range(start, start + count)]
            base_l = px(float(xs[start]), y0)
            base_r = px(float(xs[start + count - 1]), y0)
            commands.append(("polygon", curve + [base_r, base_l], color, None, 0, alpha))
            commands.append(("polyline", curve, color, width, min(1.0, alpha + 0.3)))
        return
    if kind == "poly":
        width = float(layer.get("linewidth") or 1.5)
        for start, count in _groups(layer, n):
            if count < 3:
                continue
            color = _hex(colors[start])
            tris = _poly_triangles(xs, ys, start, count, px)
            if tris:
                # One mask, so shared edges do not darken the fill.
                commands.append(("polymask", tris, color, alpha))
            curve = [px(float(xs[i]), float(ys[i])) for i in range(start, start + count)]
            commands.append(("polyline", curve, color, width, min(1.0, alpha + 0.3)))
        return
    width = float(layer.get("linewidth") or 2.0)
    dashes = layer.get("dashes")
    for index, (start, count) in enumerate(_groups(layer, n)):
        if count < 2:
            continue
        curve = [px(float(xs[i]), float(ys[i])) for i in range(start, start + count)]
        dash = dashes[index] if dashes and index < len(dashes) else layer.get("dash")
        color = _hex(colors[start])
        if not dash:
            commands.append(("polyline", curve, color, width, alpha))
            continue
        for a, b in _dash_path(curve, dash, width):
            commands.append(("line", a[0], a[1], b[0], b[1], color, width, alpha))


def _poly_triangles(xs, ys, start, count, px):
    tris = []
    half = count // 2
    if half >= 2 and half * 2 == count:
        for i in range(half - 1):
            l0, l1 = start + i, start + i + 1
            r0, r1 = start + count - 1 - i, start + count - 2 - i
            for tri in ((l0, r0, l1), (r0, r1, l1)):
                tris.append([px(float(xs[k]), float(ys[k])) for k in tri])
        return tris
    for i in range(1, count - 1):
        tris.append([
            px(float(xs[k]), float(ys[k])) for k in (start, start + i, start + i + 1)
        ])
    return tris


def _draw_boxes(layer, blobs, gz, n, colors, alpha, px, commands, spec) -> None:
    hw = float(layer.get("width") or 0.08) * 0.5
    cap = hw * 0.55
    ink = _hex(_rgb((spec.get("theme") or {}).get("ink") or "#ffffff"))
    ymin = _norm_channel(layer, "ymin", blobs, gz, n)
    lower = _norm_channel(layer, "lower", blobs, gz, n)
    middle = _norm_channel(layer, "middle", blobs, gz, n)
    upper = _norm_channel(layer, "upper", blobs, gz, n)
    ymax = _norm_channel(layer, "ymax", blobs, gz, n)
    xs = _norm_channel(layer, "x", blobs, gz, n)
    fill_alpha = min(1.0, alpha * 0.35)
    # aes(fill=): a filled box with a dark outline, whiskers, and median.
    filled = bool(layer.get("fillMapped"))
    line_ink = _hex(_rgb((spec.get("theme") or {}).get("ink2") or "#333333"))
    for i in range(n):
        color = _hex(colors[i])
        x = float(xs[i])
        corners = [
            px(x - hw, float(lower[i])),
            px(x + hw, float(upper[i])),
        ]
        rect = (
            min(corners[0][0], corners[1][0]),
            min(corners[0][1], corners[1][1]),
            abs(corners[1][0] - corners[0][0]),
            abs(corners[1][1] - corners[0][1]),
        )
        stroke = line_ink if filled else color
        commands.append((
            "rect", rect[0], rect[1], rect[2], rect[3], color, color, 1,
            min(1.0, alpha * 0.9) if filled else fill_alpha,
        ))
        commands.append(("rect", rect[0], rect[1], rect[2], rect[3], None, stroke, 1, alpha))
        _seg(commands, px(x, float(ymin[i])), px(x, float(lower[i])), stroke, alpha)
        _seg(commands, px(x, float(upper[i])), px(x, float(ymax[i])), stroke, alpha)
        _seg(commands, px(x - cap, float(ymin[i])), px(x + cap, float(ymin[i])), stroke, alpha)
        _seg(commands, px(x - cap, float(ymax[i])), px(x + cap, float(ymax[i])), stroke, alpha)
        _seg(commands, px(x - hw, float(middle[i])), px(x + hw, float(middle[i])), line_ink if filled else ink, alpha)
    n_out = int(layer.get("nOut") or 0)
    if n_out <= 0 or layer.get("ox") is None:
        return
    ox = _norm_channel(layer, "ox", blobs, gz, n_out)
    oy = _norm_channel(layer, "oy", blobs, gz, n_out)
    radius = float(layer.get("outlierSize") or 3.0) / 2.0
    ocolors = _outlier_colors(layer, spec, blobs, gz, n_out, colors)
    for i in range(n_out):
        cx, cy = px(float(ox[i]), float(oy[i]))
        commands.append(("circle", cx, cy, radius, _hex(ocolors[i]), None, 0, alpha))


def _seg(commands, a, b, color, alpha) -> None:
    commands.append(("line", a[0], a[1], b[0], b[1], color, 1.25, alpha))


def _draw_3d(
    spec, blobs, x, y, w, h, commands, labs, theme, fonts,
    extra_right=0.0, extra_bottom=0.0,
) -> None:
    box = _box_3d(x, y, w, h, labs, fonts, extra_right, extra_bottom)
    scales = spec.get("scales") or {}
    spans = []
    for axis in ("x", "y", "z"):
        scale = scales.get(axis) or {}
        spans.append((float(scale.get("hi", 1.0)) - float(scale.get("lo", 0.0))) or 1.0)
    max_span = max(spans + [1e-12])
    coord = spec.get("coord") or {}
    aspect = coord.get("aspect") or "data"
    ext = [1.0, 1.0, 1.0] if aspect == "equal" else [s / max_span for s in spans]
    if coord.get("ext") and len(coord["ext"]) == 3:
        ext = [float(v) for v in coord["ext"]]
    # Same camera as the viewer. A uniform scale about the projected
    # centre fills the panel and leaves a centred mark where it was.
    void = bool(theme.get("void"))
    layout = _axis_layout(fonts[0])
    # No tick labels to make room for: the box fills the panel.
    inner = _inset_box(box, 8.0 if void else layout["pad"], top=8.0)
    camera = coord.get("camera") or {}
    zoom = float(camera.get("zoom") or 1.0)
    base = _projector(ext, box, camera.get("dir"), zoom)
    project, centre = _fit_projector(base, ext, inner, zoom)
    if not void and centre is not None and _labels_on_top(project, centre, ext, inner):
        # Perspective can put an axis along the top of the box: give its
        # labels the same room as the others.
        inner = _inset_box(box, layout["pad"])
        project, centre = _fit_projector(base, ext, inner, zoom)
    ink = theme.get("ink") or "#ffffff"
    gz = bool(spec.get("gz"))
    min_dim = min(box[2], box[3])

    if not void:
        _draw_back_panes(spec, ext, project, commands, theme)
    # A zoomed camera can put marks past the panel: keep them inside it.
    commands.append(("clip", box[0], box[1], box[2], box[3]))

    triangles = []
    lines = []
    points = []
    for layer in spec.get("layers") or []:
        kind = layer.get("kind")
        if kind not in _KINDS:
            raise ValueError(f"ggsave() cannot draw a {kind!r} layer")
        if kind in {"col", "box", "area", "poly"}:
            raise ValueError(f"ggsave() cannot draw a {kind!r} layer on a 3D figure")
        n = int(layer.get("n") or 0)
        if n <= 0:
            continue
        colors = _layer_colors(layer, spec, blobs, gz, n)
        alpha = float(layer.get("alpha") if layer.get("alpha") is not None else 1.0)
        xs = _norm_channel(layer, "x", blobs, gz, n)
        ys = _norm_channel(layer, "y", blobs, gz, n)
        zs = _norm_channel(layer, "z", blobs, gz, n)
        world = np.column_stack([
            xs * ext[0], ys * ext[1], zs * ext[2],
        ])
        if kind in {"surface", "isosurface"}:
            _append_surface(
                layer, blobs, gz, world, colors, alpha, theme, project, triangles, lines,
            )
        elif kind == "point":
            radii = _point_radii(layer, blobs, gz, n, scene=True, min_dim=min_dim * 0.6)
            mode = (spec.get("coord") or {}).get("sizeMode") or "scene"
            if mode == "screen":
                radii = np.maximum(
                    _raw_size_numbers(layer, blobs, gz, n) / 2.0, 0.75,
                )
            for i in range(n):
                hit = project(world[i])
                if hit is None:
                    continue
                points.append((hit[2], hit[0], hit[1], float(radii[i]), _hex(colors[i]), alpha))
        else:
            width = float(layer.get("linewidth") or 2.0)
            for start, count in _groups(layer, n):
                if count < 2:
                    continue
                projected = []
                for i in range(start, start + count):
                    hit = project(world[i])
                    if hit is None:
                        if len(projected) >= 2:
                            lines.append((projected, _hex(colors[start]), width, alpha))
                        projected = []
                        continue
                    projected.append((hit[0], hit[1]))
                if len(projected) >= 2:
                    lines.append((projected, _hex(colors[start]), width, alpha))

    # Far triangles first, then lines and points.
    triangles.sort(key=lambda item: -item[0])
    for _depth, poly, color, alpha in triangles:
        # Neighbouring triangles leave antialiased hairlines between them;
        # a stroke in the fill colour closes them on an opaque surface.
        if alpha >= 0.9:
            commands.append(("polygon", poly, color, color, 0.6, alpha))
        else:
            commands.append(("polygon", poly, color, None, 0, alpha))
    for projected, color, width, alpha in lines:
        commands.append(("polyline", projected, color, width, alpha))
    points.sort(key=lambda item: -item[0])
    for _depth, sx, sy, radius, color, alpha in points:
        if box[0] - radius <= sx <= box[0] + box[2] + radius and box[1] - radius <= sy <= box[1] + box[3] + radius:
            commands.append(("circle", sx, sy, radius, color, None, 0, alpha))
    commands.append(("unclip",))

    if not void:
        _draw_3d_axes(
            spec, ext, project, centre, commands, theme, fonts, labs, (x, y, w, h), layout,
        )
    if labs.get("title"):
        commands.append((
            "text", x + 12, y + 6, labs["title"], fonts[1], ink,
            "start", "top", 0, 600,
        ))


_CUBE_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _back_sides(corners) -> tuple[int, int, int] | None:
    """For x, y, z: which side (0 = min, 1 = max) faces away from the eye."""
    if any(c is None for c in corners):
        return None
    sides = []
    for bit in range(3):
        near = [c[2] for i, c in enumerate(corners) if not (i >> bit) & 1]
        far = [c[2] for i, c in enumerate(corners) if (i >> bit) & 1]
        sides.append(0 if sum(near) > sum(far) else 1)
    return tuple(sides)


def _labels_on_top(project, centre, ext, inner) -> bool:
    """Whether an axis is labelled along an edge near the top of the panel."""
    corners = _cube_corners(project, ext)
    if any(c is None for c in corners):
        return False
    hull = _hull_edge_set(corners)
    top = inner[1] + 0.2 * inner[3]
    for axis in ("x", "y", "z"):
        pair = _visible_axis_edge(axis, hull, corners, centre)
        a, b = corners[pair[0]], corners[pair[1]]
        if axis != "z" and min(a[1], b[1]) < top and _outward_normal(a, b, centre)[1] < 0:
            return True
    return False


def _draw_back_panes(spec, ext, project, commands, theme) -> None:
    """The three far walls with grid lines, as matplotlib and plotly draw a
    3D box. The three edges at the near corner are left out, so they never
    cross the data."""
    corners = _cube_corners(project, ext)
    back = _back_sides(corners)
    if back is None:
        return
    surface = theme.get("surface") or "#0b1020"
    grid = theme.get("grid") or "#1c2742"
    axis_color = theme.get("axis") or "#2e3a5c"
    scales = spec.get("scales") or {}
    opts = spec.get("themeOpts") or {}
    if opts.get("panelGrid", True) and _rgb(grid) != _rgb(surface):
        for wall in range(3):
            fixed = float(back[wall]) * ext[wall]
            for axis in range(3):
                if axis == wall:
                    continue
                other = 3 - wall - axis
                scale = scales.get("xyz"[axis]) or {}
                for value, _lab in _ticks(scale):
                    try:
                        u = _unit(scale, value)
                    except (TypeError, ValueError):
                        continue
                    if not 0.001 < u < 0.999:
                        continue
                    a = np.zeros(3)
                    a[wall] = fixed
                    a[axis] = u * ext[axis]
                    b = a.copy()
                    b[other] = ext[other]
                    pa, pb = project(a), project(b)
                    if pa is None or pb is None:
                        continue
                    commands.append(("line", pa[0], pa[1], pb[0], pb[1], grid, 1, 1.0))
    if str(theme.get("frame") or "box") == "none":
        return
    for a, b in _CUBE_EDGES:
        bit = (a ^ b).bit_length() - 1
        on_back = any(
            ((a >> wall) & 1) == back[wall] for wall in range(3) if wall != bit
        )
        if not on_back:
            continue
        pa, pb = corners[a], corners[b]
        commands.append(("line", pa[0], pa[1], pb[0], pb[1], axis_color, 1, 1.0))


def _append_surface(layer, blobs, gz, world, colors, alpha, theme, project, triangles, lines) -> None:
    node = layer.get("indices")
    if not isinstance(node, dict) or "id" not in node:
        raise ValueError(f"ggsave() cannot draw a {layer.get('kind')!r} layer without triangles")
    flat = _decode(blobs[node["id"]], node.get("dtype") or "u32", gz)
    usable = flat.size - (flat.size % 3)
    if usable <= 0:
        return
    faces = flat[:usable].reshape(-1, 3)
    if layer.get("wireframe"):
        ink = (theme or {}).get("ink2") or "#c3c2b7"
        base = layer.get("constColor") or ink
        seen = set()
        for face in faces:
            ids = [int(v) for v in face]
            for i in range(3):
                a = ids[i]
                b = ids[(i + 1) % 3]
                if a < 0 or b < 0 or a >= len(world) or b >= len(world):
                    continue
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                pa = project(world[a])
                pb = project(world[b])
                if pa is None or pb is None:
                    continue
                lines.append(([(pa[0], pa[1]), (pb[0], pb[1])], base, 1.0, alpha))
        return
    light = np.array([1.2, 0.8, 1.5], dtype=np.float64)
    light /= np.linalg.norm(light)
    for face in faces:
        idx = [int(v) for v in face]
        if any(v < 0 or v >= len(world) for v in idx):
            continue
        tri = world[idx]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            continue
        hits = [project(tri[k]) for k in range(3)]
        if any(hit is None for hit in hits):
            continue
        shade = min(1.0, 0.55 + 0.7 * abs(float(np.dot(normal / length, light))))
        rgb = tuple(int(round(channel * shade)) for channel in _average(colors, idx))
        depth = sum(hit[2] for hit in hits) / 3.0
        triangles.append((depth, [(hit[0], hit[1]) for hit in hits], _hex(rgb), alpha))


def _y_gutter(spec, labs, fonts, width: float) -> float:
    """Pixels reserved on the left for the y ticks and the rotated y title."""
    tick = fonts[0]
    yticks = [str(lab) for _t, lab in _ticks(spec.get("scales", {}).get("y") or {})]
    tick_w = max((_text_width(lab, tick) for lab in yticks), default=0)
    y_name = labs.get("y") or ""
    name_w = _line_height(tick) if y_name else 0
    left = 8 + name_w + (6 if y_name else 0) + tick_w + 8
    return min(float(left), float(width) * 0.42)


def _legend_budget(position, width: float, height: float, left: float) -> tuple[float, float]:
    """Largest legend box that stays inside the figure and leaves a panel."""
    if position == "none":
        return 0.0, 0.0
    if position == "bottom":
        return max(32.0, width - 8.0), max(24.0, height * 0.34)
    if isinstance(position, tuple):
        return max(32.0, width * 0.62), max(24.0, height * 0.62)
    # The panel keeps about half the figure. The 14px base margin and a
    # small gap sit between the panel and the legend box.
    min_panel = max(72.0, width * 0.46)
    box_w = width - left - 14.0 - 8.0 - min_panel
    if box_w < 44.0:
        box_w = max(36.0, min(width * 0.38, width - 8.0))
    return min(max(28.0, box_w), width - 4.0), max(24.0, height - 4.0)


_WRAP_OPS = {"−", "+", "×", "·", "=", "/", "-", "–"}


def _wrap_pieces(text: str) -> list[str]:
    """Words, with a lone operator glued to the word that follows it."""
    words = [word for word in str(text).split(" ") if word]
    pieces: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word in _WRAP_OPS and index + 1 < len(words):
            pieces.append(word + " " + words[index + 1])
            index += 2
        else:
            pieces.append(word)
            index += 1
    return pieces


def _wrap_text(text: str, size: float, max_width: float) -> list[str]:
    """Break ``text`` on spaces. A token wider than the line is shortened, not split."""
    raw = str(text).strip()
    if not raw:
        return []
    limit = max(8.0, float(max_width))
    if _text_width(raw, size) <= limit:
        return [raw]
    lines: list[str] = []
    current = ""
    for piece in _wrap_pieces(raw):
        trial = piece if not current else current + " " + piece
        if _text_width(trial, size) <= limit:
            current = trial
            continue
        if current:
            lines.append(current)
            current = ""
        if _text_width(piece, size) <= limit:
            current = piece
        else:
            lines.append(_ellipsis(piece, size, limit))
    if current:
        lines.append(current)
    return lines or [raw]


def _ellipsis(text: str, size: float, limit: float) -> str:
    if _text_width(text, size) <= limit:
        return text
    piece = text
    while piece and _text_width(piece + "...", size) > limit:
        piece = piece[:-1]
    return (piece + "...") if piece else "..."


def _split_caption(label: str) -> tuple[str, str]:
    """Separate ``formula  (a = 2, b = 5)`` into the formula and the tail."""
    text = str(label).strip()
    if "  (" in text and text.endswith(")"):
        formula, _, rest = text.rpartition("  (")
        return formula.strip(), "(" + rest
    return text, ""


def _value_units(tail: str) -> list[str]:
    """``(a = 2, b = 5)`` becomes ``a = 2`` and ``b = 5``, each kept whole."""
    if not tail:
        return []
    body = tail[1:-1] if tail.startswith("(") and tail.endswith(")") else tail
    return [part.strip() for part in body.split(",") if part.strip()]


def _pack_units(units: list[str], size: float, limit: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for unit in units:
        piece = unit.strip()
        if not piece:
            continue
        if _text_width(piece, size) > limit:
            if current:
                lines.append(current)
                current = ""
            lines.append(_ellipsis(piece, size, limit))
            continue
        trial = piece if not current else current + " " + piece
        if _text_width(trial, size) <= limit:
            current = trial
        else:
            if current:
                lines.append(current)
            current = piece
    if current:
        lines.append(current)
    return lines


def _cut_lines(lines: list[str], count: int, size: float, limit: float) -> list[str]:
    if len(lines) <= count:
        return lines
    kept = lines[: max(0, count)]
    if not kept:
        return kept
    last = kept[-1]
    if _text_width(last + "...", size) <= limit:
        kept[-1] = last + "..."
    else:
        kept[-1] = _ellipsis(last, size, limit)
    return kept


def _formula_segments(text: str) -> list[str]:
    """Split juxtaposed factors ``f)(g`` so a wrap can fall between them."""
    if ")(" not in text:
        return [text]
    parts = text.split(")(")
    segments = []
    for index, part in enumerate(parts):
        if index == 0:
            segments.append(part + ")")
        elif index < len(parts) - 1:
            segments.append("(" + part + ")")
        else:
            segments.append("(" + part)
    return segments


def _wrap_formula(text: str, size: float, limit: float) -> list[str]:
    """Wrap a formula. A break between ``)(`` is preferred to a break before ``−``."""
    raw = str(text).strip()
    if not raw or _text_width(raw, size) <= limit:
        return [raw] if raw else []
    lines: list[str] = []
    current = ""
    for segment in _formula_segments(raw):
        trial = (current + segment) if current else segment
        if _text_width(trial, size) <= limit:
            current = trial
            continue
        wrapped = _wrap_text(segment, size, limit)
        if current:
            lines.append(current)
            current = ""
        if len(wrapped) <= 1:
            current = wrapped[0] if wrapped else ""
        else:
            lines.extend(wrapped[:-1])
            current = wrapped[-1]
    if current:
        lines.append(current)
    return lines or [raw]


def _legend_parts(label: str, size: float, width: float) -> tuple[list[str], list[str]]:
    """Formula lines and parameter lines. ``a = 2`` is never split across a line."""
    formula, tail = _split_caption(label)
    limit = max(8.0, float(width))
    if tail and _text_width(tail, size) <= limit:
        value_lines = [tail]
    else:
        units = _value_units(tail)
        shown = [
            unit + ("," if index < len(units) - 1 else "")
            for index, unit in enumerate(units)
        ]
        value_lines = _pack_units(shown, size, limit)
    formula_lines = _wrap_formula(formula, size, limit) if formula else []
    return formula_lines, value_lines


def _legend_lines(label: str, size: float, width: float, max_lines: int) -> list[str]:
    """Wrap a legend label. The parameter values survive when the box is short."""
    formula_lines, value_lines = _legend_parts(label, size, width)
    limit = max(8.0, float(width))
    if not value_lines:
        return _cut_lines(formula_lines, max_lines, size, limit)
    room = max_lines - len(value_lines)
    if room < 1:
        return _cut_lines(value_lines, max_lines, size, limit)
    return _cut_lines(formula_lines, room, size, limit) + value_lines


def _plan_entry_lines(labels: list[str], size: float, width: float, budget: int) -> list[list[str]]:
    """Share ``budget`` lines so every entry keeps its values."""
    if not labels:
        return []
    budget = max(len(labels), int(budget))
    limit = max(8.0, float(width))
    parts = [_legend_parts(label, size, width) for label in labels]
    full = [formula + values for formula, values in parts]
    if sum(len(lines) for lines in full) <= budget:
        return full
    value_count = sum(len(values) for _formula, values in parts)
    if value_count > budget:
        planned = [
            _cut_lines(values, 1, size, limit) or values[:1]
            for _formula, values in parts
        ]
        spare = budget - sum(len(lines) for lines in planned)
        for index, (_formula, values) in enumerate(parts):
            if spare <= 0:
                break
            longer = _cut_lines(values, len(planned[index]) + spare, size, limit)
            extra = len(longer) - len(planned[index])
            if extra > 0:
                planned[index] = longer
                spare -= extra
        return planned
    # A formula line for one curve and not the other is harder to read.
    formula_budget = budget - value_count
    cap = 0 if formula_budget < len(labels) else formula_budget // len(labels)
    return [
        (_cut_lines(formula, cap, size, limit) if cap else []) + values
        for formula, values in parts
    ]


def _legend_prefers_bottom(spec, fonts, width: float, height: float, left: float) -> bool:
    """True when a right-hand key would split a token or cover the panel."""
    entries = [str(entry.get("label") or "") for entry in (spec.get("legend") or [])]
    if not entries:
        return False
    max_w, _max_h = _legend_budget("right", width, height, left)
    label_w = max(16.0, max_w - 16.0 - 18.0)
    size = fonts[0]
    row_h = _line_height(size) + 4
    total = 0
    for label in entries:
        lines = _legend_lines(label, size, label_w, 100)
        total += max(1, len(lines))
        for line in lines:
            if line.endswith("...") and not label.endswith("..."):
                return True
            if line.strip() in _WRAP_OPS:
                return True
    return total * row_h > float(height) * 0.40


def _note_lines(notes, size: float, width: float) -> list[str]:
    lines: list[str] = []
    limit = max(40.0, float(width) - 24.0)
    for note in notes:
        text = str(note).strip()
        if text:
            lines.extend(_wrap_text(text, size, limit))
    return lines


def _legend_position(spec):
    raw = spec.get("legendPosition")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            return "right"
    if raw in {"right", "bottom", "none"}:
        return raw
    return "right"


def _legend_metrics(
    spec, theme, fonts, color_label: str = "", max_box=None, *, columns: bool = False
):
    """Rows and the legend box size, or None when there is nothing to draw.

    Labels wrap to ``max_box`` so the box cannot be wider than the figure.
    ``columns`` (a legend under the panel) puts short entries side by side.
    """
    entries = list(spec.get("legend") or [])
    color = spec.get("color") or {}
    size_legend = spec.get("sizeLegend")
    if color.get("guide") is False:
        # guides(colour="none"): the colours stay, the bar goes.
        color = {}
    if (
        not entries and color.get("kind") != "num" and not size_legend
        and not spec.get("shapeLegend") and not spec.get("linetypeLegend")
    ):
        return None
    ink = theme.get("ink") or "#ffffff"
    tick = fonts[0]
    row_h = _line_height(tick) + 4
    max_w = None if max_box is None else float(max_box[0])
    max_h = None if max_box is None else float(max_box[1])
    separate_keys = spec.get("shapeLegend") or spec.get("linetypeLegend")
    if columns and entries and not size_legend and not separate_keys and max_w is not None:
        grid = _legend_grid(entries, color_label, tick, row_h, max_w, ink)
        if grid is not None:
            return grid
    label_w = None if max_w is None else max(16.0, max_w - 16.0 - 18.0)
    title_w = None if max_w is None else max(16.0, max_w - 16.0)
    max_lines = 8
    if max_h is not None and row_h:
        max_lines = max(2, int((max_h - 8) / row_h))
    rows = []
    title = color_label or ""
    title_count = 0
    if title and (entries or color.get("kind") == "num"):
        parts = _legend_lines(title, tick, title_w or 10_000, max_lines) if title_w else [title]
        rows.append(("title", parts[0] if parts else title))
        for extra in parts[1:]:
            rows.append(("cont", extra))
        title_count = len(parts) if parts else 1
    labels = [str(entry.get("label") or "") for entry in entries]
    if max_h is not None and row_h:
        # Never drop an entry: a missing row hides a curve. The box may grow
        # past its budget by one line per entry; _legend_reserve makes room.
        needed = 8 + row_h * (title_count + len(labels))
        if needed > max_h:
            max_h = float(needed)
            max_lines = max(max_lines, title_count + len(labels))
    budget = max_lines - title_count if max_h is not None else max(max_lines, len(labels) or 1)
    plans = _plan_entry_lines(labels, tick, label_w or 10_000, budget)
    for entry, parts in zip(entries, plans):
        color_hex = entry.get("color") or ink
        if not parts:
            parts = [str(entry.get("label") or "")]
        rows.append(("swatch", parts[0], color_hex, entry.get("shape"), entry.get("dash")))
        for extra in parts[1:]:
            rows.append(("cont", extra))
    ramp = color.get("ramp") or []
    if not entries and color.get("kind") == "num" and ramp:
        rows.append(("ramp", ramp, color.get("lo"), color.get("hi")))
    bar_title = (spec.get("labs") or {}).get("colorBar")
    if entries and color.get("kind") == "num" and ramp and bar_title is not None:
        # Class entries and a colour bar together (boxes on a height cloud).
        if bar_title:
            rows.append(("size-title", str(bar_title)))
        rows.append(("ramp", ramp, color.get("lo"), color.get("hi")))
    rows.extend(_key_rows(spec, ink))
    if size_legend and size_legend.get("breaks"):
        rows.append(("size-title", str(size_legend.get("label") or "size")))
        for br in size_legend["breaks"]:
            rows.append(("bubble", str(br.get("label") or ""), float(br.get("t") or 0)))
    if not rows:
        return None
    text_w = 0
    for row in rows:
        if row[0] in {"title", "size-title"}:
            text_w = max(text_w, _text_width(row[1], tick))
        elif row[0] in {"swatch", "bubble", "cont"}:
            text_w = max(text_w, _text_width(row[1], tick) + 18)
        elif row[0] == "ramp":
            # A vertical colour bar (ggplot2's colourbar): bar, gap, labels.
            labels = [_ramp_label(v) for v in _ramp_ticks(row[2], row[3])]
            text_w = max(text_w, 10 + 6 + max(_text_width(t, max(9, tick - 1)) for t in labels))
    box_w = text_w + 16
    if max_w is not None:
        box_w = min(box_w, max_w)
    box_h = 8.0
    kept = []
    for row in rows:
        if row[0] == "ramp":
            step = _RAMP_HEIGHT + 6
        elif row[0] == "bubble":
            step = max(row_h, 8 + int(round(row[2] * 16)))
        else:
            step = row_h
        if max_h is not None and kept and box_h + step > max_h:
            break
        kept.append(row)
        box_h += step
    if max_h is not None:
        box_h = min(box_h, max_h)
    return {"rows": kept, "w": box_w, "h": box_h, "row_h": row_h}


def _legend_grid(entries, title: str, tick: float, row_h: float, max_w: float, ink: str):
    """Entries in columns under the panel, or None when two do not fit side by side.

    Every entry is kept: a bottom legend that drops rows hides a curve.
    """
    labels = [str(entry.get("label") or "") for entry in entries]
    swatch, gap, pad = 18.0, 14.0, 16.0
    col_w = max(_text_width(label, tick) for label in labels) + swatch
    ncols = int((max_w - pad + gap) // (col_w + gap))
    ncols = min(ncols, len(labels))
    if ncols < 2:
        return None
    nrows = -(-len(labels) // ncols)
    rows = []
    if title:
        rows.append(("title", title))
    cells = [
        ("swatch", label, entry.get("color") or ink, entry.get("shape"), entry.get("dash"))
        for label, entry in zip(labels, entries)
    ]
    body_w = ncols * col_w + (ncols - 1) * gap
    box_w = min(max_w, max(body_w, _text_width(title, tick) if title else 0.0) + pad)
    box_h = 8.0 + row_h * ((1 if title else 0) + nrows)
    return {
        "rows": rows,
        "cells": cells,
        "ncols": ncols,
        "col_w": col_w + gap,
        "w": box_w,
        "h": box_h,
        "row_h": row_h,
    }


def _key_rows(spec, ink: str) -> list:
    """Rows for shape and linetype legends that are not merged with colour."""
    rows = []
    for key, legend in (("shape", spec.get("shapeLegend")), ("dash", spec.get("linetypeLegend"))):
        if not legend:
            continue
        rows.append(("size-title", str(legend.get("label") or key)))
        for entry in legend.get("entries") or []:
            shape = entry.get("shape") if key == "shape" else None
            dash = entry.get("dash") if key == "dash" else None
            rows.append(("swatch", str(entry.get("label") or ""), ink, shape, dash))
    return rows


def _short_number(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.3g}".replace("-", "−")


def _ramp_label(value) -> str:
    """A colour-bar value written like the axis numbers."""
    from plot3.scales import fmt_num

    number = float(value)
    if number != 0 and (abs(number) >= 1e4 or abs(number) < 1e-3):
        return f"{number:.3g}"
    return fmt_num(float(f"{number:.4g}"))


def _legend_key(commands, x, y, row) -> None:
    """The key in front of a legend label: a square, a point shape, or a line."""
    color = row[2]
    shape = row[3] if len(row) > 3 else None
    dash = row[4] if len(row) > 4 else None
    if shape:
        _marker(commands, x + 4.5, y + 4.5, 4.0, shape, color, 1.0)
    elif dash is not None:
        a, b = (x - 1, y + 4.5), (x + 11, y + 4.5)
        if dash:
            for p0, p1 in _dash_segments(a, b, dash, 1.6):
                commands.append(("line", p0[0], p0[1], p1[0], p1[1], color, 1.6, 1.0))
        else:
            commands.append(("line", a[0], a[1], b[0], b[1], color, 1.6, 1.0))
    else:
        commands.append(("rect", x, y, 9, 9, color, None, 0, 1.0))


def _legend_reserve(position, metrics, width: float, height: float, left: float = 0.0) -> tuple[float, float]:
    if metrics is None or position in {"none"} or isinstance(position, tuple):
        return 0.0, 0.0
    if position == "bottom":
        # Usually a third of the figure; more only when every entry needs it.
        cap = max(height * 0.36, min(float(metrics["h"]) + 10.0, height * 0.55))
        return 0.0, min(float(metrics["h"]) + 10.0, cap)
    # Leave the panel at least ~46% of the figure, after the y-axis gutter.
    room = width - left - max(64.0, width * 0.46)
    return min(float(metrics["w"]) + 12.0, max(0.0, room)), 0.0


def _legend_origin(position, box, metrics, x, y, w, h) -> tuple[float, float]:
    box_w = min(float(metrics["w"]), max(1.0, w - 4.0))
    box_h = min(float(metrics["h"]), max(1.0, h - 4.0))
    metrics["w"] = box_w
    metrics["h"] = box_h
    if position == "bottom":
        lx = box[0] + max(0.0, (box[2] - box_w) / 2.0)
        ly = y + h - box_h - 6
    elif isinstance(position, tuple):
        px, py = position
        lx = box[0] + px * max(0.0, box[2] - box_w)
        ly = box[1] + (1.0 - py) * max(0.0, box[3] - box_h)
    else:
        lx = x + w - box_w - 6
        ly = box[1]
    # A box wider than the cell used to clamp to a negative x. Pin it to the cell.
    lx = min(max(lx, x), x + max(0.0, w - box_w))
    ly = min(max(ly, y), y + max(0.0, h - box_h))
    return lx, ly


def _paint_legend(commands, origin, metrics, theme, fonts) -> None:
    ink = theme.get("ink") or "#ffffff"
    ink2 = theme.get("ink2") or "#c3c2b7"
    surface = theme.get("surface") or "#0b1020"
    grid = theme.get("grid") or "#1c2742"
    if _rgb(grid) == _rgb(surface):
        grid = theme.get("muted") or "#898781"
    tick = fonts[0]
    row_h = metrics["row_h"]
    lx, ly = origin
    # Opaque, so a legend inside the panel does not fade the marks under it.
    commands.append((
        "rect", lx, ly, metrics["w"], metrics["h"], surface, grid, 1, 1.0,
    ))
    cursor = ly + 6
    for row in metrics["rows"]:
        if row[0] in {"title", "size-title"}:
            commands.append(("text", lx + 8, cursor, row[1], tick, ink, "start", "top", 0, 600))
            cursor += row_h
        elif row[0] == "swatch":
            _legend_key(commands, lx + 8, cursor + 2, row)
            commands.append(("text", lx + 22, cursor, row[1], tick, ink2, "start", "top", 0, 400))
            cursor += row_h
        elif row[0] == "cont":
            commands.append(("text", lx + 22, cursor, row[1], tick, ink2, "start", "top", 0, 400))
            cursor += row_h
        elif row[0] == "ramp":
            # Vertical colour bar: high at the top, as ggplot2 draws it, with
            # a few labelled values beside it (three significant figures).
            top, height = cursor + 2, float(_RAMP_HEIGHT)
            _draw_ramp_vertical(commands, lx + 8, top, 10, height, row[1])
            lo, hi = float(row[2]), float(row[3])
            for value in _ramp_ticks(lo, hi):
                frac = 0.0 if hi == lo else (value - lo) / (hi - lo)
                ty = top + height * (1.0 - frac)
                # ggplot2's colour bar: short white ticks inside the bar.
                commands.append(("line", lx + 8, ty, lx + 10.5, ty, "#ffffff", 1, 0.9))
                commands.append(("line", lx + 15.5, ty, lx + 18, ty, "#ffffff", 1, 0.9))
                commands.append((
                    "text", lx + 22, ty, _ramp_label(value), max(9, tick - 1), ink2,
                    "start", "middle", 0, 400,
                ))
            cursor += _RAMP_HEIGHT + 6
        else:
            diameter = max(4.0, row[2] * 16.0)
            commands.append((
                "circle", lx + 8 + diameter / 2, cursor + diameter / 2, diameter / 2,
                ink2, None, 0, 0.85,
            ))
            commands.append((
                "text", lx + 8 + diameter + 6, cursor, row[1], tick, ink2, "start", "top", 0, 400,
            ))
            cursor += max(row_h, diameter + 3)
    cells = metrics.get("cells") or []
    ncols = int(metrics.get("ncols") or 1)
    for index, cell in enumerate(cells):
        row, col = divmod(index, ncols)
        cx = lx + 8 + col * float(metrics["col_w"])
        cy = cursor + row * row_h
        _legend_key(commands, cx, cy + 2, cell)
        commands.append(("text", cx + 14, cy, cell[1], tick, ink2, "start", "top", 0, 400))


_RAMP_HEIGHT = 84


def _ramp_ticks(lo, hi) -> list[float]:
    """Nice values inside the bar, as ggplot2 labels it; the ends when
    fewer than two nice values fit."""
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    from plot3.scales import nice_ticks

    eps = 1e-9 * (hi - lo)
    inner = [float(t) for t in nice_ticks(lo, hi, 5) if lo - eps <= t <= hi + eps]
    if len(inner) > 5:
        inner = inner[::2]
    if len(inner) >= 2:
        return inner
    return [lo] + inner + [hi]


def _draw_ramp_vertical(commands, x, y, w, h, ramp) -> None:
    stops = ramp or ["#000000", "#ffffff"]
    steps = max(2, int(h))
    for i in range(steps):
        color = _hex(_ramp_at([_rgb(stop) for stop in stops], 1.0 - i / (steps - 1)))
        commands.append(("rect", x, y + i * h / steps, w, h / steps + 0.5, color, None, 0, 1.0))


def _draw_ramp(commands, x, y, w, h, ramp) -> None:
    stops = ramp or ["#000000", "#ffffff"]
    steps = max(2, int(w))
    for i in range(steps):
        color = _hex(_ramp_at([_rgb(stop) for stop in stops], i / (steps - 1)))
        commands.append(("rect", x + i * w / steps, y, w / steps + 0.5, h, color, None, 0, 1.0))


# Corner order matches _draw_3d: iz, then iy, then ix. The first edge of
# each axis is the min edge the viewer labels.
_AXIS_EDGES = {
    "x": ((0, 1), (2, 3), (4, 5), (6, 7)),
    "y": ((0, 2), (1, 3), (4, 6), (5, 7)),
    "z": ((0, 4), (1, 5), (2, 6), (3, 7)),
}


def _axis_layout(size: float) -> dict:
    """Pixels between the cube and the axis title, and the pad that reserves them."""
    line = float(_line_height(size))
    # Tick text starts just outside the edge. The title clears that whole line.
    tick_gap = 6.0
    title_gap = tick_gap + line + 8.0
    pad = title_gap + line + 4.0
    return {"line": line, "tick_gap": tick_gap, "title_gap": title_gap, "pad": pad}


def _inset_box(box, pad: float, top: float | None = None):
    """Room for axis labels. The camera never puts labels along the top of
    the cube, so that side keeps only a small margin and the cube grows."""
    x, y, w, h = box
    pad = min(max(0.0, float(pad)), w * 0.28, h * 0.28)
    top = pad if top is None else min(max(0.0, float(top)), pad)
    return (x + pad, y + top, max(4.0, w - 2.0 * pad), max(4.0, h - pad - top))


def _fit_projector(project, ext, inner, zoom: float = 1.0):
    """Scale and centre the projected cube's outline until it meets `inner`,
    then scale by ``zoom`` (coord_3d(zoom=2) shows the middle, twice as
    large). Perspective makes the outline lopsided about the cube's centre,
    so the outline, not the centre, is what gets centred."""
    corners = [
        hit for hit in (
            project(np.array([ix, iy, iz], dtype=np.float64))
            for iz in (0.0, float(ext[2]))
            for iy in (0.0, float(ext[1]))
            for ix in (0.0, float(ext[0]))
        ) if hit is not None
    ]
    if len(corners) < 2:
        return project, None
    xs = [hit[0] for hit in corners]
    ys = [hit[1] for hit in corners]
    bx, by = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    bw, bh = max(max(xs) - min(xs), 1e-9), max(max(ys) - min(ys), 1e-9)
    x0, y0, w, h = inner
    scale = min(w / bw, h / bh) * max(float(zoom), 1e-6)
    tx, ty = x0 + w / 2.0, y0 + h / 2.0

    def fitted(point):
        hit = project(point)
        if hit is None:
            return None
        return (tx + (hit[0] - bx) * scale, ty + (hit[1] - by) * scale, hit[2])

    centre_hit = fitted(np.asarray(ext, dtype=np.float64) / 2.0)
    return fitted, (None if centre_hit is None else (centre_hit[0], centre_hit[1]))


def _cube_corners(project, ext):
    return [
        project(np.array([ix, iy, iz], dtype=np.float64))
        for iz in (0.0, float(ext[2]))
        for iy in (0.0, float(ext[1]))
        for ix in (0.0, float(ext[0]))
    ]


def _hull_edge_set(points) -> set[tuple[int, int]]:
    """Index pairs of the convex hull. `points` may contain None."""
    indexed = [(i, p[0], p[1]) for i, p in enumerate(points) if p is not None]
    indexed.sort(key=lambda item: (item[1], item[2]))
    if len(indexed) < 3:
        return set()

    def cross(o, a, b):
        return (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])

    lower: list = []
    for point in indexed:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list = []
    for point in reversed(indexed):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = [point[0] for point in lower[:-1] + upper[:-1]]
    edges = set()
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        edges.add((a, b) if a < b else (b, a))
    return edges


def _corner_xyz(index: int, ext) -> np.ndarray:
    return np.array([
        (index & 1) * ext[0],
        ((index >> 1) & 1) * ext[1],
        ((index >> 2) & 1) * ext[2],
    ], dtype=np.float64)


def _visible_axis_edge(axis: str, hull, corners, centre):
    """A silhouette edge parallel to `axis`. Prefer the viewer's min edge."""
    pairs = _AXIS_EDGES[axis]
    best = None
    best_dist = -1.0
    for pair in pairs:
        key = pair if pair[0] < pair[1] else (pair[1], pair[0])
        if key not in hull:
            continue
        if pair == pairs[0]:
            return pair
        a = corners[pair[0]]
        b = corners[pair[1]]
        mx = (a[0] + b[0]) / 2.0 - centre[0]
        my = (a[1] + b[1]) / 2.0 - centre[1]
        dist = mx * mx + my * my
        if dist > best_dist:
            best_dist = dist
            best = pair
    return best if best is not None else pairs[0]


def _outward_normal(p0, p1, centre) -> tuple[float, float]:
    """Screen normal of an edge, pointing away from the cube centre."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0, 1.0
    nx, ny = -dy / length, dx / length
    mx = (p0[0] + p1[0]) / 2.0 - centre[0]
    my = (p0[1] + p1[1]) / 2.0 - centre[1]
    if nx * mx + ny * my < 0.0:
        nx, ny = -nx, -ny
    return nx, ny


def _outside_anchor(nx: float, ny: float) -> tuple[str, str]:
    """Anchor so the glyphs sit further outside the edge than the anchor point."""
    if abs(nx) >= abs(ny):
        return ("start" if nx > 0.0 else "end"), "middle"
    return "middle", ("top" if ny > 0.0 else "alphabetic")


def _label_box(x: float, y: float, text: str, size: float, anchor: str, baseline: str):
    width = float(_text_width(text, size)) + 2.0
    height = 1.2 * float(size)
    left = x - width if anchor == "end" else x - width / 2.0 if anchor == "middle" else x
    top = y if baseline == "top" else y - height if baseline == "alphabetic" else y - height / 2.0
    return (left, top, width, height)


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _clamp_point(px: float, py: float, bounds) -> tuple[float, float]:
    x, y, w, h = bounds
    return (
        min(max(px, x + 3.0), x + w - 3.0),
        min(max(py, y + 3.0), y + h - 3.0),
    )


def _thin_marks(marks, air: float) -> list:
    """Every k-th tick, the smallest k whose labels do not touch, keeping
    round values (0, 5, 10 rather than 1, 4, 7) when the ticks allow."""
    if len(marks) < 3:
        return marks

    def roomy(box):
        return (box[0] - air, box[1] - air, box[2] + 2 * air, box[3] + 2 * air)

    for k in range(1, len(marks)):
        offsets = list(range(k))
        try:
            step = abs(float(marks[1][0]) - float(marks[0][0])) * k
            offsets.sort(key=lambda o: 0 if abs(math.remainder(float(marks[o][0]), step)) < 1e-9 * max(step, 1e-300) else 1)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        for offset in offsets:
            kept = marks[offset::k]
            if all(not _boxes_overlap(roomy(a[5]), b[5]) for a, b in zip(kept, kept[1:])):
                return kept
    return marks[:1]


def _draw_3d_axes(spec, ext, project, centre, commands, theme, fonts, labs, bounds, layout) -> None:
    """Tick numbers and axis names just outside the three visible edges."""
    if centre is None:
        return
    muted = theme.get("muted") or "#898781"
    ink2 = theme.get("ink2") or "#c3c2b7"
    corners = _cube_corners(project, ext)
    hull = _hull_edge_set(corners)
    scales = spec.get("scales") or {}
    tick_gap = layout["tick_gap"]
    title_gap = layout["title_gap"]
    placed: list[tuple[float, float, float, float]] = []  # label boxes, all axes
    for axis in ("x", "y", "z"):
        pair = _visible_axis_edge(axis, hull, corners, centre)
        p0 = corners[pair[0]]
        p1 = corners[pair[1]]
        if p0 is None or p1 is None:
            continue
        nx, ny = _outward_normal(p0, p1, centre)
        anchor, baseline = _outside_anchor(nx, ny)
        scale = scales.get(axis) or {}
        marks = []
        for value, lab in _ticks(scale):
            try:
                u = _unit(scale, value)
            except (TypeError, ValueError):
                continue
            if u < -0.001 or u > 1.001:
                continue
            hit = project(_corner_xyz(pair[0], ext) * (1.0 - u) + _corner_xyz(pair[1], ext) * u)
            if hit is None:
                continue
            tx, ty = _clamp_point(hit[0] + nx * tick_gap, hit[1] + ny * tick_gap, bounds)
            box = _label_box(tx, ty, str(lab), fonts[0], anchor, baseline)
            marks.append((value, str(lab), hit, tx, ty, box))
        reach = max(3.0, tick_gap - 2.0)
        for value, lab, hit, tx, ty, label_box in _thin_marks(marks, 0.25 * float(fonts[0])):
            commands.append((
                "line", hit[0], hit[1], hit[0] + nx * reach, hit[1] + ny * reach,
                muted, 1, 1.0,
            ))
            # Where two axis edges meet ("3" ending x, "-3" starting y), the
            # second label would sit on the first: leave it out.
            if any(_boxes_overlap(label_box, other) for other in placed):
                continue
            placed.append(label_box)
            commands.append((
                "text", tx, ty, lab, fonts[0], muted,
                anchor, baseline, 0, 400,
            ))
        text = labs.get(axis) or ""
        if not text:
            continue
        mid = project(_corner_xyz(pair[0], ext) * 0.5 + _corner_xyz(pair[1], ext) * 0.5)
        if mid is None:
            continue
        lx, ly = _clamp_point(mid[0] + nx * title_gap, mid[1] + ny * title_gap, bounds)
        commands.append((
            "text", lx, ly, text, fonts[0], ink2, anchor, baseline, 0, 400,
        ))


def _projector(ext, box, direction=None, zoom: float = 1.0):
    """The 3D viewer's camera: same direction, distance, and perspective.

    The eye sits on the same side of the cube as the viewer (fov 60, up
    = +z), so near things are larger and parallel edges converge, as on
    screen. The cube's centre projects to the centre of ``box``, so fitting
    the panel does not slide a centred mark.
    """
    ctr = np.array([ext[0] / 2, ext[1] / 2, ext[2] / 2], dtype=np.float64)
    zoom = max(float(zoom), 1e-6)
    backward = np.array(direction if direction else [0.55, -0.85, 0.5], dtype=np.float64)
    backward /= max(float(np.linalg.norm(backward)), 1e-12)
    radius = max(float(np.linalg.norm(ext)) / 2.0, 1e-3)
    # The viewer's camera: fov 60, the same distance, the same zoom.
    dist = radius * 1.25 / math.tan(math.radians(30.0)) / zoom
    eye = ctr + backward * dist
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, backward)
    norm = float(np.linalg.norm(right))
    if norm < 1e-8:
        right = np.cross(np.array([0.0, 1.0, 0.0]), backward)
        norm = float(np.linalg.norm(right))
    right /= norm
    cam_up = np.cross(backward, right)
    samples = []
    for iz in (0.0, float(ext[2])):
        for iy in (0.0, float(ext[1])):
            for ix in (0.0, float(ext[0])):
                rel = np.array([ix, iy, iz], dtype=np.float64) - eye
                depth = max(-float(np.dot(rel, backward)), 1e-6)
                samples.append((float(np.dot(rel, right)) / depth, float(np.dot(rel, cam_up)) / depth))
    span = max(
        max(point[0] for point in samples) - min(point[0] for point in samples),
        max(point[1] for point in samples) - min(point[1] for point in samples),
        1e-6,
    )
    half = span / 2.0
    limit = min(box[2], box[3]) / 2.0

    def project(point):
        rel = np.asarray(point, dtype=np.float64) - eye
        cam_x = float(np.dot(rel, right))
        cam_y = float(np.dot(rel, cam_up))
        cam_z = float(np.dot(rel, backward))
        depth = -cam_z
        if depth < 1e-6:
            return None
        sx = box[0] + box[2] / 2.0 + (cam_x / depth / half) * limit
        sy = box[1] + box[3] / 2.0 - (cam_y / depth / half) * limit
        return sx, sy, depth

    return project


def _view_window(spec, plot_w, plot_h):
    coord = spec.get("coord") or {}
    if coord.get("expand") is False:
        # coord_cartesian(expand=False): the limits are the panel's edges.
        return (0.0, 1.0, 0.0, 1.0)
    if coord.get("aspect") != "equal":
        return (-_PAD, 1 + _PAD, -_PAD, 1 + _PAD)
    ratio = float(coord.get("ratio") or 1.0) or 1.0
    scales = spec.get("scales") or {}
    xspan = (float(scales.get("x", {}).get("hi", 1)) - float(scales.get("x", {}).get("lo", 0))) or 1.0
    yspan = (float(scales.get("y", {}).get("hi", 1)) - float(scales.get("y", {}).get("lo", 0))) or 1.0
    target = (plot_w / max(plot_h, 1.0)) * (yspan / xspan) / ratio
    need = 1 + 2 * _PAD
    nx = ny = need
    if nx / ny < target:
        nx = ny * target
    else:
        ny = nx / target
    return (0.5 - nx / 2, 0.5 + nx / 2, 0.5 - ny / 2, 0.5 + ny / 2)


def _ticks(scale: dict) -> list:
    if not scale:
        return []
    if scale.get("kind") == "cat":
        return [[i, str(cat)] for i, cat in enumerate(scale.get("cats") or [])]
    ticks = scale.get("ticks")
    if ticks:
        return ticks
    ladder = scale.get("ladder")
    if ladder and ladder[0]:
        return _date_ticks(scale, ladder)
    return []


def _date_ticks(scale: dict, ladder: list) -> list:
    """Pick a date ladder level the way the viewer does, sized for print.

    Levels run coarse to fine. Take the first that shows 3 to 8 ticks;
    otherwise thin the closest level so labels do not overlap.
    """
    lo = float(scale.get("lo", -math.inf))
    hi = float(scale.get("hi", math.inf))
    levels = [[t for t in level if lo <= float(t[0]) <= hi] for level in ladder]
    for visible in levels:
        if 3 <= len(visible) <= 8:
            return visible
    visible = levels[0] if len(levels[0]) >= 3 else levels[-1]
    if len(visible) <= 8:
        return visible
    step = -(-len(visible) // 7)
    return visible[::step]


def _draw_text_ann(commands, ann, text, sx, sy, surface, boxes, scale: float = 1.0) -> None:
    """geom_text / geom_label: anchored by hjust and vjust like ggplot2."""
    size = float(ann.get("size") or 14.7) * scale
    width = float(_text_width(text, size))
    height = 1.2 * size
    hj = float(ann.get("hjust", 0.5))
    vj = float(ann.get("vjust", 0.5))
    left = sx - hj * width
    top = sy - (1.0 - vj) * height
    if not ann.get("overlap", True):
        # check_overlap=True: skip a label that would cover one already drawn.
        for bl, bt, bw, bh in boxes:
            if left < bl + bw and bl < left + width and top < bt + bh and bt < top + height:
                return
    boxes.append((left, top, width, height))
    color = ann.get("color") or "#000000"
    if ann.get("style") == "label":
        pad = 0.25 * size
        commands.append((
            "rect", left - pad, top - pad * 0.6, width + 2 * pad, height + pad * 1.2,
            surface, color, 0.8, 1.0,
        ))
    # Draw from the box centre: the renderers agree on "middle" anchoring.
    commands.append((
        "text", left + width / 2.0, top + height / 2.0, text, size, color,
        "middle", "middle", 0, int(ann.get("weight") or 400),
    ))


def _point_shapes(layer, blobs, gz, n: int) -> list[str]:
    node = layer.get("shape")
    if isinstance(node, str):
        return [node] * n
    if not isinstance(node, dict) or "id" not in node:
        return ["circle"] * n
    names = node.get("names") or ["circle"]
    codes = _decode(blobs[node["id"]], node.get("dtype") or "u16", gz)
    return [names[int(codes[i]) % len(names)] for i in range(n)]


def _marker(commands, cx, cy, r, shape, color, alpha) -> None:
    """A point symbol of radius ``r`` (circle, triangle, square, diamond, plus, cross)."""
    if shape == "triangle":
        h = r * 1.25
        commands.append(("polygon", [(cx, cy - h), (cx + h * 0.95, cy + h * 0.6), (cx - h * 0.95, cy + h * 0.6)], color, None, 0, alpha))
    elif shape == "square":
        k = r * 0.9
        commands.append(("polygon", [(cx - k, cy - k), (cx + k, cy - k), (cx + k, cy + k), (cx - k, cy + k)], color, None, 0, alpha))
    elif shape == "diamond":
        k = r * 1.2
        commands.append(("polygon", [(cx, cy - k), (cx + k, cy), (cx, cy + k), (cx - k, cy)], color, None, 0, alpha))
    elif shape in {"plus", "cross"}:
        k = r * 1.1
        w = max(1.0, r * 0.45)
        if shape == "plus":
            segs = [((cx - k, cy), (cx + k, cy)), ((cx, cy - k), (cx, cy + k))]
        else:
            d = k * 0.75
            segs = [((cx - d, cy - d), (cx + d, cy + d)), ((cx - d, cy + d), (cx + d, cy - d))]
        for a, b in segs:
            commands.append(("line", a[0], a[1], b[0], b[1], color, w, alpha))
    else:
        commands.append(("circle", cx, cy, r, color, None, 0, alpha))


def _dash_path(points, pattern, width):
    """Dashes along a polyline, continuing the pattern across vertices."""
    unit = max(float(width), 1.0)
    steps = [max(float(p), 0.5) * unit for p in pattern]
    out = []
    index, left = 0, steps[0]
    for a, b in zip(points, points[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length <= 0:
            continue
        dx, dy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        at = 0.0
        while at < length:
            step = min(left, length - at)
            if index % 2 == 0:
                out.append(((a[0] + dx * at, a[1] + dy * at), (a[0] + dx * (at + step), a[1] + dy * (at + step))))
            at += step
            left -= step
            if left <= 1e-9:
                index += 1
                left = steps[index % len(steps)]
    return out


def _arrow_head(tip, back, angle: float, length: float):
    """The two barb ends of an arrowhead at ``tip``, pointing away from ``back``."""
    dx, dy = tip[0] - back[0], tip[1] - back[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return None
    ux, uy = dx / norm, dy / norm
    a = math.radians(angle)
    barbs = []
    for sign in (1.0, -1.0):
        cos_a, sin_a = math.cos(a), sign * math.sin(a)
        rx = ux * cos_a - uy * sin_a
        ry = ux * sin_a + uy * cos_a
        barbs.append((tip[0] - rx * length, tip[1] - ry * length))
    return barbs


def _draw_arrows(spec, commands, px) -> None:
    """arrow() heads, drawn in screen space at the ends of their lines."""
    scales = spec.get("scales") or {}
    sx, sy = scales.get("x") or {}, scales.get("y") or {}
    for item in spec.get("arrows") or []:
        tip = px(_unit(sx, item["x1"]), _unit(sy, item["y1"]))
        back = px(_unit(sx, item["x0"]), _unit(sy, item["y0"]))
        barbs = _arrow_head(tip, back, float(item.get("angle", 30.0)), float(item.get("length", 24.0)))
        if barbs is None:
            continue
        color = _hex(item.get("color") or "#000000")
        width = float(item.get("width") or 1.0)
        if item.get("type") == "closed":
            commands.append(("polygon", [barbs[0], tip, barbs[1]], color, color, width, 1.0))
        else:
            for barb in barbs:
                commands.append(("line", tip[0], tip[1], barb[0], barb[1], color, width, 1.0))


def _draw_rugs(spec, commands, box, px, window) -> None:
    """geom_rug: a short tick at the panel edge for every value."""
    scales = spec.get("scales") or {}
    left, right, bottom, top = window
    for rug in spec.get("rugs") or []:
        side = rug.get("side")
        axis = "x" if side in {"b", "t"} else "y"
        scale = scales.get(axis) or {}
        reach = float(rug.get("length") or 0.03) * (box[3] if axis == "x" else box[2])
        width = float(rug.get("width") or 0.75)
        alpha = float(rug.get("alpha", 1.0))
        colors = rug.get("colors")
        base = rug.get("color") or "#000000"
        for i, value in enumerate(rug.get("values") or []):
            u = _unit(scale, value)
            color = _hex(colors[i]) if colors else base
            if axis == "x":
                if not left <= u <= right:
                    continue
                sx, _sy = px(u, bottom)
                y0 = box[1] + box[3] if side == "b" else box[1]
                y1 = y0 - reach if side == "b" else y0 + reach
                commands.append(("line", sx, y0, sx, y1, color, width, alpha))
            else:
                if not bottom <= u <= top:
                    continue
                _sx, sy = px(left, u)
                x0 = box[0] if side == "l" else box[0] + box[2]
                x1 = x0 + reach if side == "l" else x0 - reach
                commands.append(("line", x0, sy, x1, sy, color, width, alpha))


def _draw_refs(spec, commands, window, px) -> None:
    """geom_hline / geom_vline / geom_abline, clipped to the panel."""
    scales = spec.get("scales") or {}
    sx, sy = scales.get("x") or {}, scales.get("y") or {}
    left, right, bottom, top = window
    for ref in spec.get("refs") or []:
        kind = ref.get("kind")
        if kind == "hline":
            v = _unit(sy, ref["value"])
            if not bottom <= v <= top:
                continue
            ends = [(left, v), (right, v)]
        elif kind == "vline":
            u = _unit(sx, ref["value"])
            if not left <= u <= right:
                continue
            ends = [(u, bottom), (u, top)]
        else:
            ends = _abline_ends(ref, sx, sy, window)
            if ends is None:
                continue
        a, b = px(*ends[0]), px(*ends[1])
        width = float(ref.get("width") or 1.0)
        color = ref.get("color") or "#000000"
        alpha = float(ref.get("alpha", 1.0))
        for p0, p1 in _dash_segments(a, b, ref.get("dash"), width):
            commands.append(("line", p0[0], p0[1], p1[0], p1[1], color, width, alpha))


def _abline_ends(ref, sx, sy, window):
    """The visible piece of y = intercept + slope x, in unit coordinates."""
    left, right, bottom, top = window
    xlo, xhi = float(sx.get("lo", 0.0)), float(sx.get("hi", 1.0))
    ylo, yhi = float(sy.get("lo", 0.0)), float(sy.get("hi", 1.0))
    def unit_y(u):
        x = xlo + u * (xhi - xlo)
        y = float(ref["intercept"]) + float(ref["slope"]) * x
        return (y - ylo) / ((yhi - ylo) or 1.0)
    u0, u1 = left, right
    v0, v1 = unit_y(u0), unit_y(u1)
    # Clip the segment to bottom <= v <= top.
    if v0 == v1:
        return [(u0, v0), (u1, v1)] if bottom <= v0 <= top else None
    t_lo = (bottom - v0) / (v1 - v0)
    t_hi = (top - v0) / (v1 - v0)
    t0, t1 = max(0.0, min(t_lo, t_hi)), min(1.0, max(t_lo, t_hi))
    if t1 <= t0:
        return None
    return [
        (u0 + t0 * (u1 - u0), v0 + t0 * (v1 - v0)),
        (u0 + t1 * (u1 - u0), v0 + t1 * (v1 - v0)),
    ]


def _dash_segments(a, b, pattern, width):
    """Split a line into dashes; the pattern is in multiples of the width."""
    if not pattern:
        return [(a, b)]
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length <= 0:
        return []
    unit = max(float(width), 1.0)
    steps = [max(float(p), 0.5) * unit for p in pattern]
    dx, dy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    out, at, index = [], 0.0, 0
    while at < length:
        step = steps[index % len(steps)]
        end = min(at + step, length)
        if index % 2 == 0:
            out.append(((a[0] + dx * at, a[1] + dy * at), (a[0] + dx * end, a[1] + dy * end)))
        at = end
        index += 1
    return out


def _unit(scale: dict, value) -> float:
    lo = float(scale.get("lo", 0.0))
    hi = float(scale.get("hi", 1.0))
    span = hi - lo or 1.0
    return (float(value) - lo) / span


def _groups(layer, n: int):
    groups = layer.get("groups")
    if not groups:
        return [(0, n)]
    out = []
    for start, count in groups:
        start = int(start)
        count = int(count)
        if start < 0 or start >= n or count <= 0:
            continue
        out.append((start, min(count, n - start)))
    return out or [(0, n)]


def _decode(b64: str, dtype: str, gz: bool) -> np.ndarray:
    raw = base64.b64decode(b64)
    if gz:
        raw = gzip.decompress(raw)
    if dtype == "f32":
        return np.frombuffer(raw, dtype="<f4").astype(np.float64, copy=False)
    if dtype == "u32":
        return np.frombuffer(raw, dtype="<u4").copy()
    if dtype == "u8":
        return np.frombuffer(raw, dtype=np.uint8).copy()
    buf = np.frombuffer(raw, dtype=np.uint8)
    if gz:
        half = buf.size // 2
        delta = buf[:half].astype(np.uint32) | (buf[half:half + half].astype(np.uint32) << 8)
        return (np.cumsum(delta, dtype=np.uint32) & np.uint32(0xFFFF)).astype(np.uint16)
    return np.frombuffer(raw, dtype="<u2").copy()


def _norm_channel(layer, name, blobs, gz, n: int) -> np.ndarray:
    node = layer.get(name)
    if not isinstance(node, dict) or "id" not in node:
        raise ValueError(f"ggsave() is missing the {name} channel on a {layer.get('kind')!r} layer")
    values = _decode(blobs[node["id"]], node.get("dtype") or "u16", gz)
    if values.dtype == np.float64 or str(node.get("dtype")) == "f32":
        out = np.asarray(values, dtype=np.float64)
    else:
        out = values.astype(np.float64) / 65535.0
    if out.size < n:
        raise ValueError(f"ggsave() {name} channel is shorter than the layer")
    return out[:n]


def _layer_colors(layer, spec, blobs, gz, n: int):
    theme = spec.get("theme") or {}
    default = _rgb(layer.get("constColor") or (theme.get("cat") or ["#3987e5"])[0])
    node = layer.get("color")
    if not isinstance(node, dict) or "id" not in node:
        return [default] * n
    data = _decode(blobs[node["id"]], node.get("dtype") or "u16", gz)
    if node.get("kind") == "cat":
        palette = [
            _rgb(item) for item in ((spec.get("color") or {}).get("palette") or theme.get("cat") or [])
        ]
        if not palette:
            return [default] * n
        return [palette[int(data[i]) % len(palette)] for i in range(n)]
    ramp = [_rgb(item) for item in ((spec.get("color") or {}).get("ramp") or theme.get("seq") or [])]
    if data.dtype == np.float64:
        return [_ramp_at(ramp, float(data[i])) for i in range(n)]
    return [_ramp_at(ramp, float(data[i]) / 65535.0) for i in range(n)]


def _outlier_colors(layer, spec, blobs, gz, n, colors):
    node = layer.get("ocolor")
    if not isinstance(node, dict) or "id" not in node:
        if layer.get("constColor"):
            return [_rgb(layer["constColor"])] * n
        return [colors[0]] * n
    fake = {"color": node, "constColor": layer.get("constColor"), "n": n}
    return _layer_colors(fake, spec, blobs, gz, n)


def _point_radii(layer, blobs, gz, n, *, scene: bool, min_dim: float) -> np.ndarray:
    node = layer.get("size")
    if isinstance(node, dict) and node.get("id"):
        frac = _norm_channel(layer, "size", blobs, gz, n)
        diam = frac * float(node.get("max") or 1.0)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        diam = np.full(n, float(node), dtype=np.float64)
    else:
        diam = np.full(n, 6.0, dtype=np.float64)
    if scene:
        diam = np.maximum(diam * float(min_dim) * 0.5, 1.5)
    return np.maximum(diam / 2.0, 0.6)


def _raw_size_numbers(layer, blobs, gz, n) -> np.ndarray:
    node = layer.get("size")
    if isinstance(node, dict) and node.get("id"):
        return _norm_channel(layer, "size", blobs, gz, n) * float(node.get("max") or 1.0)
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return np.full(n, float(node), dtype=np.float64)
    return np.full(n, 2.0, dtype=np.float64)


def _average(colors, idx):
    acc = np.zeros(3, dtype=np.float64)
    for i in idx:
        acc += colors[i]
    return acc / len(idx)


_CSS_COLOURS = {
    "aliceblue": "f0f8ff", "antiquewhite": "faebd7", "aqua": "00ffff", "aquamarine": "7fffd4",
    "azure": "f0ffff", "beige": "f5f5dc", "bisque": "ffe4c4", "black": "000000",
    "blanchedalmond": "ffebcd", "blue": "0000ff", "blueviolet": "8a2be2", "brown": "a52a2a",
    "burlywood": "deb887", "cadetblue": "5f9ea0", "chartreuse": "7fff00", "chocolate": "d2691e",
    "coral": "ff7f50", "cornflowerblue": "6495ed", "cornsilk": "fff8dc", "crimson": "dc143c",
    "cyan": "00ffff", "darkblue": "00008b", "darkcyan": "008b8b", "darkgoldenrod": "b8860b",
    "darkgray": "a9a9a9", "darkgreen": "006400", "darkgrey": "a9a9a9", "darkkhaki": "bdb76b",
    "darkmagenta": "8b008b", "darkolivegreen": "556b2f", "darkorange": "ff8c00", "darkorchid": "9932cc",
    "darkred": "8b0000", "darksalmon": "e9967a", "darkseagreen": "8fbc8f", "darkslateblue": "483d8b",
    "darkslategray": "2f4f4f", "darkslategrey": "2f4f4f", "darkturquoise": "00ced1", "darkviolet": "9400d3",
    "deeppink": "ff1493", "deepskyblue": "00bfff", "dimgray": "696969", "dimgrey": "696969",
    "dodgerblue": "1e90ff", "firebrick": "b22222", "floralwhite": "fffaf0", "forestgreen": "228b22",
    "fuchsia": "ff00ff", "gainsboro": "dcdcdc", "ghostwhite": "f8f8ff", "gold": "ffd700",
    "goldenrod": "daa520", "gray": "808080", "green": "008000", "greenyellow": "adff2f",
    "grey": "808080", "honeydew": "f0fff0", "hotpink": "ff69b4", "indianred": "cd5c5c",
    "indigo": "4b0082", "ivory": "fffff0", "khaki": "f0e68c", "lavender": "e6e6fa",
    "lavenderblush": "fff0f5", "lawngreen": "7cfc00", "lemonchiffon": "fffacd", "lightblue": "add8e6",
    "lightcoral": "f08080", "lightcyan": "e0ffff", "lightgoldenrodyellow": "fafad2", "lightgray": "d3d3d3",
    "lightgreen": "90ee90", "lightgrey": "d3d3d3", "lightpink": "ffb6c1", "lightsalmon": "ffa07a",
    "lightseagreen": "20b2aa", "lightskyblue": "87cefa", "lightslategray": "778899", "lightslategrey": "778899",
    "lightsteelblue": "b0c4de", "lightyellow": "ffffe0", "lime": "00ff00", "limegreen": "32cd32",
    "linen": "faf0e6", "magenta": "ff00ff", "maroon": "800000", "mediumaquamarine": "66cdaa",
    "mediumblue": "0000cd", "mediumorchid": "ba55d3", "mediumpurple": "9370db", "mediumseagreen": "3cb371",
    "mediumslateblue": "7b68ee", "mediumspringgreen": "00fa9a", "mediumturquoise": "48d1cc", "mediumvioletred": "c71585",
    "midnightblue": "191970", "mintcream": "f5fffa", "mistyrose": "ffe4e1", "moccasin": "ffe4b5",
    "navajowhite": "ffdead", "navy": "000080", "oldlace": "fdf5e6", "olive": "808000",
    "olivedrab": "6b8e23", "orange": "ffa500", "orangered": "ff4500", "orchid": "da70d6",
    "palegoldenrod": "eee8aa", "palegreen": "98fb98", "paleturquoise": "afeeee", "palevioletred": "db7093",
    "papayawhip": "ffefd5", "peachpuff": "ffdab9", "peru": "cd853f", "pink": "ffc0cb",
    "plum": "dda0dd", "powderblue": "b0e0e6", "purple": "800080", "rebeccapurple": "663399",
    "red": "ff0000", "rosybrown": "bc8f8f", "royalblue": "4169e1", "saddlebrown": "8b4513",
    "salmon": "fa8072", "sandybrown": "f4a460", "seagreen": "2e8b57", "seashell": "fff5ee",
    "sienna": "a0522d", "silver": "c0c0c0", "skyblue": "87ceeb", "slateblue": "6a5acd",
    "slategray": "708090", "slategrey": "708090", "snow": "fffafa", "springgreen": "00ff7f",
    "steelblue": "4682b4", "tan": "d2b48c", "teal": "008080", "thistle": "d8bfd8",
    "tomato": "ff6347", "turquoise": "40e0d0", "violet": "ee82ee", "wheat": "f5deb3",
    "white": "ffffff", "whitesmoke": "f5f5f5", "yellow": "ffff00", "yellowgreen": "9acd32",
}


def _named_colour(text: str) -> str | None:
    """CSS names ("steelblue"), R's greys ("grey50", "gray80"), rgb(r, g, b)."""
    name = text.strip().lower().replace(" ", "")
    if name in _CSS_COLOURS:
        return _CSS_COLOURS[name]
    for prefix in ("grey", "gray"):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            level = int(round(int(name[len(prefix):]) / 100.0 * 255))
            if 0 <= level <= 255:
                return f"{level:02x}" * 3
    if name.startswith("rgb(") and name.endswith(")"):
        try:
            parts = [int(float(v)) for v in name[4:-1].split(",")[:3]]
            return "".join(f"{max(0, min(255, v)):02x}" for v in parts)
        except ValueError:
            return None
    return None


def _rgb(color) -> tuple[int, int, int]:
    text = str(color or "#000000").strip()
    if not text.startswith("#"):
        text = _named_colour(text) or text
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except (ValueError, IndexError):
        return (0, 0, 0)


def _hex(color) -> str:
    if isinstance(color, str):
        named = None if color.startswith("#") else _named_colour(color)
        return f"#{named}" if named else color
    return "#{:02x}{:02x}{:02x}".format(*color)


def _ramp_at(ramp, t: float) -> tuple[int, int, int]:
    if not ramp:
        return (128, 128, 128)
    if len(ramp) == 1:
        return ramp[0]
    k = min(len(ramp) - 1.001, max(0.0, float(t) * (len(ramp) - 1)))
    i = int(math.floor(k))
    f = k - i
    a = ramp[i]
    b = ramp[min(i + 1, len(ramp) - 1)]
    return tuple(int(round(a[c] + (b[c] - a[c]) * f)) for c in range(3))


def _scale_for(size: float) -> int:
    return max(1, int(round(float(size) / 7.0)))


def _line_height(size: float) -> int:
    return 7 * _scale_for(size)


def _note_line_height(size: float) -> float:
    # Real fonts need about 1.25 em; the bitmap height alone lets two
    # stacked notes touch.
    return max(_line_height(size) + 2, 1.25 * float(size))


def _glyph_for(ch: str):
    if ch in _FONT:
        return _FONT[ch], "normal"
    if ch in _SUP_CHARS:
        return _FONT.get(_SUP_CHARS[ch], _FONT["?"]), "sup"
    if ch in _SUB_CHARS:
        return _FONT.get(_SUB_CHARS[ch], _FONT["?"]), "sub"
    folded = ch.translate(_FOLD)
    if folded in _FONT:
        return _FONT[folded], "normal"
    return _FONT["?"], "normal"


_SUP_CHARS = dict(zip(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ",
    "0123456789+-()abcdefghijklmnoprstuvwxyz",
))
_SUB_CHARS = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋₍₎", "0123456789+-()"))


def _pieces(text: str, scale: int):
    """Glyphs as (columns, pixel width, pixel height, kind)."""
    out = []
    for ch in text.replace("\n", " "):
        cols, kind = _glyph_for(ch)
        used = 1 if kind != "normal" else scale
        if ch == " ":
            out.append((None, 4 * used, 7 * used, "space"))
            continue
        out.append((cols, 5 * used, 7 * used, kind))
    return out


# Helvetica advance widths (AFM, 1/1000 em). Arial matches them. Used when
# the file is drawn with real fonts (SVG, PDF, Cairo PNG); the bitmap
# fallback keeps its own wider metrics below.
_HELVETICA = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667,
    "'": 191, "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333,
    ".": 278, "/": 278, ":": 278, ";": 278, "<": 584, "=": 584, ">": 584,
    "?": 556, "@": 1015, "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556,
    "`": 333, "{": 334, "|": 260, "}": 334, "~": 584,
    "A": 667, "B": 667, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778,
    "H": 722, "I": 278, "J": 500, "K": 667, "L": 556, "M": 833, "N": 722,
    "O": 778, "P": 667, "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722,
    "V": 667, "W": 944, "X": 667, "Y": 667, "Z": 611,
    "a": 556, "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556,
    "h": 556, "i": 222, "j": 222, "k": 500, "l": 222, "m": 833, "n": 556,
    "o": 556, "p": 556, "q": 556, "r": 333, "s": 500, "t": 278, "u": 556,
    "v": 500, "w": 722, "x": 500, "y": 500, "z": 500,
    "−": 584, "±": 584, "×": 584, "÷": 584, "·": 278, "°": 400, "…": 1000,
    "≤": 584, "≥": 584, "≠": 584, "≈": 584, "∞": 713, "∫": 274, "√": 549,
}
_RAISED = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜᵃᵇᶜᵈᵉᶠᵍʰʲᵏˡᵐᵒᵖʳˢᵗᵘᵛʷˣʸᶻᵢⱼᵣᵤᵥ")
_REAL_FONT = contextvars.ContextVar("plot3_real_font", default=False)


def _real_width(text: str, size: float) -> int:
    units = 0
    for ch in text.replace("\n", " "):
        if ch in _HELVETICA:
            units += _HELVETICA[ch]
        elif ch in _RAISED:
            units += 380
        elif ch.isdigit():
            units += 556
        else:
            units += 600  # Greek and other symbols: a little wider than a letter
    # A few percent of headroom: the real font may be Arial or a fallback.
    return int(math.ceil(units * float(size) / 1000.0 * 1.04))


def _text_width(text: str, size: float) -> int:
    if _REAL_FONT.get():
        return _real_width(text, size)
    scale = _scale_for(size)
    pieces = _pieces(text, scale)
    if not pieces:
        return 0
    gaps = scale * (len(pieces) - 1)
    return sum(item[1] for item in pieces) + gaps


def _svg_text(
    commands,
    width: int,
    height: int,
    *,
    svg_width: str | None = None,
    svg_height: str | None = None,
    family: str | None = None,
    wide: str | None = None,
) -> str:
    """``wide`` is the font for text with scripts the main family lacks
    (CJK, ✓, emoji): one name for Cairo, which does not fall back per
    character, or a fallback list for browsers."""
    shown_w = str(width) if svg_width is None else svg_width
    shown_h = str(height) if svg_height is None else svg_height
    family_name = family or _DEFAULT_FAMILY
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{shown_w}" height="{shown_h}" '
            f'viewBox="0 0 {width} {height}" overflow="hidden">'
        ),
        f"<metadata>plot3 {escape(__version__)}</metadata>",
    ]
    clips = 0
    for cmd in commands:
        if cmd[0] == "clip":
            # Marks drawn until the matching "unclip" stay inside this box.
            clips += 1
            _op, x, y, w, h = cmd
            parts.append(
                f'<clipPath id="plot3clip{clips}"><rect x="{_num(x)}" y="{_num(y)}" '
                f'width="{_num(max(w, 0))}" height="{_num(max(h, 0))}"/></clipPath>'
                f'<g clip-path="url(#plot3clip{clips})">'
            )
            continue
        if cmd[0] == "unclip":
            parts.append("</g>")
            continue
        if wide and cmd[0] == "text" and _needs_wide_font(str(cmd[3])):
            parts.append(_svg_cmd(cmd, wide))
        else:
            parts.append(_svg_cmd(cmd, family_name))
    parts.append("</svg>")
    return "".join(parts)


_WIDE_FALLBACKS = (
    "'Arial Unicode MS', 'Hiragino Sans', 'PingFang SC', 'Noto Sans CJK SC', "
    "'Microsoft YaHei', 'Segoe UI Symbol', 'DejaVu Sans', sans-serif"
)
# Scripts and symbols Helvetica / Arial do not have.
_WIDE_RANGES = (
    (0x0590, 0x0FFF),   # Hebrew, Arabic, Indic, Thai, Tibetan
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x2600, 0x27BF),   # symbols and dingbats (✓, ★)
    (0x2E80, 0x9FFF),   # CJK, kana
    (0xAC00, 0xD7AF),   # Hangul
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),   # full-width forms
    (0x1F000, 0x1FAFF),  # emoji
)
_WIDE_CANDIDATES = {
    "darwin": [
        ("Arial Unicode MS", ["/Library/Fonts/Arial Unicode.ttf",
                              "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]),
        ("Hiragino Sans", ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                           "/System/Library/Fonts/Hiragino Sans GB.ttc"]),
    ],
    "win32": [
        ("Microsoft YaHei", ["C:/Windows/Fonts/msyh.ttc"]),
        ("Arial Unicode MS", ["C:/Windows/Fonts/ARIALUNI.TTF"]),
    ],
}
_WIDE_FONT: list = []


def _needs_wide_font(text: str) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in _WIDE_RANGES)


def _wide_font() -> str | None:
    """One installed font with CJK and symbol glyphs, or None."""
    if _WIDE_FONT:
        return _WIDE_FONT[0]
    import os
    import shutil
    import subprocess
    import sys

    found = None
    for name, paths in _WIDE_CANDIDATES.get(sys.platform, []):
        if any(os.path.exists(path) for path in paths):
            found = name
            break
    if found is None and shutil.which("fc-list"):
        try:
            listing = subprocess.run(
                ["fc-list", ":", "family"], capture_output=True, text=True, timeout=5,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            listing = ""
        for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
                     "WenQuanYi Zen Hei", "Arial Unicode MS", "DejaVu Sans"):
            if name in listing:
                found = name
                break
    _WIDE_FONT.append(found)
    return found


def _svg_cmd(cmd, family: str = _DEFAULT_FAMILY) -> str:
    op = cmd[0]
    if op == "rect":
        _op, x, y, w, h, fill, stroke, sw, alpha = cmd
        attrs = [
            f'x="{_num(x)}"', f'y="{_num(y)}"', f'width="{_num(max(w, 0))}"',
            f'height="{_num(max(h, 0))}"',
        ]
        if fill:
            attrs.append(f'fill="{fill}"')
            if alpha < 0.999:
                attrs.append(f'fill-opacity="{alpha:.3f}"')
        else:
            attrs.append('fill="none"')
        if stroke and sw:
            attrs.append(f'stroke="{stroke}"')
            attrs.append(f'stroke-width="{_num(sw)}"')
            if alpha < 0.999 and not fill:
                attrs.append(f'stroke-opacity="{alpha:.3f}"')
        return "<rect " + " ".join(attrs) + "/>"
    if op == "line":
        _op, x1, y1, x2, y2, stroke, sw, alpha = cmd
        opacity = f' stroke-opacity="{alpha:.3f}"' if alpha < 0.999 else ""
        return (
            f'<line x1="{_num(x1)}" y1="{_num(y1)}" x2="{_num(x2)}" y2="{_num(y2)}" '
            f'stroke="{stroke}" stroke-width="{_num(sw)}" stroke-linecap="round"{opacity}/>'
        )
    if op == "polymask":
        _op, tris, color, alpha = cmd
        parts = []
        for tri in tris:
            if len(tri) < 3:
                continue
            step = [f"M{_num(tri[0][0])},{_num(tri[0][1])}"]
            step.extend(f"L{_num(px)},{_num(py)}" for px, py in tri[1:])
            step.append("Z")
            parts.append("".join(step))
        if not parts:
            return ""
        opacity = f' fill-opacity="{alpha:.3f}"' if alpha < 0.999 else ""
        return f'<path d="{" ".join(parts)}" fill="{color}" fill-rule="nonzero"{opacity}/>'
    if op in {"polyline", "polygon"}:
        _op, pts, color, extra, sw, alpha = _poly_fields(cmd)
        points = " ".join(f"{_num(px)},{_num(py)}" for px, py in pts)
        if op == "polyline":
            opacity = f' stroke-opacity="{alpha:.3f}"' if alpha < 0.999 else ""
            return (
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                f'stroke-width="{_num(sw)}" stroke-linejoin="round" stroke-linecap="round"{opacity}/>'
            )
        fill_op = f' fill-opacity="{alpha:.3f}"' if alpha < 0.999 else ""
        stroke = ""
        if extra and sw:
            stroke = f' stroke="{extra}" stroke-width="{_num(sw)}"'
        return f'<polygon points="{points}" fill="{color or "none"}"{fill_op}{stroke}/>'
    if op == "circle":
        _op, cx, cy, r, fill, stroke, sw, alpha = cmd
        if r <= 0:
            return ""
        opacity = f' fill-opacity="{alpha:.3f}"' if alpha < 0.999 else ""
        stroke_attr = f' stroke="{stroke}" stroke-width="{_num(sw)}"' if stroke and sw else ""
        return (
            f'<circle cx="{_num(cx)}" cy="{_num(cy)}" r="{_num(r)}" '
            f'fill="{fill or "none"}"{opacity}{stroke_attr}/>'
        )
    if op == "text":
        _op, x, y, text, size, fill, anchor, baseline, rotate, weight = cmd
        if not text:
            return ""
        dominant = {"top": "hanging", "middle": "middle", "alphabetic": "alphabetic"}[baseline]
        transform = f' transform="rotate({int(rotate)} {_num(x)} {_num(y)})"' if rotate else ""
        family_attr = f"font-family={quoteattr(family)}"
        return (
            f'<text x="{_num(x)}" y="{_num(y)}" fill="{fill}" font-size="{int(round(float(size)))}" '
            f'{family_attr} font-weight="{int(weight)}" text-anchor="{anchor}" '
            f'dominant-baseline="{dominant}"{transform}>{escape(str(text))}</text>'
        )
    return ""


def _poly_fields(cmd):
    if cmd[0] == "polyline":
        return cmd[0], cmd[1], cmd[2], None, cmd[3], cmd[4]
    return cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5]


def _num(value) -> str:
    text = f"{float(value):.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _raster(commands, width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    saved: list = []
    for cmd in commands:
        if cmd[0] == "clip":
            saved.append((image.copy(), cmd[1:]))
            continue
        if cmd[0] == "unclip":
            if saved:
                before, (x, y, w, h) = saved.pop()
                keep = np.ones(image.shape[:2], dtype=bool)
                x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
                x1, y1 = min(width, int(math.ceil(x + w))), min(height, int(math.ceil(y + h)))
                keep[y0:y1, x0:x1] = False
                image[keep] = before[keep]
            continue
        _paint(image, cmd)
    return image


def _paint(image, cmd) -> None:
    op = cmd[0]
    if op == "rect":
        _op, x, y, w, h, fill, stroke, sw, alpha = cmd
        if fill and w > 0 and h > 0:
            _blend_rect(image, x, y, w, h, _rgb(fill), alpha)
        if stroke and sw and w > 0 and h > 0:
            color = _rgb(stroke)
            t = float(sw)
            _blend_rect(image, x, y, w, t, color, alpha)
            _blend_rect(image, x, y + h - t, w, t, color, alpha)
            _blend_rect(image, x, y, t, h, color, alpha)
            _blend_rect(image, x + w - t, y, t, h, color, alpha)
        return
    if op == "line":
        _op, x1, y1, x2, y2, stroke, sw, alpha = cmd
        _stroke_segment(image, x1, y1, x2, y2, float(sw), _rgb(stroke), alpha)
        return
    if op == "polyline":
        _op, pts, color, sw, alpha = cmd
        rgb = _rgb(color)
        for a, b in zip(pts, pts[1:]):
            _stroke_segment(image, a[0], a[1], b[0], b[1], float(sw), rgb, alpha)
        return
    if op == "polymask":
        _op, tris, fill, alpha = cmd
        _fill_polymask(image, tris, _rgb(fill), alpha)
        return
    if op == "polygon":
        _op, pts, fill, stroke, sw, alpha = cmd
        if fill and len(pts) >= 3:
            _fill_polygon(image, pts, _rgb(fill), alpha)
        if stroke and sw and len(pts) >= 2:
            rgb = _rgb(stroke)
            loop = list(pts) + [pts[0]]
            for a, b in zip(loop, loop[1:]):
                _stroke_segment(image, a[0], a[1], b[0], b[1], float(sw), rgb, alpha)
        return
    if op == "circle":
        _op, cx, cy, r, fill, stroke, sw, alpha = cmd
        if fill and r > 0:
            _fill_circle(image, cx, cy, float(r), _rgb(fill), alpha)
        if stroke and sw and r > 0:
            _stroke_circle(image, cx, cy, float(r), float(sw), _rgb(stroke), alpha)
        return
    if op == "text":
        _paint_text(image, cmd)


def _blend_rect(image, x, y, w, h, color, alpha) -> None:
    height, width = image.shape[:2]
    x0 = max(0, int(math.floor(x)))
    y0 = max(0, int(math.floor(y)))
    x1 = min(width, int(math.ceil(x + w)))
    y1 = min(height, int(math.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return
    view = image[y0:y1, x0:x1]
    _blend_array(view, color, alpha)


def _blend_array(view, color, alpha) -> None:
    if alpha >= 0.999:
        view[:] = color
        return
    paint = np.array(color, dtype=np.float32)
    mixed = view.astype(np.float32) * (1.0 - alpha) + paint * alpha + 0.5
    view[:] = np.clip(mixed, 0, 255).astype(np.uint8)


def _blend_mask(image, y0, x0, mask, color, alpha) -> None:
    height, width = image.shape[:2]
    if y0 >= height or x0 >= width or y0 + mask.shape[0] <= 0 or x0 + mask.shape[1] <= 0:
        return
    sub_y = 0
    sub_x = 0
    y1 = y0 + mask.shape[0]
    x1 = x0 + mask.shape[1]
    if y0 < 0:
        sub_y = -y0
        y0 = 0
    if x0 < 0:
        sub_x = -x0
        x0 = 0
    y1 = min(height, y1)
    x1 = min(width, x1)
    mask = mask[sub_y:sub_y + (y1 - y0), sub_x:sub_x + (x1 - x0)]
    if not mask.any():
        return
    view = image[y0:y1, x0:x1]
    if alpha >= 0.999:
        view[mask] = color
        return
    paint = np.array(color, dtype=np.float32)
    pix = view[mask].astype(np.float32)
    view[mask] = np.clip(pix * (1.0 - alpha) + paint * alpha + 0.5, 0, 255).astype(np.uint8)


def _fill_circle(image, cx, cy, radius, color, alpha) -> None:
    pad = radius + 1
    x0 = int(math.floor(cx - pad))
    y0 = int(math.floor(cy - pad))
    x1 = int(math.ceil(cx + pad)) + 1
    y1 = int(math.ceil(cy + pad)) + 1
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    _blend_mask(image, y0, x0, mask, color, alpha)


def _stroke_circle(image, cx, cy, radius, width, color, alpha) -> None:
    outer = radius + width / 2
    inner = max(0.0, radius - width / 2)
    pad = outer + 1
    x0 = int(math.floor(cx - pad))
    y0 = int(math.floor(cy - pad))
    x1 = int(math.ceil(cx + pad)) + 1
    y1 = int(math.ceil(cy + pad)) + 1
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist2 <= outer ** 2) & (dist2 >= inner ** 2)
    _blend_mask(image, y0, x0, mask, color, alpha)


def _stroke_segment(image, x0, y0, x1, y1, width, color, alpha) -> None:
    radius = max(width, 0.8) / 2.0
    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    pad = radius + 1
    minx = int(math.floor(min(x0, x1) - pad))
    miny = int(math.floor(min(y0, y1) - pad))
    maxx = int(math.ceil(max(x0, x1) + pad)) + 1
    maxy = int(math.ceil(max(y0, y1) + pad)) + 1
    if maxx <= minx or maxy <= miny:
        return
    yy, xx = np.ogrid[miny:maxy, minx:maxx]
    if length2 < 1e-8:
        mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius ** 2
    else:
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / length2, 0.0, 1.0)
        px = x0 + t * dx
        py = y0 + t * dy
        mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2
    _blend_mask(image, miny, minx, mask, color, alpha)


def _fill_polymask(image, tris, color, alpha) -> None:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for pts in tris:
        if len(pts) >= 3:
            _fill_polygon(mask, pts, None, 1.0, mark=True)
    if not mask.any():
        return
    if alpha >= 0.999:
        image[mask] = color
        return
    paint = np.array(color, dtype=np.float32)
    pix = image[mask].astype(np.float32)
    image[mask] = np.clip(pix * (1.0 - alpha) + paint * alpha + 0.5, 0, 255).astype(np.uint8)


def _fill_polygon(image, pts, color, alpha, mark=False) -> None:
    height, width = image.shape[:2]
    ys = [p[1] for p in pts]
    y0 = max(0, int(math.floor(min(ys))))
    y1 = min(height - 1, int(math.floor(max(ys))))
    count = len(pts)
    for y in range(y0, y1 + 1):
        scan = y + 0.5
        hits = []
        for i in range(count):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % count]
            if ay > by:
                ax, ay, bx, by = bx, by, ax, ay
            if ay <= scan < by:
                hits.append(ax + (scan - ay) / (by - ay) * (bx - ax))
        hits.sort()
        for i in range(0, len(hits) - 1, 2):
            if mark:
                _mark_span(image, y, hits[i], hits[i + 1])
            else:
                _blend_span(image, y, hits[i], hits[i + 1], color, alpha, width)


def _mark_span(mask, y, x0, x1) -> None:
    if x1 < x0:
        x0, x1 = x1, x0
    width = mask.shape[1]
    ia = max(0, int(math.floor(x0)))
    ib = min(width, int(math.ceil(x1)))
    if ib > ia:
        mask[y, ia:ib] = True


def _blend_span(image, y, x0, x1, color, alpha, width) -> None:
    if x1 < x0:
        x0, x1 = x1, x0
    ia = max(0, int(math.floor(x0)))
    ib = min(width, int(math.ceil(x1)))
    if ib <= ia:
        return
    _blend_array(image[y:y + 1, ia:ib], color, alpha)


def _paint_text(image, cmd) -> None:
    _op, ax, ay, text, size, fill, anchor, baseline, rotate, weight = cmd
    if not text:
        return
    scale = _scale_for(size)
    pieces = _pieces(str(text), scale)
    if not pieces:
        return
    gaps = scale
    total = sum(item[1] for item in pieces) + gaps * (len(pieces) - 1)
    line_h = 7 * scale
    local_x = 0.0
    if anchor == "middle":
        local_x = -total / 2
    elif anchor == "end":
        local_x = -total
    local_y = 0.0
    if baseline == "middle":
        local_y = -line_h / 2
    elif baseline == "alphabetic":
        local_y = -line_h
    color = _rgb(fill)
    cursor = local_x
    for cols, gw, gh, kind in pieces:
        if kind != "space" and cols is not None:
            top = local_y
            if kind == "sub":
                top = local_y + line_h - gh
            _blit(image, ax, ay, cursor, top, cols, gw // 5, color, rotate)
            if weight >= 600:
                _blit(image, ax, ay, cursor + 1, top, cols, gw // 5, color, rotate)
        cursor += gw + gaps


def _blit(image, ax, ay, local_x, local_y, cols, scale, color, rotate) -> None:
    scale = max(1, int(scale))
    for col, bits in enumerate(cols):
        if not bits:
            continue
        for row in range(7):
            if not bits & (1 << row):
                continue
            for dy in range(scale):
                for dx in range(scale):
                    px = local_x + col * scale + dx
                    py = local_y + row * scale + dy
                    if rotate == -90:
                        sx = int(round(ax + py))
                        sy = int(round(ay - px))
                    else:
                        sx = int(round(ax + px))
                        sy = int(round(ay + py))
                    if 0 <= sy < image.shape[0] and 0 <= sx < image.shape[1]:
                        image[sy, sx] = color


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _phys_chunk(dpi: float) -> bytes:
    """PNG pHYs: pixels per metre, so a 300 dpi file is not read as 72 dpi."""
    ppm = max(1, int(round(float(dpi) / 0.0254)))
    return _png_chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))


def _with_phys(data: bytes, dpi: float) -> bytes:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data
    phys = _phys_chunk(dpi)
    pos = 8
    out = bytearray(data[:8])
    inserted = False
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        chunk_end = pos + 12 + length
        if chunk_end > len(data):
            return data
        if tag != b"pHYs":
            out += data[pos:chunk_end]
        if tag == b"IHDR" and not inserted:
            out += phys
            inserted = True
        pos = chunk_end
        if tag == b"IEND":
            break
    return bytes(out)


def _png_bytes(rgb: np.ndarray, dpi: float = _CSS_DPI) -> bytes:
    height, width = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _phys_chunk(dpi)
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )


def _stamp_pdf(data: bytes, version: str) -> bytes:
    """Append an incremental Info dictionary naming the plot3 version.

    Cairo writes its own Producer entry inside a compressed stream. An
    incremental update keeps that file valid and adds an uncompressed
    ``/Creator (plot3 <version>)`` a reader can see.
    """
    if not data.startswith(b"%PDF") or b"startxref" not in data:
        return data
    start_at = data.rfind(b"startxref")
    tail = data[start_at + len(b"startxref"):].splitlines()
    prev = None
    for line in tail:
        text = line.strip()
        if not text:
            continue
        if text == b"%%EOF":
            break
        try:
            prev = int(text)
        except ValueError:
            return data
        break
    if prev is None:
        return data
    root = re.search(br"/Root\s+\d+\s+\d+\s+R", data)
    size = re.search(br"/Size\s+(\d+)", data)
    if root is None or size is None:
        return data
    obj_num = int(size.group(1))
    if not data.endswith(b"\n"):
        data += b"\n"
    info = f"<< /Creator (plot3 {version}) /Producer (plot3 {version}) >>".encode()
    obj = f"{obj_num} 0 obj\n".encode() + info + b"\nendobj\n"
    obj_at = len(data)
    xref_at = obj_at + len(obj)
    xref = b"xref\n" + f"{obj_num} 1\n".encode() + f"{obj_at:010d} 00000 n \n".encode()
    trailer = (
        b"trailer\n<< "
        + root.group(0)
        + f" /Size {obj_num + 1} /Prev {prev} /Info {obj_num} 0 R ".encode()
        + b">>\nstartxref\n"
        + f"{xref_at}\n".encode()
        + b"%%EOF\n"
    )
    return data + obj + xref + trailer
