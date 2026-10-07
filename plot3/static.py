"""Offline PNG, SVG, and PDF export.

PNG, SVG, and PDF replay one list of drawing commands. SVG is text.
PNG and PDF are that same SVG rendered by cairosvg when the optional
``plot3[export]`` extra is installed, so the three files share one
drawing and one font. Without cairosvg, PNG falls back to a zlib RGB
file and a built-in 5×7 font, and PDF raises with an install hint.
The geometry is the static channel already stored on each layer (the
last frame of a transition, the low end of a slider).
"""

from __future__ import annotations

import base64
import gzip
import io
import math
import struct
import zlib
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import numpy as np

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
    "ggsave() needs the optional export extra to write this file with a "
    "journal font. Install it with: pip install 'plot3[export]'"
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
) -> str:
    """Write ``fig`` to a ``.png``, ``.svg``, or ``.pdf`` file.

    Bare ``width`` and ``height`` are pixels. ``units="in"`` (also
    ``"cm"`` and ``"mm"``) with ``dpi`` (default 300) sets a physical
    page. Layout stays in CSS pixels (96 per inch) so type keeps its
    size, and cairosvg rasterizes that SVG at ``dpi``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".png", ".svg", ".pdf"}:
        raise ValueError("save_static() writes .png, .svg, or .pdf")
    size = _figure_size(fig, width, height, units, dpi)
    layout_w, layout_h = size["layout"]
    base_pt = _resolve_base_pt(fig, fontsize)
    family_name = _resolve_family(fig, family)
    commands = _figure_commands(fig, layout_w, layout_h, base_pt)
    svg = _svg_text(
        commands,
        layout_w,
        layout_h,
        svg_width=size["svg_width"],
        svg_height=size["svg_height"],
        family=family_name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fallback = False
    if suffix == ".svg":
        data = svg.encode("utf-8")
    elif suffix == ".pdf":
        data = _pdf_bytes(svg)
    else:
        data, fallback = _png_file_bytes(svg, commands, size)
    path.write_bytes(data)
    print(f"plot3: saved {path} ({len(data) // 1024} KB)")
    if fallback:
        print(
            "plot3: PNG used the built-in font. "
            "pip install 'plot3[export]' for Helvetica."
        )
    return str(path)


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
    data = _cairo_bytes("svg2png", svg, dpi=size["dpi"])
    if data:
        return data, False
    sx, sy = size["scale"]
    png_w, png_h = size["png"]
    if sx != 1.0 or sy != 1.0:
        commands = _scale_commands(commands, sx, sy)
    return _png_bytes(_raster(commands, png_w, png_h)), True


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


def _figure_commands(fig, width: int, height: int, base_pt: float | None = None) -> list:
    panels, grid, parent_title = _panels(fig)
    theme = panels[0][0].get("theme") or {}
    surface = theme.get("surface") or "#0b1020"
    ink = theme.get("ink") or "#ffffff"
    commands: list = [("rect", 0, 0, width, height, surface, None, 0, 1.0)]
    if grid is None:
        spec, blobs = panels[0]
        _draw_spec(spec, blobs, 0, 0, width, height, commands, border=False, base_pt=base_pt)
        return commands

    ncol, nrow = grid
    pad = 8
    title_h = 0
    if parent_title:
        parent_title = _with_frame(parent_title, _static_label(panels[0][0]))
        title_size = _font_sizes(width, base_pt)[1]
        title_h = title_size + 12
        commands.append((
            "text", 14, 6, parent_title, title_size, ink, "start", "top", 0, 600,
        ))
    cell_w = (width - pad * (ncol + 1)) / ncol
    cell_h = (height - title_h - pad * (nrow + 1)) / nrow
    for index, (spec, blobs) in enumerate(panels):
        row, col = divmod(index, ncol)
        x = pad + col * (cell_w + pad)
        y = title_h + pad + row * (cell_h + pad)
        _draw_spec(
            spec, blobs, x, y, cell_w, cell_h, commands, border=True, base_pt=base_pt,
        )
    return commands


def _panels(fig):
    payload = getattr(fig, "_payload", None)
    if payload is not None:
        return [(payload["spec"], payload.get("blobs") or {})], None, ""

    facet = getattr(fig, "facet", None)
    if facet is None:
        from plot3.build import build_spec

        spec, pairs = build_spec(fig)
        return [(spec, dict(pairs))], None, ""

    from plot3.build import (
        _clone_ggplot_with_data,
        _global_numeric_domains,
        _panel_grid,
        build_spec,
    )
    from plot3.mathtext import split_math
    from plot3.table import filter_equal, has_column, unique_levels

    if fig.data is None:
        raise ValueError("ggplot has no data")
    column = facet.variable
    if not has_column(fig.data, column):
        raise KeyError(f"facet column not in DataFrame: {column!r}")
    levels = unique_levels(fig.data, column)
    if not levels:
        raise ValueError("facet_wrap() found no panel levels")
    ncol, nrow = _panel_grid(len(levels), facet.ncol, facet.nrow)
    force = _global_numeric_domains(fig) if facet.scales == "fixed" else {}
    panels = []
    for level in levels:
        if _is_missing(level):
            panel_data = filter_equal(fig.data, column, None)
            label = "NA"
        else:
            panel_data = filter_equal(fig.data, column, level)
            label = str(level)
        panel = _clone_ggplot_with_data(fig, panel_data)
        base = panel.labs.get("title", "")
        panel.labs = dict(panel.labs)
        panel.labs["title"] = f"{base} — {label}" if base else label
        if force:
            panel._force_scales = force
        spec, pairs = build_spec(panel)
        panels.append((spec, dict(pairs)))
    raw = str(fig.labs.get("title", "") or "")
    if "$" in raw:
        raw, _segments = split_math(raw)
    return panels, (ncol, nrow), raw


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
    if is3d:
        _draw_3d(spec, blobs, x, y, w, h, commands, labs, theme, fonts)
    else:
        _draw_2d(spec, blobs, x, y, w, h, commands, labs, theme, fonts)
    _draw_legend(
        spec, commands, _plot_box_after(spec, x, y, w, h, labs, fonts, is3d),
        theme, fonts, labs.get("color") or "",
    )
    notes = spec.get("notes") or []
    if notes:
        commands.append((
            "text", x + 12, y + h - 4, "   ".join(str(n) for n in notes),
            fonts[2], theme.get("muted") or "#898781", "start", "alphabetic", 0, 400,
        ))


def _labs(spec) -> dict:
    label = _static_label(spec)
    raw = spec.get("labs") or {}
    return {key: _with_frame(raw.get(key) or "", label) for key in ("title", "x", "y", "z", "color")}


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


def _plot_box_after(spec, x, y, w, h, labs, fonts, is3d):
    if is3d:
        return _box_3d(x, y, w, h, labs, fonts)
    return _box_2d(spec, x, y, w, h, labs, fonts)


def _box_3d(x, y, w, h, labs, fonts):
    margin = 22
    top = margin + (fonts[1] + 8 if labs.get("title") else 0)
    return (x + margin, y + top, max(8, w - 2 * margin), max(8, h - top - margin))


def _box_2d(spec, x, y, w, h, labs, fonts):
    tick, title, note = fonts
    yticks = [str(lab) for _t, lab in _ticks(spec.get("scales", {}).get("y") or {})]
    tick_w = max((_text_width(lab, tick) for lab in yticks), default=0)
    y_name = labs.get("y") or ""
    name_w = _line_height(tick) if y_name else 0
    left = 8 + name_w + (6 if y_name else 0) + tick_w + 8
    x_name = labs.get("x") or ""
    notes = spec.get("notes") or []
    bottom = 6 + _line_height(tick) + (4 + _line_height(tick) if x_name else 0) + 8
    if notes:
        bottom += _line_height(note) + 4
    top = 10 + (title + 6 if labs.get("title") else 0)
    right = 14
    left = min(left, w * 0.42)
    bottom = min(bottom, h * 0.38)
    top = min(top, h * 0.32)
    return (
        x + left,
        y + top,
        max(8, w - left - right),
        max(8, h - top - bottom),
    )


def _draw_2d(spec, blobs, x, y, w, h, commands, labs, theme, fonts) -> None:
    box = _box_2d(spec, x, y, w, h, labs, fonts)
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
    if _rgb(grid) != _rgb(surface):
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
    for layer in spec.get("layers") or []:
        _draw_layer_2d(layer, spec, blobs, gz, px, commands)

    _draw_frame(commands, box, axis, frame)
    for value, lab in _ticks(scales.get("x") or {}):
        u = _unit(scales.get("x") or {}, value)
        if u < window[0] - 0.02 or u > window[1] + 0.02:
            continue
        sx, _sy = px(u, 0)
        if sx < box[0] - 1 or sx > box[0] + box[2] + 1:
            continue
        commands.append((
            "text", sx, box[1] + box[3] + 4, str(lab), tick_size, muted,
            "middle", "top", 0, 400,
        ))
    for value, lab in _ticks(scales.get("y") or {}):
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
    if labs.get("x"):
        commands.append((
            "text", box[0] + box[2] / 2,
            box[1] + box[3] + 6 + _line_height(tick_size),
            labs["x"], tick_size, ink2, "middle", "top", 0, 400,
        ))
    if labs.get("y"):
        commands.append((
            "text", x + 8 + _line_height(tick_size) / 2, box[1] + box[3] / 2,
            labs["y"], tick_size, ink2, "middle", "middle", -90, 400,
        ))
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
        commands.append((
            "text", sx, sy, text, max(tick_size, 13), ink,
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
        for i in range(n):
            cx, cy = px(float(xs[i]), float(ys[i]))
            commands.append(("circle", cx, cy, float(radii[i]), _hex(colors[i]), None, 0, alpha))
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
    for start, count in _groups(layer, n):
        if count < 2:
            continue
        curve = [px(float(xs[i]), float(ys[i])) for i in range(start, start + count)]
        commands.append(("polyline", curve, _hex(colors[start]), width, alpha))


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
        commands.append(("rect", rect[0], rect[1], rect[2], rect[3], color, color, 1, fill_alpha))
        commands.append(("rect", rect[0], rect[1], rect[2], rect[3], None, color, 1, alpha))
        _seg(commands, px(x, float(ymin[i])), px(x, float(lower[i])), color, alpha)
        _seg(commands, px(x, float(upper[i])), px(x, float(ymax[i])), color, alpha)
        _seg(commands, px(x - cap, float(ymin[i])), px(x + cap, float(ymin[i])), color, alpha)
        _seg(commands, px(x - cap, float(ymax[i])), px(x + cap, float(ymax[i])), color, alpha)
        _seg(commands, px(x - hw, float(middle[i])), px(x + hw, float(middle[i])), ink, alpha)
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


def _draw_3d(spec, blobs, x, y, w, h, commands, labs, theme, fonts) -> None:
    box = _box_3d(x, y, w, h, labs, fonts)
    scales = spec.get("scales") or {}
    spans = []
    for axis in ("x", "y", "z"):
        scale = scales.get(axis) or {}
        spans.append((float(scale.get("hi", 1.0)) - float(scale.get("lo", 0.0))) or 1.0)
    max_span = max(spans + [1e-12])
    aspect = (spec.get("coord") or {}).get("aspect") or "data"
    ext = [1.0, 1.0, 1.0] if aspect == "equal" else [s / max_span for s in spans]
    project = _projector(ext, box)
    axis_color = theme.get("axis") or "#2e3a5c"
    ink2 = theme.get("ink2") or "#c3c2b7"
    ink = theme.get("ink") or "#ffffff"
    gz = bool(spec.get("gz"))
    min_dim = min(box[2], box[3])

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
            radii = _point_radii(layer, blobs, gz, n, scene=True, min_dim=min_dim)
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

    corners = []
    for iz in (0.0, ext[2]):
        for iy in (0.0, ext[1]):
            for ix in (0.0, ext[0]):
                corners.append(project(np.array([ix, iy, iz], dtype=np.float64)))
    edges = (
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    # Far triangles first, then the cube, then lines and points.
    triangles.sort(key=lambda item: -item[0])
    for _depth, poly, color, alpha in triangles:
        commands.append(("polygon", poly, color, None, 0, alpha))
    if str(theme.get("frame") or "box") != "none":
        for a, b in edges:
            pa, pb = corners[a], corners[b]
            if pa is None or pb is None:
                continue
            commands.append(("line", pa[0], pa[1], pb[0], pb[1], axis_color, 1, 1.0))
    for projected, color, width, alpha in lines:
        commands.append(("polyline", projected, color, width, alpha))
    points.sort(key=lambda item: -item[0])
    for _depth, sx, sy, radius, color, alpha in points:
        commands.append(("circle", sx, sy, radius, color, None, 0, alpha))

    _axis_names_3d(project, ext, labs, commands, ink2, fonts[0])
    if labs.get("title"):
        commands.append((
            "text", x + 12, y + 6, labs["title"], fonts[1], ink,
            "start", "top", 0, 600,
        ))


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


def _draw_legend(spec, commands, box, theme, fonts, color_label: str = "") -> None:
    entries = list(spec.get("legend") or [])
    color = spec.get("color") or {}
    size_legend = spec.get("sizeLegend")
    if not entries and color.get("kind") != "num" and not size_legend:
        return
    ink = theme.get("ink") or "#ffffff"
    ink2 = theme.get("ink2") or "#c3c2b7"
    surface = theme.get("surface") or "#0b1020"
    grid = theme.get("grid") or "#1c2742"
    if _rgb(grid) == _rgb(surface):
        grid = theme.get("muted") or "#898781"
    tick = fonts[0]
    row_h = _line_height(tick) + 4
    rows = []
    title = color_label or ""
    if title and (entries or color.get("kind") == "num"):
        rows.append(("title", title))
    for entry in entries:
        rows.append(("swatch", str(entry.get("label") or ""), entry.get("color") or ink))
    ramp = color.get("ramp") or []
    if not entries and color.get("kind") == "num" and ramp:
        rows.append(("ramp", ramp, color.get("lo"), color.get("hi")))
    if size_legend and size_legend.get("breaks"):
        rows.append(("size-title", str(size_legend.get("label") or "size")))
        for br in size_legend["breaks"]:
            rows.append(("bubble", str(br.get("label") or ""), float(br.get("t") or 0)))
    if not rows:
        return
    text_w = 0
    for row in rows:
        if row[0] in {"title", "size-title"}:
            text_w = max(text_w, _text_width(row[1], tick))
        elif row[0] in {"swatch", "bubble"}:
            text_w = max(text_w, _text_width(row[1], tick) + 18)
        elif row[0] == "ramp":
            text_w = max(text_w, 110)
    box_w = text_w + 16
    box_h = 8
    for row in rows:
        if row[0] == "ramp":
            box_h += 28
        elif row[0] == "bubble":
            box_h += max(row_h, 8 + int(round(row[2] * 16)))
        else:
            box_h += row_h
    lx = box[0] + box[2] - box_w - 8
    ly = box[1] + 8
    if lx < box[0]:
        lx = box[0]
    commands.append(("rect", lx, ly, box_w, box_h, surface, grid, 1, 0.92))
    cursor = ly + 6
    for row in rows:
        if row[0] in {"title", "size-title"}:
            commands.append(("text", lx + 8, cursor, row[1], tick, ink, "start", "top", 0, 600))
            cursor += row_h
        elif row[0] == "swatch":
            commands.append(("rect", lx + 8, cursor + 2, 9, 9, row[2], None, 0, 1.0))
            commands.append(("text", lx + 22, cursor, row[1], tick, ink2, "start", "top", 0, 400))
            cursor += row_h
        elif row[0] == "ramp":
            _draw_ramp(commands, lx + 8, cursor + 2, 110, 8, row[1])
            lo = _fmt_tick(row[2], False)
            hi = _fmt_tick(row[3], False)
            commands.append(("text", lx + 8, cursor + 12, lo, max(9, tick - 1), ink2, "start", "top", 0, 400))
            commands.append(("text", lx + 118, cursor + 12, hi, max(9, tick - 1), ink2, "end", "top", 0, 400))
            cursor += 28
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


def _draw_ramp(commands, x, y, w, h, ramp) -> None:
    stops = ramp or ["#000000", "#ffffff"]
    steps = max(2, int(w))
    for i in range(steps):
        color = _hex(_ramp_at([_rgb(stop) for stop in stops], i / (steps - 1)))
        commands.append(("rect", x + i * w / steps, y, w / steps + 0.5, h, color, None, 0, 1.0))


def _axis_names_3d(project, ext, labs, commands, color, size) -> None:
    anchors = [
        np.array([0.5 * ext[0], -0.28 * ext[1], 0.0]),
        np.array([-0.28 * ext[0], 0.5 * ext[1], 0.0]),
        np.array([-0.22 * ext[0], -0.06 * ext[1], 0.62 * ext[2]]),
    ]
    for axis, anchor in zip(("x", "y", "z"), anchors):
        text = labs.get(axis) or ""
        if not text:
            continue
        hit = project(anchor)
        if hit is None:
            continue
        commands.append(("text", hit[0], hit[1], text, size, color, "middle", "middle", 0, 400))


def _projector(ext, box):
    """Perspective camera used by the 3D viewer (fov 60, up = +z)."""
    ctr = np.array([ext[0] / 2, ext[1] / 2, ext[2] / 2], dtype=np.float64)
    backward = np.array([0.55, -0.85, 0.5], dtype=np.float64)
    backward /= np.linalg.norm(backward)
    radius = max(float(np.linalg.norm(ext)) / 2.0, 1e-3)
    dist = radius * 1.55 / math.tan(math.radians(30.0))
    eye = ctr + backward * dist
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, backward)
    norm = float(np.linalg.norm(right))
    if norm < 1e-8:
        right = np.cross(np.array([0.0, 1.0, 0.0]), backward)
        norm = float(np.linalg.norm(right))
    right /= norm
    cam_up = np.cross(backward, right)
    fov_tan = math.tan(math.radians(30.0))
    aspect = box[2] / max(box[3], 1.0)

    def project(point):
        rel = np.asarray(point, dtype=np.float64) - eye
        cam_x = float(np.dot(rel, right))
        cam_y = float(np.dot(rel, cam_up))
        cam_z = float(np.dot(rel, backward))
        depth = -cam_z
        if depth < 1e-6:
            return None
        ndc_x = (cam_x / depth) / fov_tan / aspect
        ndc_y = (cam_y / depth) / fov_tan
        sx = box[0] + (ndc_x * 0.5 + 0.5) * box[2]
        sy = box[1] + (-ndc_y * 0.5 + 0.5) * box[3]
        return sx, sy, depth

    return project


def _view_window(spec, plot_w, plot_h):
    coord = spec.get("coord") or {}
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
        return ladder[0]
    return []


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


def _rgb(color) -> tuple[int, int, int]:
    text = str(color or "#000000").strip()
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
        return color
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


def _text_width(text: str, size: float) -> int:
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
) -> str:
    shown_w = str(width) if svg_width is None else svg_width
    shown_h = str(height) if svg_height is None else svg_height
    family_name = family or _DEFAULT_FAMILY
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{shown_w}" height="{shown_h}" '
            f'viewBox="0 0 {width} {height}" overflow="hidden">'
        ),
    ]
    for cmd in commands:
        parts.append(_svg_cmd(cmd, family_name))
    parts.append("</svg>")
    return "".join(parts)


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
    for cmd in commands:
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


def _png_bytes(rgb: np.ndarray) -> bytes:
    height, width = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
