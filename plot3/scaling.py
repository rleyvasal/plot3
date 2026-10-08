"""ggplot2's scale functions: axis limits, breaks, labels, and palettes.

Position::

    scale_x_continuous(name="Dose (mg)", limits=(0, 10), breaks=[0, 5, 10])
    scale_y_continuous(labels="percent")          # 0.25 -> 25%
    scale_x_discrete(limits=["low", "mid", "high"], labels={"mid": "medium"})
    scale_y_reverse(), xlim(0, 10), ylim("a", "b"), lims(x=(0, 1))
    scale_x_date(date_breaks="3 months", date_labels="%b %Y")

Colour and fill (one channel: fill colours filled shapes, colour the rest)::

    scale_colour_manual(values={"ctrl": "grey", "drug": "firebrick"})
    scale_fill_brewer(palette="Set2"), scale_colour_viridis_d()
    scale_colour_gradient(low="white", high="darkblue")
    scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0)

Shape and linetype::

    scale_shape_manual(values=["circle", "triangle"])
    scale_linetype_manual(values=["solid", "dashed"])

Limits on a continuous axis drop rows outside them, as in ggplot2 (and say
so). A function's samples and computed layers are clipped by the panel.
"""

from __future__ import annotations

import colorsys
import math
from typing import Any

import numpy as np

# ── label formats ────────────────────────────────────────────────────────────


def _trim(number: float, digits: int = 6) -> str:
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def format_label(value: float, labels: Any) -> str:
    """One tick label: "percent", "comma", "dollar", "scientific", a format
    string such as "{:.1f} kg", or a function of the value."""
    if callable(labels):
        return str(labels(value))
    if labels == "percent":
        return f"{_trim(value * 100.0, 4)}%"
    if labels == "comma":
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    if labels == "dollar":
        sign = "-" if value < 0 else ""
        body = f"{abs(value):,.0f}" if float(value).is_integer() else f"{abs(value):,.2f}"
        return f"{sign}${body}"
    if labels == "scientific":
        return f"{value:.2e}"
    if isinstance(labels, str) and "{" in labels:
        return labels.format(value)
    raise ValueError(
        'labels is "percent", "comma", "dollar", "scientific", a format string '
        'such as "{:.1f}", a function, or a list matching breaks'
    )


# ── position scales ──────────────────────────────────────────────────────────

_DATE_UNITS = {
    "sec": "s", "second": "s", "min": "min", "minute": "min", "hour": "h",
    "day": "D", "week": "W-MON", "month": "MS", "quarter": "QS", "year": "YS",
}


def date_freq(spec: str) -> str:
    """ "3 months" -> "3MS" (a pandas frequency)."""
    parts = str(spec).strip().lower().split()
    count, unit = (parts[0], parts[1]) if len(parts) == 2 else ("1", parts[0])
    unit = unit.rstrip("s") if unit not in {"s"} else unit
    if unit not in _DATE_UNITS:
        raise ValueError(
            f"date_breaks {spec!r}: use a count and a unit, such as '3 months', "
            f"'1 week', '2 years' ({', '.join(sorted(_DATE_UNITS))})"
        )
    return f"{int(count)}{_DATE_UNITS[unit]}"


class PositionScale:
    """One x or y scale. ``kind`` is continuous, discrete, or date."""

    def __init__(self, axis, kind, *, name=None, limits=None, breaks=None,
                 labels=None, trans=None, date_breaks=None, date_labels=None):
        self.axis = axis
        self.kind = kind
        self.name = name
        self.limits = limits
        self.breaks = None if breaks is None else list(breaks)
        self.labels = labels
        self.trans = trans
        self.date_breaks = date_breaks
        self.date_labels = date_labels
        if kind == "continuous" and limits is not None:
            if len(limits) != 2:
                raise ValueError(f"{axis} limits are (low, high); use None for an open end")
        if trans not in {None, "log10", "reverse", "identity"}:
            raise ValueError("trans is 'log10', 'reverse', or None")
        if isinstance(labels, (list, tuple)) and self.breaks is not None and len(labels) != len(self.breaks):
            raise ValueError("labels must have one entry per break")
        if date_breaks is not None:
            date_freq(date_breaks)


def _scale(axis, kind, **kw):
    return PositionScale(axis, kind, **kw)


def scale_x_continuous(name=None, *, limits=None, breaks=None, labels=None, trans=None):
    """x axis title, ``limits=(lo, hi)``, ``breaks=[...]``, ``labels=``, ``trans=``."""
    return _scale("x", "continuous", name=name, limits=limits, breaks=breaks, labels=labels, trans=trans)


def scale_y_continuous(name=None, *, limits=None, breaks=None, labels=None, trans=None):
    """y axis: see :func:`scale_x_continuous`."""
    return _scale("y", "continuous", name=name, limits=limits, breaks=breaks, labels=labels, trans=trans)


def scale_x_reverse(name=None, *, limits=None, breaks=None, labels=None):
    """x runs from high to low."""
    return _scale("x", "continuous", name=name, limits=limits, breaks=breaks, labels=labels, trans="reverse")


def scale_y_reverse(name=None, *, limits=None, breaks=None, labels=None):
    """y runs from high to low (depth, rank)."""
    return _scale("y", "continuous", name=name, limits=limits, breaks=breaks, labels=labels, trans="reverse")


def scale_x_discrete(name=None, *, limits=None, labels=None):
    """Category order (``limits``, which also drops the rest) and display ``labels``."""
    return _scale("x", "discrete", name=name, limits=None if limits is None else [str(v) for v in limits], labels=labels)


def scale_y_discrete(name=None, *, limits=None, labels=None):
    return _scale("y", "discrete", name=name, limits=None if limits is None else [str(v) for v in limits], labels=labels)


def scale_x_date(name=None, *, limits=None, date_breaks=None, date_labels=None):
    """Dates on x: ``date_breaks="3 months"``, ``date_labels="%b %Y"``."""
    return _scale("x", "date", name=name, limits=limits, date_breaks=date_breaks, date_labels=date_labels)


def scale_y_date(name=None, *, limits=None, date_breaks=None, date_labels=None):
    return _scale("y", "date", name=name, limits=limits, date_breaks=date_breaks, date_labels=date_labels)


scale_x_datetime = scale_x_date
scale_y_datetime = scale_y_date


def _lim(axis, values):
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = tuple(values[0])
    if len(values) == 2 and all(v is None or isinstance(v, (int, float, np.integer, np.floating)) for v in values):
        return _scale(axis, "continuous", limits=tuple(values))
    if all(isinstance(v, str) for v in values):
        return _scale(axis, "discrete", limits=[str(v) for v in values])
    if len(values) == 2:
        return _scale(axis, "date", limits=tuple(values))
    raise ValueError(f"{axis}lim() takes two numbers, two dates, or category names")


def xlim(*values):
    """``xlim(0, 10)`` (rows outside are dropped) or ``xlim("a", "b")`` (order)."""
    return _lim("x", values)


def ylim(*values):
    """``ylim(0, 100)`` or ``ylim("low", "high")``."""
    return _lim("y", values)


class _Lims(list):
    """Several scales at once."""


def lims(*, x=None, y=None):
    out = _Lims()
    if x is not None:
        out.append(xlim(x))
    if y is not None:
        out.append(ylim(y))
    return out


# ── colour and fill ──────────────────────────────────────────────────────────

_BREWER = {
    "Set1": ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"],
    "Set2": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"],
    "Set3": ["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F"],
    "Dark2": ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"],
    "Paired": ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928"],
    "Pastel1": ["#FBB4AE", "#B3CDE3", "#CCEBC5", "#DECBE4", "#FED9A6", "#FFFFCC", "#E5D8BD", "#FDDAEC", "#F2F2F2"],
    "Pastel2": ["#B3E2CD", "#FDCDAC", "#CBD5E8", "#F4CAE4", "#E6F5C9", "#FFF2AE", "#F1E2CC", "#CCCCCC"],
    "Accent": ["#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17", "#666666"],
    "Blues": ["#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"],
    "Greens": ["#E5F5E0", "#C7E9C0", "#A1D99B", "#74C476", "#41AB5D", "#238B45", "#006D2C", "#00441B"],
    "Reds": ["#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D", "#A50F15", "#67000D"],
    "Oranges": ["#FEE6CE", "#FDD0A2", "#FDAE6B", "#FD8D3C", "#F16913", "#D94801", "#A63603", "#7F2704"],
    "Purples": ["#EFEDF5", "#DADAEB", "#BCBDDC", "#9E9AC8", "#807DBA", "#6A51A3", "#54278F", "#3F007D"],
    "Greys": ["#F0F0F0", "#D9D9D9", "#BDBDBD", "#969696", "#737373", "#525252", "#252525", "#000000"],
    "RdBu": ["#B2182B", "#D6604D", "#F4A582", "#FDDBC7", "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC"],
    "PuOr": ["#B35806", "#E08214", "#FDB863", "#FEE0B6", "#D8DAEB", "#B2ABD2", "#8073AC", "#542788"],
}
_SEQUENTIAL = {"Blues", "Greens", "Reds", "Oranges", "Purples", "Greys", "RdBu", "PuOr"}
# Okabe-Ito: distinguishable with every common colour-vision deficiency.
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def to_hex(colour) -> str:
    """Any colour plot3 accepts ("grey50", "steelblue", "#abc") as #rrggbb."""
    from plot3.static import _named_colour

    if not isinstance(colour, str):
        return colour
    text = colour.strip()
    if text.startswith("#"):
        body = text[1:]
        return "#" + ("".join(ch * 2 for ch in body) if len(body) == 3 else body).lower()
    named = _named_colour(text)
    if named is None:
        raise ValueError(
            f"colour {colour!r} is not a CSS name, an R grey such as 'grey50', or #rrggbb"
        )
    return f"#{named}"


def _hex_rgb(colour: str) -> tuple[float, float, float]:
    from plot3.static import _rgb

    r, g, b = _rgb(colour)
    return r / 255.0, g / 255.0, b / 255.0


def _rgb_hex(rgb) -> str:
    return "#" + "".join(f"{int(round(max(0.0, min(1.0, c)) * 255)):02x}" for c in rgb)


def ramp_at(stops: list[str], t: float) -> str:
    """Colour at ``t`` (0..1) along evenly spaced ``stops``."""
    t = max(0.0, min(1.0, float(t)))
    if len(stops) == 1:
        return stops[0]
    pos = t * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    frac = pos - i
    a, b = _hex_rgb(stops[i]), _hex_rgb(stops[i + 1])
    return _rgb_hex([a[k] + (b[k] - a[k]) * frac for k in range(3)])


def _sample(stops: list[str], n: int, begin: float = 0.0, end: float = 1.0) -> list[str]:
    if n <= 1:
        return [ramp_at(stops, (begin + end) / 2)]
    return [ramp_at(stops, begin + (end - begin) * i / (n - 1)) for i in range(n)]


def extend_palette(base: list[str], n: int) -> list[str]:
    """``n`` colours: the theme's own, then evenly spaced hues (ggplot2's hue
    wheel) instead of an error past eight groups."""
    if n <= len(base):
        return list(base[:n])
    extra = n - len(base)
    hues = [(0.07 + i / extra) % 1.0 for i in range(extra)]
    return list(base) + [_rgb_hex(colorsys.hls_to_rgb(h, 0.55, 0.62)) for h in hues]


class ColourScale:
    """A discrete palette or a continuous ramp for the colour/fill channel."""

    def __init__(self, kind, *, name=None, values=None, palette_fn=None,
                 breaks=None, labels=None, na_value="#7f7f7f",
                 low=None, mid=None, high=None, midpoint=None, limits=None,
                 stops=None, positions=None, identity=False):
        self.kind = kind  # "discrete" | "continuous"
        self.stops = None if stops is None else [to_hex(c) for c in stops]
        self.positions = None if positions is None else [float(v) for v in positions]
        # scale_*_identity(): the column holds the colours themselves.
        self.identity = bool(identity)
        self.name = name
        self.values = values
        self.palette_fn = palette_fn
        self.breaks = None if breaks is None else [str(b) for b in breaks]
        self.labels = labels
        self.na_value = na_value
        self.low, self.mid, self.high, self.midpoint = low, mid, high, midpoint
        self.limits = limits

    def colours(self, levels: list[str], default: list[str]) -> list[str]:
        """One colour per level, in level order, as #rrggbb."""
        return [to_hex(c) for c in self._colours(levels, default)]

    def _colours(self, levels: list[str], default: list[str]) -> list[str]:
        if self.identity:
            return [self.na_value if level in {"nan", "None", "<NA>"} else level for level in levels]
        if isinstance(self.values, dict):
            given = {str(k): v for k, v in self.values.items()}
            return [given.get(level, self.na_value) for level in levels]
        if self.values is not None:
            values = list(self.values)
            order = self.breaks or levels
            by_level = {level: values[i] for i, level in enumerate(order) if i < len(values)}
            if len(values) < len(levels):
                raise ValueError(
                    f"scale_*_manual() has {len(values)} colours for {len(levels)} groups"
                )
            return [by_level.get(level, values[levels.index(level)]) for level in levels]
        if self.palette_fn is not None:
            return list(self.palette_fn(len(levels)))
        return extend_palette(default, len(levels))

    def ramp(self, lo: float, hi: float) -> list[str] | None:
        """Continuous stops; a diverging scale centres ``mid`` on ``midpoint``."""
        if self.kind != "continuous":
            return None
        if self.stops is not None:
            if self.positions is None:
                return list(self.stops)
            # gradientn(values=): stops at those places along the scale.
            rgb = np.array([_hex_rgb(c) for c in self.stops])
            where = np.asarray(self.positions, dtype=np.float64)
            grid = np.linspace(0.0, 1.0, 65)
            return [_rgb_hex([np.interp(u, where, rgb[:, k]) for k in range(3)]) for u in grid]
        if self.mid is None:
            return [to_hex(self.low), to_hex(self.high)]
        midpoint = 0.0 if self.midpoint is None else float(self.midpoint)
        span = max(hi - lo, 1e-12)
        centre = min(max((midpoint - lo) / span, 0.0), 1.0)
        out = []
        for i in range(65):
            t = i / 64
            if t <= centre:
                u = 0.5 * (t / centre) if centre > 0 else 0.5
            else:
                u = 0.5 + 0.5 * ((t - centre) / (1 - centre)) if centre < 1 else 1.0
            out.append(ramp_at([self.low, self.mid, self.high], u))
        return out

    def legend_entries(self, levels: list[str], colours: list[str]) -> list[dict]:
        order = self.breaks if self.breaks is not None else levels
        by_level = dict(zip(levels, colours))
        out = []
        for i, level in enumerate(order):
            if level not in by_level:
                continue
            label = level
            if isinstance(self.labels, dict):
                label = str(self.labels.get(level, level))
            elif isinstance(self.labels, (list, tuple)) and i < len(self.labels):
                label = str(self.labels[i])
            elif callable(self.labels):
                label = str(self.labels(level))
            out.append({"label": label, "color": by_level[level], "_level": level})
        return out


def _discrete(name, values=None, palette_fn=None, breaks=None, labels=None, na_value="#7f7f7f"):
    return ColourScale("discrete", name=name, values=values, palette_fn=palette_fn,
                       breaks=breaks, labels=labels, na_value=na_value)


def scale_colour_manual(values, *, breaks=None, labels=None, name=None, na_value="#7f7f7f"):
    """Your colours: a list in level order, or ``{"level": "colour"}``."""
    return _discrete(name, values=values, breaks=breaks, labels=labels, na_value=na_value)


def scale_colour_brewer(palette="Set1", *, direction=1, breaks=None, labels=None, name=None):
    """ColorBrewer palettes: Set1, Set2, Set3, Dark2, Paired, Pastel1, Pastel2,
    Accent (qualitative); Blues, Greens, Reds, Oranges, Purples, Greys, RdBu,
    PuOr (ordered)."""
    if palette not in _BREWER:
        raise ValueError(f"palette {palette!r} is not one of {sorted(_BREWER)}")
    stops = _BREWER[palette]

    def fn(n):
        if palette in _SEQUENTIAL:
            out = _sample(stops, n, 0.15 if n < len(stops) else 0.0, 1.0)
        else:
            out = extend_palette(stops, n)
        return out[::-1] if direction == -1 else out

    return _discrete(name, palette_fn=fn, breaks=breaks, labels=labels)


def scale_colour_okabe_ito(*, breaks=None, labels=None, name=None):
    """Okabe-Ito: eight colours safe for colour-blind readers."""
    return _discrete(name, palette_fn=lambda n: extend_palette(OKABE_ITO, n), breaks=breaks, labels=labels)


def scale_colour_viridis_d(option="viridis", *, begin=0.0, end=1.0, direction=1,
                           breaks=None, labels=None, name=None):
    """Evenly spaced colours from viridis (or magma, turbo) for categories."""
    from plot3.themes import _CONT_PALETTES

    if option not in _CONT_PALETTES:
        raise ValueError(f"option {option!r} is not one of {sorted(_CONT_PALETTES)}")
    stops = _CONT_PALETTES[option]

    def fn(n):
        out = _sample(stops, n, begin, end)
        return out[::-1] if direction == -1 else out

    return _discrete(name, palette_fn=fn, breaks=breaks, labels=labels)


def scale_colour_grey(start=0.2, end=0.8, *, breaks=None, labels=None, name=None):
    """Greys for black-and-white print (0 black, 1 white)."""
    def fn(n):
        return [_rgb_hex([start + (end - start) * (i / max(n - 1, 1))] * 3) for i in range(n)]

    return _discrete(name, palette_fn=fn, breaks=breaks, labels=labels)


def scale_colour_gradient(low="#132B43", high="#56B1F7", *, limits=None, name=None):
    """A continuous ramp from ``low`` to ``high`` (ggplot2's default blues)."""
    return ColourScale("continuous", name=name, low=low, high=high, limits=limits)


def scale_colour_gradient2(low="#832424", mid="#FFFFFF", high="#3A3A98", *, midpoint=0.0,
                           limits=None, name=None):
    """Diverging: ``low`` below ``midpoint``, ``mid`` at it, ``high`` above."""
    return ColourScale("continuous", name=name, low=low, mid=mid, high=high,
                       midpoint=midpoint, limits=limits)


def scale_colour_gradientn(colours=None, *, values=None, limits=None, name=None, colors=None):
    """A continuous ramp through several ``colours``; ``values`` (0..1, one
    per colour) places them along the scale."""
    stops = colours if colours is not None else colors
    if not stops or len(stops) < 2:
        raise ValueError("scale_colour_gradientn() needs at least two colours")
    if values is not None and len(values) != len(stops):
        raise ValueError("scale_colour_gradientn(values=) needs one value per colour")
    return ColourScale("continuous", name=name, stops=list(stops), positions=values, limits=limits)


def scale_colour_distiller(palette="Blues", *, direction=-1, limits=None, name=None):
    """A ColorBrewer palette stretched over a number (ggplot2's distiller).
    ``direction=-1`` (the default, as in ggplot2) puts the darkest colour at
    the low end."""
    if palette not in _BREWER:
        raise ValueError(f"palette {palette!r} is not one of {sorted(_BREWER)}")
    stops = list(_BREWER[palette])
    if direction == -1:
        stops = stops[::-1]
    return ColourScale("continuous", name=name, stops=stops, limits=limits)


def scale_colour_identity(*, name=None, na_value="#7f7f7f"):
    """Use the column's own colours ("red", "#1b9e77"), with no legend."""
    return ColourScale("discrete", name=name, identity=True, na_value=na_value)


def hcl_hex(h: float, c: float, l: float) -> str:
    """R's hcl(): polar CIE-LUV (D65) to sRGB, out-of-gamut values clipped."""

    xn, yn, zn = 95.047, 100.0, 108.883
    un = 4 * xn / (xn + 15 * yn + 3 * zn)
    vn = 9 * yn / (xn + 15 * yn + 3 * zn)
    if l <= 0:
        return "#000000"
    u = c * math.cos(math.radians(h))
    v = c * math.sin(math.radians(h))
    y = yn * (((l + 16) / 116) ** 3 if l > 8 else l / (24389 / 27))
    up, vp = u / (13 * l) + un, v / (13 * l) + vn
    x = 9.0 * y * up / (4 * vp)
    z = -x / 3 - 5 * y + 3 * y / vp
    x, y, z = x / 100, y / 100, z / 100
    lin = (3.240479 * x - 1.537150 * y - 0.498535 * z,
           -0.969256 * x + 1.875992 * y + 0.041556 * z,
           0.055648 * x - 0.204043 * y + 1.057311 * z)

    def gamma(ch):
        ch = min(max(ch, 0.0), 1.0)
        return 12.92 * ch if ch <= 0.0031308 else 1.055 * ch ** (1 / 2.4) - 0.055

    return "#" + "".join(f"{int(round(gamma(ch) * 255)):02x}" for ch in lin)


def scale_colour_hue(*, h=(15, 375), c=100, l=65, h_start=0, direction=1,
                     breaks=None, labels=None, name=None):
    """ggplot2's default discrete colours: evenly spaced hues at one
    chroma and lightness (#F8766D, #00BA38, #619CFF for three groups)."""
    lo, hi = float(h[0]), float(h[1])

    def fn(n):
        top = hi - 360.0 / n if (hi - lo) % 360 < 1 else hi
        hues = [lo + (top - lo) * i / max(n - 1, 1) for i in range(n)] if n > 1 else [lo]
        hues = [(value + h_start) % 360 for value in hues]
        out = [hcl_hex(value, c, l) for value in hues]
        return out[::-1] if direction == -1 else out

    return _discrete(name, palette_fn=fn, breaks=breaks, labels=labels)


scale_fill_hue = scale_colour_hue
scale_color_hue = scale_colour_hue
scale_fill_gradientn = scale_colour_gradientn
scale_color_gradientn = scale_colour_gradientn
scale_fill_distiller = scale_colour_distiller
scale_color_distiller = scale_colour_distiller
scale_fill_identity = scale_colour_identity
scale_color_identity = scale_colour_identity
scale_fill_manual = scale_colour_manual
scale_fill_brewer = scale_colour_brewer
scale_fill_okabe_ito = scale_colour_okabe_ito
scale_fill_viridis_d = scale_colour_viridis_d
scale_fill_grey = scale_colour_grey
scale_fill_gradient = scale_colour_gradient
scale_fill_gradient2 = scale_colour_gradient2
scale_color_manual = scale_colour_manual
scale_color_brewer = scale_colour_brewer
scale_color_okabe_ito = scale_colour_okabe_ito
scale_color_viridis_d = scale_colour_viridis_d
scale_color_grey = scale_colour_grey
scale_color_gradient = scale_colour_gradient
scale_color_gradient2 = scale_colour_gradient2


# ── shape and linetype ───────────────────────────────────────────────────────


class KeyScale:
    """Manual values for aes(shape=) or aes(linetype=)."""

    def __init__(self, aesthetic, values, breaks=None, labels=None, name=None):
        self.aesthetic = aesthetic
        self.values = list(values)
        self.breaks = None if breaks is None else [str(b) for b in breaks]
        self.labels = labels
        self.name = name
        if aesthetic == "shape":
            from plot3.geoms import shape_name

            self.values = [shape_name(v) for v in self.values]
        else:
            from plot3.geoms import dash_pattern

            for v in self.values:
                dash_pattern(v)

    def value_for(self, index: int, level: str):
        order = self.breaks or []
        if level in order and order.index(level) < len(self.values):
            return self.values[order.index(level)]
        return self.values[index % len(self.values)]


def scale_shape_manual(values, *, breaks=None, labels=None, name=None):
    """Symbols per level: names ("triangle") or R numbers (17)."""
    return KeyScale("shape", values, breaks, labels, name)


def scale_linetype_manual(values, *, breaks=None, labels=None, name=None):
    """Dash patterns per level: "solid", "dashed", "dotted", … or hex "44"."""
    return KeyScale("linetype", values, breaks, labels, name)


# ── size and alpha ───────────────────────────────────────────────────────────


class SizeScale:
    """How a number maps to point size: by area across a ``range``
    (scale_size), or by area from zero (scale_size_area)."""

    def __init__(self, kind, *, range=None, max_size=None, limits=None, breaks=None, name=None):
        self.kind = kind  # "range" | "area"
        self.range = None if range is None else (float(range[0]), float(range[1]))
        self.max_size = None if max_size is None else float(max_size)
        self.limits = None if limits is None else (float(limits[0]), float(limits[1]))
        self.breaks = None if breaks is None else [float(b) for b in breaks]
        self.name = name
        if self.range is not None and not (0 <= self.range[0] <= self.range[1] and self.range[1] > 0):
            raise ValueError("scale_size(range=) is (smallest, largest), for example (4, 23)")


def scale_size(name=None, *, range=(4.0, 23.0), limits=None, breaks=None):
    """Point area across ``range``: the smallest value gets the first size,
    the largest the second (ggplot2's default size scale). Sizes are in the
    units of ``geom_point(size=)`` (pixels in 2D); (4, 23) is ggplot2's
    ``range = c(1, 6)``."""
    return SizeScale("range", range=range, limits=limits, breaks=breaks, name=name)


def scale_size_area(name=None, *, max_size=23.0, breaks=None):
    """Point area in proportion to the value, zero at zero (plot3's default,
    with ``max_size`` the largest point)."""
    return SizeScale("area", max_size=max_size, breaks=breaks, name=name)


class AlphaScale:
    """How a number maps to opacity: across ``range`` (0..1)."""

    def __init__(self, *, range=(0.1, 1.0), limits=None, name=None):
        lo, hi = float(range[0]), float(range[1])
        if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0):
            raise ValueError("scale_alpha(range=) values are opacities from 0 to 1")
        self.range = (lo, hi)
        self.limits = None if limits is None else (float(limits[0]), float(limits[1]))
        self.name = name


def scale_alpha(name=None, *, range=(0.1, 1.0), limits=None):
    """Opacity for ``aes(alpha=)``: the smallest value is ``range[0]``, the
    largest ``range[1]``, as in ggplot2."""
    return AlphaScale(range=range, limits=limits, name=name)


scale_alpha_continuous = scale_alpha


__all__ = [
    "scale_x_continuous", "scale_y_continuous", "scale_x_reverse", "scale_y_reverse",
    "scale_x_discrete", "scale_y_discrete", "scale_x_date", "scale_y_date",
    "scale_x_datetime", "scale_y_datetime", "xlim", "ylim", "lims",
    "scale_colour_manual", "scale_fill_manual", "scale_color_manual",
    "scale_colour_brewer", "scale_fill_brewer", "scale_color_brewer",
    "scale_colour_okabe_ito", "scale_fill_okabe_ito", "scale_color_okabe_ito",
    "scale_colour_viridis_d", "scale_fill_viridis_d", "scale_color_viridis_d",
    "scale_colour_grey", "scale_fill_grey", "scale_color_grey",
    "scale_colour_gradient", "scale_fill_gradient", "scale_color_gradient",
    "scale_colour_gradient2", "scale_fill_gradient2", "scale_color_gradient2",
    "scale_shape_manual", "scale_linetype_manual",
    "scale_colour_gradientn", "scale_fill_gradientn", "scale_color_gradientn",
    "scale_colour_distiller", "scale_fill_distiller", "scale_color_distiller",
    "scale_colour_identity", "scale_fill_identity", "scale_color_identity",
    "scale_size", "scale_size_area", "scale_alpha", "scale_alpha_continuous",
    "scale_colour_hue", "scale_fill_hue", "scale_color_hue",
]
