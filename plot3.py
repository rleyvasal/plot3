"""plot3 — grammar-of-graphics plotting on three.js for SolveIt / Jupyter.

ggplot2-style API (https://github.com/tidyverse/ggplot2), rendered as a
self-contained WebGL figure that survives sslive slide export::

    from plot3 import ggplot, aes, geom_point, geom_line, labs, theme_light

    (ggplot(df, aes(x="time", y="temp", colour="sensor"))
     + geom_point(size=4)
     + geom_line(linewidth=2)
     + labs(title="Sensor temps", y="degC"))

- ``aes(z=...)`` switches the figure to a 3D orbit viewer.
- 2D figures pan (drag) and zoom (wheel, cursor-anchored); hover shows values.
- Payloads are uint16-quantized + delta + byte-shuffle + gzip, decoded in the
  browser via DecompressionStream. ``ggplot(..., quantize=False)`` ships raw
  float32 for deep-zoom (>60x) fidelity.
- ``read_bin(path)`` loads (N, stride) float32 point-cloud files (nuScenes
  .pcd.bin etc.) into a DataFrame; ``remote=True`` streams via CRAFT's SSH hop.
- ``%plot3 df x=a y=b color=c`` snapshots a DataFrame living on the CRAFT GPU
  kernel (columns only) and plots it host-side; without CRAFT it evaluates in
  the local namespace. The cell is red-eyed out of LLM context (``hide=0`` to
  keep it).

ggplot2 conventions honored: ``+`` layering, ``colour``/``color`` both accepted,
``geom_line(linewidth=)``, ``geom_path`` preserves data order while ``geom_line``
sorts by x, ``ggsave()``. Categorical hues come from a fixed-order validated
palette (never cycled; >8 categories is an error, not an 11th hue).

Data may also arrive through Python's ``>>`` operator.  ``ggplot(aes(...))``
creates a deferred figure, layers can be added normally, and ``data >> figure``
binds the data without mutating the reusable figure template.
"""

from __future__ import annotations

import base64
import copy
import gzip as _gzip
import html as _htmlesc
import json
import math
import shlex
import subprocess

import numpy as np
import pandas as pd

try:
    from IPython import get_ipython
except Exception:  # pragma: no cover - outside IPython
    get_ipython = None

__version__ = "0.2.0"

__all__ = [
    "ggplot",
    "aes",
    "geom_point",
    "geom_line",
    "geom_path",
    "geom_col",
    "geom_bar",
    "geom_histogram",
    "labs",
    "scale_colour_continuous",
    "scale_color_continuous",
    "scale_colour_viridis_c",
    "scale_color_viridis_c",
    "theme_dark",
    "theme_light",
    "ggsave",
    "read_bin",
    "autohide",
]

# ═════════════════════════════════════════════════════════════════════════════
# Palette (validated: light on #fcfcfb, dark on #0b1020 — dataviz reference)
# ═════════════════════════════════════════════════════════════════════════════

_CAT_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
              "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_CAT_DARK = ["#3987e5", "#199e70", "#c98500", "#008300",
             "#9085e9", "#e66767", "#d55181", "#d95926"]
# Sequential blue ramp, steps 100..700 (light -> dark). Light mode uses it as
# given (low=light); dark mode reversed so "near zero" recedes to the surface.
_SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]

# Perceptually-ordered scientific colormaps (exact matplotlib 16-stop hex) —
# multi-hue but monotonic in lightness, CVD-safe; the lidar/heatmap choice.
# Same stops in both themes (standard practice for viridis-family maps).
_VIRIDIS = ["#440154", "#481a6c", "#472f7d", "#414487", "#39568c", "#31688e",
            "#2a788e", "#23888e", "#1f988b", "#22a884", "#35b779", "#54c568",
            "#7ad151", "#a5db36", "#d2e21b", "#fde725"]
_MAGMA = ["#000004", "#0b0924", "#20114b", "#3b0f70", "#57157e", "#721f81",
          "#8c2981", "#a8327d", "#c43c75", "#de4968", "#f1605d", "#fa7f5e",
          "#fe9f6d", "#febf84", "#fddea0", "#fcfdbf"]
_TURBO = ["#30123b", "#4143a7", "#4771e9", "#3e9bfe", "#22c5e2", "#1ae4b6",
          "#46f884", "#88ff4e", "#b9f635", "#e1dd37", "#faba39", "#fd8d27",
          "#f05b12", "#d63506", "#af1801", "#7a0403"]
_CONT_PALETTES = {"viridis": _VIRIDIS, "magma": _MAGMA, "turbo": _TURBO}

_THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", cat=_CAT_LIGHT, seq=_SEQ,
    ),
    "dark": dict(
        surface="#0b1020", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#1c2742", axis="#2e3a5c", cat=_CAT_DARK,
        seq=list(reversed(_SEQ)),
    ),
}

# ═════════════════════════════════════════════════════════════════════════════
# Payload packing (same scheme proven in pcviz embed)
# ═════════════════════════════════════════════════════════════════════════════


def _delta_u16(a: np.ndarray) -> np.ndarray:
    """Column-wise delta mod 2^16 (lossless; tiny values on ordered data)."""
    d = a.astype(np.int32)
    d[1:] = (d[1:] - d[:-1]) % 65536
    return d.astype("<u2")


def _pack_u16(q: np.ndarray, compress: bool) -> str:
    """uint16 array -> base64; compressed = delta + byte-plane shuffle + gzip."""
    if compress:
        d = _delta_u16(q).view(np.uint8).reshape(-1, 2)
        raw = _gzip.compress(
            np.ascontiguousarray(d[:, 0]).tobytes()
            + np.ascontiguousarray(d[:, 1]).tobytes(),
            6,
        )
    else:
        raw = np.ascontiguousarray(q, dtype="<u2").tobytes()
    return base64.b64encode(raw).decode("ascii")


def _pack_f32(v: np.ndarray, compress: bool) -> str:
    raw = np.ascontiguousarray(v, dtype="<f4").tobytes()
    if compress:
        raw = _gzip.compress(raw, 6)
    return base64.b64encode(raw).decode("ascii")


def _encode_norm(v: np.ndarray, lo: float, hi: float, *, quantize: bool,
                 compress: bool) -> dict:
    """Encode values normalized to [0,1] over [lo,hi] (u16 or f32)."""
    span = (hi - lo) or 1.0
    t = (np.asarray(v, dtype=np.float64) - lo) / span
    if quantize:
        q = np.round(np.clip(t, 0.0, 1.0) * 65535.0).astype("<u2")
        return {"dtype": "u16", "b64": _pack_u16(q, compress)}
    return {"dtype": "f32", "b64": _pack_f32(t.astype(np.float32), compress)}


def _encode_codes(codes: np.ndarray, compress: bool) -> dict:
    q = np.ascontiguousarray(codes, dtype="<u2")
    return {"dtype": "u16", "b64": _pack_u16(q, compress), "raw": True}


# ═════════════════════════════════════════════════════════════════════════════
# Scales & ticks (Python side: 3D static ticks, datetime ladders, cat labels)
# ═════════════════════════════════════════════════════════════════════════════


def _nice_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    t0 = math.ceil(lo / step) * step
    out = []
    t = t0
    while t <= hi + step * 1e-9:
        out.append(0.0 if abs(t) < step * 1e-9 else t)
        t += step
    return out


def _fmt_num(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e6 or a < 1e-4:
        return f"{v:.3g}"
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s


_DT_LADDERS = [
    # (span_seconds >, [(pandas freq, strftime fmt), coarse -> fine])
    (2 * 365 * 86400, [("YS", "%Y"), ("QS", "%b %Y"), ("MS", "%b %Y")]),
    (90 * 86400, [("MS", "%b %Y"), ("W", "%b %d"), ("D", "%b %d")]),
    (3 * 86400, [("D", "%b %d"), ("6h", "%d %Hh"), ("h", "%H:%M")]),
    (3 * 3600, [("h", "%H:%M"), ("15min", "%H:%M"), ("min", "%H:%M")]),
    (0, [("min", "%H:%M"), ("15s", "%H:%M:%S"), ("s", "%H:%M:%S")]),
]


def _dt_ladder(lo_s: float, hi_s: float) -> list[list[list]]:
    """3-level [position_seconds, label] ladders; JS picks by visible count."""
    span = hi_s - lo_s
    for min_span, freqs in _DT_LADDERS:
        if span > min_span:
            break
    lo_ts = pd.Timestamp(lo_s, unit="s")
    hi_ts = pd.Timestamp(hi_s, unit="s")
    ladder = []
    for freq, fmt in freqs:
        try:
            idx = pd.date_range(lo_ts.floor("s"), hi_ts.ceil("s"), freq=freq)
        except Exception:
            idx = pd.DatetimeIndex([lo_ts, hi_ts])
        if len(idx) > 400:
            idx = idx[:: len(idx) // 400 + 1]
        ladder.append(
            [[t.timestamp(), t.strftime(fmt)] for t in idx]
        )
    return ladder


class _Scale:
    """Resolved positional scale: numeric, datetime or categorical."""

    def __init__(self, kind: str):
        self.kind = kind  # "num" | "dt" | "cat"
        self.lo = math.inf
        self.hi = -math.inf
        self.cats: list[str] = []

    def widen(self, values: np.ndarray):
        if len(values):
            self.lo = min(self.lo, float(np.nanmin(values)))
            self.hi = max(self.hi, float(np.nanmax(values)))

    def finish(self):
        if self.kind == "cat":
            self.lo, self.hi = -0.5, max(0.5, len(self.cats) - 0.5)
        elif not math.isfinite(self.lo):
            self.lo, self.hi = 0.0, 1.0
        elif self.hi <= self.lo:
            self.lo, self.hi = self.lo - 0.5, self.hi + 0.5

    def spec(self) -> dict:
        d = {"kind": self.kind, "lo": self.lo, "hi": self.hi}
        if self.kind == "cat":
            d["cats"] = self.cats
        elif self.kind == "dt":
            d["ladder"] = _dt_ladder(self.lo, self.hi)
        else:
            d["ticks"] = [[t, _fmt_num(t)] for t in _nice_ticks(self.lo, self.hi)]
        return d


def _col_values(s: pd.Series) -> tuple[str, np.ndarray, list[str]]:
    """Series -> (scale kind, float64 positions, categories)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dtype, "tz", None) is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        # normalize the unit: pandas 3.0 defaults to us, not ns
        v = s.astype("datetime64[ns]").astype("int64").to_numpy(np.float64)
        return "dt", v / 1e9, []
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "cat", s.cat.codes.to_numpy(np.float64), [str(c) for c in s.cat.categories]
    if pd.api.types.is_numeric_dtype(s):
        return "num", s.to_numpy(np.float64), []
    cats = sorted(s.dropna().astype(str).unique().tolist())  # ggplot2 sorts
    idx = {c: i for i, c in enumerate(cats)}
    return "cat", s.astype(str).map(idx).to_numpy(np.float64), cats


# ═════════════════════════════════════════════════════════════════════════════
# Grammar objects
# ═════════════════════════════════════════════════════════════════════════════


class aes(dict):
    """Aesthetic mapping: aes(x=, y=, z=, colour=/color=, group=)."""

    def __init__(self, x=None, y=None, z=None, color=None, colour=None,
                 group=None):
        super().__init__()
        for k, v in (("x", x), ("y", y), ("z", z),
                     ("color", color if color is not None else colour),
                     ("group", group)):
            if v is not None:
                self[k] = v


class _Geom:
    kind = ""
    sort_x = False

    def __init__(self, mapping: aes | None = None, *, color=None, colour=None,
                 alpha=None, **params):
        self.mapping = mapping or aes()
        self.const_color = color if color is not None else colour
        self.alpha = alpha
        self.params = params


class geom_point(_Geom):
    kind = "point"

    def __init__(self, mapping=None, *, size=None, **kw):
        super().__init__(mapping, **kw)
        self.size = size


class geom_path(_Geom):
    kind = "line"
    sort_x = False  # ggplot2 geom_path: connect in data order

    def __init__(self, mapping=None, *, linewidth=None, width=None, **kw):
        super().__init__(mapping, **kw)
        self.linewidth = linewidth if linewidth is not None else (width or 2.0)


class geom_line(geom_path):
    sort_x = True  # ggplot2 geom_line: connect in order of x


class geom_col(_Geom):
    """Bars with heights from ``y`` (ggplot2 ``geom_col``).

    Requires ``aes(x=, y=)``. ``x`` may be categorical or numeric. Optional
    ``width`` is the bar width as a fraction of the median x-spacing (default
    0.9). 2D only.
    """

    kind = "col"

    def __init__(self, mapping=None, *, width=0.9, **kw):
        super().__init__(mapping, **kw)
        self.width = float(width)
        self.data_override = None  # optional layer-local frame (stats)


class geom_bar(_Geom):
    """Count bars for a discrete ``x`` (ggplot2 ``geom_bar``).

    Only ``aes(x=)`` is required; counts become ``y``. Expanded to
    ``geom_col`` at build time. 2D only.
    """

    kind = "bar"

    def __init__(self, mapping=None, *, width=0.9, **kw):
        super().__init__(mapping, **kw)
        self.width = float(width)


class geom_histogram(_Geom):
    """Histogram of a continuous ``x`` (ggplot2 ``geom_histogram``).

    Only ``aes(x=)`` is required. Bins are computed in Python and drawn as
    ``geom_col``. 2D only.
    """

    kind = "histogram"

    def __init__(self, mapping=None, *, bins=30, width=1.0, **kw):
        super().__init__(mapping, **kw)
        self.bins = int(bins)
        self.width = float(width)


class labs(dict):
    def __init__(self, title=None, x=None, y=None, z=None, color=None,
                 colour=None):
        super().__init__()
        for k, v in (("title", title), ("x", x), ("y", y), ("z", z),
                     ("color", color if color is not None else colour)):
            if v is not None:
                self[k] = v


class scale_colour_continuous:
    """Numeric colour scale control.

    trans:   "linear" | "sqrt" | "log10"
    limits:  (lo, hi) tuple, or "full" for the data min/max.
             Default (no scale added) is robust 2nd-98th percentile limits —
             skewed data (lidar intensity!) stays readable; values outside
             the limits clamp to the ramp ends.
    palette: "blue" (theme single-hue default) | "viridis" | "magma" | "turbo"
    """

    def __init__(self, trans="linear", limits=None, palette="blue"):
        if trans not in ("linear", "sqrt", "log10"):
            raise ValueError("trans must be linear, sqrt or log10")
        if palette != "blue" and palette not in _CONT_PALETTES:
            raise ValueError(
                f"palette must be blue or one of {sorted(_CONT_PALETTES)}")
        self.trans = trans
        self.limits = limits
        self.palette = palette


scale_color_continuous = scale_colour_continuous


class scale_colour_viridis_c(scale_colour_continuous):
    """ggplot2-style viridis continuous scale: option viridis|magma|turbo."""

    def __init__(self, option="viridis", trans="linear", limits=None):
        super().__init__(trans=trans, limits=limits, palette=option)


scale_color_viridis_c = scale_colour_viridis_c


class _Theme:
    def __init__(self, name: str):
        self.name = name


def theme_dark() -> _Theme:
    return _Theme("dark")


def theme_light() -> _Theme:
    return _Theme("light")


class ggplot:
    """A plot3 figure, optionally deferred until data arrives via ``>>``."""

    def __init__(self, data=None, mapping: aes | None = None, *,
                 height="480px", quantize=True, compress=True, hide=None):
        # ``ggplot(aes(...))`` is the R-shaped, pipeable form.  ``aes`` is a
        # dict subclass, so detect it before treating arbitrary mappings as
        # dataframe constructor input.
        if isinstance(data, aes) and mapping is None:
            data, mapping = None, data
        self.data = self._as_pandas(data) if data is not None else None
        self.mapping = mapping or aes()
        self.layers: list[_Geom] = []
        self.labs: dict = {}
        self.theme_name = "dark"
        self.cscale: scale_colour_continuous | None = None
        self.height = height if isinstance(height, str) else f"{int(height)}px"
        self.quantize = bool(quantize)
        self.compress = bool(compress)
        self.hide = hide  # None -> module default (autohide())

    @staticmethod
    def _as_pandas(data) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        to_pandas = getattr(data, "to_pandas", None)
        if callable(to_pandas):
            data = to_pandas()
            if isinstance(data, pd.DataFrame):
                return data
        return pd.DataFrame(data)

    def __rrshift__(self, data):
        """Bind data to a deferred ``ggplot(aes(...))`` template."""
        if self.data is not None:
            raise TypeError("cannot pipe data into a ggplot that already has data")
        g = copy.copy(self)
        g.layers = list(self.layers)
        g.labs = dict(self.labs)
        g.data = self._as_pandas(data)
        return g

    def __add__(self, other):
        g = copy.copy(self)
        g.layers = list(self.layers)
        g.labs = dict(self.labs)
        if isinstance(other, _Geom):
            g.layers.append(other)
        elif isinstance(other, labs):
            g.labs.update(other)
        elif isinstance(other, _Theme):
            g.theme_name = other.name
        elif isinstance(other, scale_colour_continuous):
            g.cscale = other
        elif isinstance(other, aes):
            m = aes()
            m.update(self.mapping)
            m.update(other)
            g.mapping = m
        else:
            raise TypeError(f"cannot add {type(other).__name__!r} to ggplot")
        return g

    # rendering --------------------------------------------------------------

    def _repr_html_(self) -> str:
        html = self._iframe()
        # SolveIt: big viewer HTML must not enter LLM context (~800K chars
        # window) — red-eye the displaying cell unless opted out.
        if self.hide if self.hide is not None else _AUTOHIDE:
            try:
                _hide_caller_from_ai()
            except Exception:
                pass
        return html

    def html(self) -> str:
        """The full standalone document (what the iframe srcdoc carries)."""
        return _build_doc(self)

    def save(self, path: str) -> str:
        doc = self.html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        print(f"plot3: saved {path} ({len(doc) // 1024} KB)")
        return path

    def _iframe(self) -> str:
        doc = self.html()
        title = self.labs.get("title", "plot3 figure")
        return (
            f'<iframe srcdoc="{_htmlesc.escape(doc, quote=True)}" '
            f'style="width:100%;height:{self.height};border:0;'
            f'border-radius:6px;background:{_THEMES[self.theme_name]["surface"]}" '
            f'title="{_htmlesc.escape(str(title))}"></iframe>'
        )


_AUTOHIDE = True


def autohide(on: bool = True) -> None:
    """Default hide-from-AI behavior for displayed figures (SolveIt red eye)."""
    global _AUTOHIDE
    _AUTOHIDE = bool(on)


def ggsave(filename, plot: ggplot | None = None, **_kw) -> str:
    """ggsave("fig.html", p) — ggplot2-style save (HTML only)."""
    if isinstance(filename, ggplot) and isinstance(plot, str):
        filename, plot = plot, filename  # tolerate swapped args
    if plot is None:
        raise ValueError("ggsave(filename, plot) needs the plot")
    return plot.save(filename)


# ═════════════════════════════════════════════════════════════════════════════
# Figure build: grammar -> spec + payloads -> document
# ═════════════════════════════════════════════════════════════════════════════


def _expand_stat_geom(geom: _Geom, base_mapping: aes, data: pd.DataFrame) -> _Geom:
    """Turn bar/histogram stats into concrete ``geom_col`` layers."""
    mapping = dict(base_mapping)
    mapping.update(geom.mapping)
    if geom.kind == "bar":
        if "x" not in mapping:
            raise ValueError("geom_bar() requires aes(x=)")
        xcol = mapping["x"]
        if xcol not in data.columns:
            raise KeyError(f"column(s) not in DataFrame: {[xcol]}")
        counts = (
            data.groupby(xcol, dropna=False, observed=True, sort=False)
            .size()
            .rename("y")
            .reset_index()
        )
        out = geom_col(aes(x=xcol, y="y"), width=getattr(geom, "width", 0.9),
                       color=geom.const_color, colour=None, alpha=geom.alpha)
        out.data_override = counts
        out.const_color = geom.const_color
        out.alpha = geom.alpha
        return out
    if geom.kind == "histogram":
        if "x" not in mapping:
            raise ValueError("geom_histogram() requires aes(x=)")
        xcol = mapping["x"]
        if xcol not in data.columns:
            raise KeyError(f"column(s) not in DataFrame: {[xcol]}")
        values = pd.to_numeric(data[xcol], errors="coerce").dropna().to_numpy(
            dtype=np.float64
        )
        if values.size == 0:
            frame = pd.DataFrame({"x": np.array([], dtype=float),
                                  "y": np.array([], dtype=float)})
            width = 1.0
        else:
            bins = max(1, int(getattr(geom, "bins", 30)))
            counts, edges = np.histogram(values, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            width = float(np.median(np.diff(edges))) if len(edges) > 1 else 1.0
            frame = pd.DataFrame({"x": centers, "y": counts.astype(np.float64)})
        out = geom_col(aes(x="x", y="y"), width=getattr(geom, "width", 1.0),
                       color=geom.const_color, alpha=geom.alpha)
        out.data_override = frame
        out.const_color = geom.const_color
        out.alpha = geom.alpha
        out._bar_width_data = width  # absolute data units
        return out
    return geom


def _build_spec(g: ggplot) -> tuple[dict, list[tuple[str, str]]]:
    if g.data is None:
        raise ValueError(
            "ggplot has no data; use ggplot(df, aes(...)) or "
            "pipe data with `data >> ggplot(aes(...))`"
        )
    if not g.layers:
        raise ValueError("add a geom: ggplot(df, aes(...)) + geom_point()")

    theme = _THEMES[g.theme_name]
    expanded = [
        _expand_stat_geom(geom, g.mapping, g.data) for geom in g.layers
    ]
    resolved = []  # per layer: (geom, mapping)
    for geom in expanded:
        m = dict(g.mapping)
        m.update(geom.mapping)
        if "x" not in m or "y" not in m:
            raise ValueError("aes(x=, y=) are required (bar/histogram supply y)")
        resolved.append((geom, m))

    is3d = any("z" in m for _, m in resolved)
    if is3d and not all("z" in m for _, m in resolved):
        raise ValueError("mix of 2D and 3D layers: every layer needs aes(z=)")
    if is3d and any(geom.kind == "col" for geom, _ in resolved):
        raise ValueError("geom_col/geom_bar/geom_histogram are 2D only")

    axes = ["x", "y", "z"] if is3d else ["x", "y"]
    scales: dict[str, _Scale] = {}
    color_scale = None  # ("num", lo, hi) | ("cat", cats)
    num_color_vals: list[np.ndarray] = []

    # Pass 1 — per-layer values + global scale domains
    layer_vals = []
    for geom, m in resolved:
        frame = getattr(geom, "data_override", None)
        if frame is None:
            frame = g.data
        cols = [m[a] for a in axes if a in m] + (
            [m["color"]] if "color" in m else []
        ) + ([m["group"]] if "group" in m else [])
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"column(s) not in DataFrame: {missing}")
        sub = frame[list(dict.fromkeys(cols))].dropna()
        vals = {}
        for a in axes:
            kind, v, cats = _col_values(sub[m[a]])
            sc = scales.get(a)
            if sc is None:
                sc = scales[a] = _Scale(kind)
            elif sc.kind != kind:
                raise ValueError(
                    f"aes {a}: layers disagree on scale type "
                    f"({sc.kind} vs {kind})"
                )
            if kind == "cat":
                # merge category sets, remap codes onto the union
                merged = list(dict.fromkeys(sc.cats + cats))
                remap = {cats.index(c) if c in cats else None: i
                         for i, c in enumerate(merged) if c in cats}
                v = np.array([remap.get(int(c), -1) for c in v], dtype=np.float64)
                sc.cats = merged
            else:
                sc.widen(v)
            vals[a] = v
        # Bars include the baseline at y=0 in the domain.
        if geom.kind == "col" and "y" in scales and scales["y"].kind == "num":
            scales["y"].widen(np.asarray([0.0], dtype=np.float64))
        if "color" in m:
            kind, cv, ccats = _col_values(sub[m["color"]])
            if kind == "cat" or (kind == "num" and ccats):
                if len(ccats) > len(theme["cat"]):
                    raise ValueError(
                        f"{len(ccats)} colour categories > {len(theme['cat'])} "
                        "palette slots — fold rare categories or map a number"
                    )
                if color_scale is None:
                    color_scale = ["cat", list(ccats)]
                else:
                    color_scale[1] = list(dict.fromkeys(color_scale[1] + ccats))
                vals["color"] = ("cat", cv, ccats)
            else:
                if color_scale is None:
                    color_scale = ["num", math.inf, -math.inf]
                color_scale[1] = min(color_scale[1], float(np.nanmin(cv)))
                color_scale[2] = max(color_scale[2], float(np.nanmax(cv)))
                num_color_vals.append(np.asarray(cv, dtype=np.float64))
                vals["color"] = ("num", cv, None)
        if "group" in m:
            _, gv, gcats = _col_values(sub[m["group"]])
            vals["group"] = (gv, gcats)
        layer_vals.append(vals)

    for a in axes:
        scales[a].finish()

    # Numeric colour limits: robust 2-98 percentile by default so skewed data
    # (lidar intensity) actually varies; override via scale_colour_continuous.
    num_color = None
    if color_scale is not None and color_scale[0] == "num":
        allv = np.concatenate(num_color_vals)
        cs = g.cscale or scale_colour_continuous()
        if cs.limits == "full":
            lo_c, hi_c = color_scale[1], color_scale[2]
        elif isinstance(cs.limits, (tuple, list)):
            lo_c, hi_c = float(cs.limits[0]), float(cs.limits[1])
        else:
            lo_c = float(np.nanpercentile(allv, 2))
            hi_c = float(np.nanpercentile(allv, 98))
        if hi_c <= lo_c:
            lo_c, hi_c = color_scale[1], color_scale[2]
        if hi_c <= lo_c:
            hi_c = lo_c + 1.0
        tf = {
            "linear": lambda a: a,
            "sqrt": lambda a: np.sqrt(np.maximum(a, 0.0)),
            "log10": lambda a: np.log10(np.maximum(a, 1e-12)),
        }[cs.trans]
        num_color = (lo_c, hi_c, cs.trans, tf)

    # Pass 2 — encode payloads per layer (quantized against the shared scales)
    payloads: list[tuple[str, str]] = []
    layer_specs = []
    for li, ((geom, m), vals) in enumerate(zip(resolved, layer_vals)):
        n = len(vals["x"])
        order = np.arange(n)
        group_vec = None
        if "group" in vals:
            group_vec = vals["group"][0]
        elif vals.get("color") and vals["color"][0] == "cat":
            group_vec = vals["color"][1]
        if geom.kind == "line":
            keys = []
            if group_vec is not None:
                keys.append(group_vec)
            if geom.sort_x:
                keys.append(vals["x"])
            if keys:
                order = np.lexsort(tuple(reversed(keys)))

        spec_l = {
            "kind": geom.kind,
            "n": int(n),
            "alpha": geom.alpha,
            "constColor": geom.const_color,
        }
        for a in axes:
            sc = scales[a]
            enc = _encode_norm(vals[a][order], sc.lo, sc.hi,
                               quantize=g.quantize, compress=g.compress)
            pid = f"p{li}{a}"
            payloads.append((pid, enc["b64"]))
            spec_l[a] = {"id": pid, "dtype": enc["dtype"]}

        if vals.get("color"):
            ckind, cv, _ = vals["color"]
            pid = f"p{li}c"
            if ckind == "cat":
                # remap onto the global cat list
                local = vals["color"][2]
                remap = {i: color_scale[1].index(c) for i, c in enumerate(local)}
                codes = np.array([remap.get(int(c), 0) for c in cv[order]],
                                 dtype="<u2")
                payloads.append((pid, _pack_u16(codes, g.compress)))
                spec_l["color"] = {"id": pid, "dtype": "u16", "kind": "cat"}
            else:
                lo_c, hi_c, _trans, tf = num_color
                cvt = tf(np.clip(cv[order], lo_c, hi_c))
                lo_t = float(tf(np.asarray(lo_c)))
                hi_t = float(tf(np.asarray(hi_c)))
                enc = _encode_norm(cvt, lo_t, hi_t, quantize=True,
                                   compress=g.compress)
                payloads.append((pid, enc["b64"]))
                spec_l["color"] = {"id": pid, "dtype": "u16", "kind": "num"}

        if geom.kind == "line":
            if group_vec is not None:
                gv = group_vec[order]
                cut = np.flatnonzero(np.diff(gv)) + 1
                starts = np.concatenate([[0], cut])
                counts = np.diff(np.concatenate([starts, [n]]))
                spec_l["groups"] = [[int(s), int(c)]
                                    for s, c in zip(starts, counts)]
            else:
                spec_l["groups"] = [[0, int(n)]]
            spec_l["linewidth"] = float(getattr(geom, "linewidth", 2.0))
        elif geom.kind == "col":
            # Bar width in normalized [0,1] x-space for the renderer.
            scx = scales["x"]
            span = max(scx.hi - scx.lo, 1e-12)
            if hasattr(geom, "_bar_width_data"):
                data_w = float(geom._bar_width_data) * float(
                    getattr(geom, "width", 1.0)
                )
            elif scx.kind == "cat":
                data_w = float(getattr(geom, "width", 0.9))
            else:
                xs = np.asarray(vals["x"][order], dtype=np.float64)
                if len(xs) >= 2:
                    gaps = np.diff(np.sort(np.unique(xs)))
                    step = float(np.median(gaps)) if len(gaps) else 1.0
                else:
                    step = span * 0.08
                data_w = step * float(getattr(geom, "width", 0.9))
            spec_l["width"] = float(np.clip(data_w / span, 1e-4, 1.0))
            scy = scales["y"]
            if scy.kind == "num":
                y_span = max(scy.hi - scy.lo, 1e-12)
                spec_l["y0"] = float(
                    np.clip((0.0 - scy.lo) / y_span, 0.0, 1.0)
                )
            else:
                spec_l["y0"] = 0.0
            if spec_l["alpha"] is None:
                spec_l["alpha"] = 0.9
        else:
            if geom.size is not None:
                spec_l["size"] = float(geom.size)
            elif is3d:
                # cube units; density-scaled so dense scans stay crisp
                spec_l["size"] = round(
                    min(0.02, max(0.0012,
                                  0.02 * (500.0 / max(1, n)) ** (1.0 / 3.0))), 5)
            else:
                spec_l["size"] = 6.0 if n <= 2000 else (4.0 if n <= 20000 else 2.5)
            if spec_l["alpha"] is None:
                spec_l["alpha"] = 0.85 if n <= 50000 else 0.6
        if spec_l["alpha"] is None:
            spec_l["alpha"] = 1.0
        layer_specs.append(spec_l)

    # color spec + legend
    cspec = {"kind": "none"}
    legend = None
    if color_scale is not None:
        if color_scale[0] == "cat":
            cats = color_scale[1]
            cspec = {"kind": "cat", "palette": theme["cat"][: len(cats)],
                     "cats": cats}
            legend = [{"label": c, "color": theme["cat"][i]}
                      for i, c in enumerate(cats)]
        else:
            pal = (g.cscale.palette if g.cscale else "blue")
            ramp = _CONT_PALETTES.get(pal, theme["seq"])
            cspec = {"kind": "num", "lo": num_color[0], "hi": num_color[1],
                     "trans": num_color[2], "ramp": ramp}

    base_map = dict(g.mapping)
    spec = {
        "v": 1,
        "is3d": is3d,
        "theme": theme,
        "labs": {
            "title": g.labs.get("title", ""),
            "x": g.labs.get("x", base_map.get("x", "x")),
            "y": g.labs.get("y", base_map.get("y", "y")),
            "z": g.labs.get("z", base_map.get("z", "z")) if is3d else "",
            "color": g.labs.get("color", base_map.get("color", "")),
        },
        "scales": {a: scales[a].spec() for a in axes},
        "color": cspec,
        "legend": legend,
        "layers": layer_specs,
        "gz": 1 if g.compress else 0,
    }
    return spec, payloads


def _build_doc(g: ggplot) -> str:
    spec, payloads = _build_spec(g)
    blocks = "\n".join(
        f'<script type="text/plain" id="{pid}">{b64}</script>'
        for pid, b64 in payloads
    )
    doc = (
        _DOC_TEMPLATE
        .replace("__SPEC__", json.dumps(spec, separators=(",", ":")))
        .replace("__PAYLOADS__", blocks)
    )
    kb = len(doc) // 1024
    rows = sum(sp["n"] for sp in spec["layers"])
    print(f"plot3: {len(spec['layers'])} layer(s), {rows:,} rows -> {kb:,} KB "
          f"portable HTML{' (3D)' if spec['is3d'] else ''}")
    if kb > 1500:
        print("plot3: warning — figure may exceed sslive's ~1.8 MB in-slide cap")
    return doc


# ═════════════════════════════════════════════════════════════════════════════
# The document template (viewer JS: shared decode; 2D pan/zoom; 3D orbit)
# ═════════════════════════════════════════════════════════════════════════════

_DOC_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"}}</script>
<style>
html,body{margin:0;height:100%;overflow:hidden;
  font:12px system-ui,-apple-system,"Segoe UI",sans-serif}
#fig{position:relative;width:100vw;height:100vh}
#title{position:absolute;left:14px;top:8px;font-size:14px;font-weight:600;z-index:4}
#canvas-host{position:absolute}
#axes{position:absolute;inset:0;pointer-events:none;z-index:2}
#legend{position:absolute;right:10px;top:8px;z-index:4;padding:6px 9px;
  border-radius:6px;font-size:11px;line-height:1.7}
#legend .sw{display:inline-block;width:9px;height:9px;border-radius:5px;
  margin-right:6px;vertical-align:-1px}
#legend .lg-e{cursor:pointer;user-select:none}
#tip{position:absolute;display:none;z-index:5;pointer-events:none;
  padding:4px 8px;border-radius:5px;font-size:11px;white-space:nowrap}
#hint{position:absolute;left:50%;bottom:46px;transform:translateX(-50%);
  z-index:5;pointer-events:none;padding:5px 10px;border-radius:5px;
  font-size:11px;opacity:0;transition:opacity .25s}
#ramp{height:8px;width:110px;border-radius:4px;margin-top:3px}
</style></head><body>
<div id="fig">
  <div id="title"></div>
  <div id="canvas-host"></div>
  <svg id="axes"></svg>
  <div id="legend" style="display:none"></div>
  <div id="tip"></div>
  <div id="hint"></div>
</div>
__PAYLOADS__
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

const S = __SPEC__;
const T = S.theme;
document.body.style.background = T.surface;
document.body.style.color = T.ink;

async function decode(id, dtype) {
  const node = document.getElementById(id);
  if (!node) return null;
  const s = atob(node.textContent.trim());
  let a = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
  if (S.gz) {
    const ds = new DecompressionStream('gzip');
    a = new Uint8Array(
      await new Response(new Blob([a]).stream().pipeThrough(ds)).arrayBuffer());
  }
  if (dtype === 'f32') return new Float32Array(a.buffer);
  let u;
  if (S.gz) {                       // undo byte planes + delta
    const m = a.length >> 1;
    u = new Uint16Array(m);
    for (let i = 0; i < m; i++) u[i] = a[i] | (a[m + i] << 8);
    for (let i = 1; i < m; i++) u[i] = (u[i] + u[i - 1]) & 0xffff;
  } else {
    u = new Uint16Array(a.buffer);
  }
  return u;
}
function toNorm(arr) {                       // u16 -> [0,1] f32 (f32 passes through)
  if (arr instanceof Float32Array) return arr;
  const f = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) f[i] = arr[i] / 65535;
  return f;
}
function hex2rgb(h) {
  return [parseInt(h.slice(1,3),16)/255, parseInt(h.slice(3,5),16)/255,
          parseInt(h.slice(5,7),16)/255];
}
const RAMP = (S.color.ramp || []).map(hex2rgb);
function rampAt(t) {
  const k = Math.min(RAMP.length - 1.001, Math.max(0, t * (RAMP.length - 1)));
  const i = Math.floor(k), f = k - i;
  const a = RAMP[i], b = RAMP[i + 1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}
const PAL = (S.color.palette || []).map(hex2rgb);

// ── payload decode for every layer ──────────────────────────────────────────
const axesList = S.is3d ? ['x','y','z'] : ['x','y'];
for (const L of S.layers) {
  for (const a of axesList) L[a].data = toNorm(await decode(L[a].id, L[a].dtype));
  if (L.color) L.color.data = await decode(L.color.id, 'u16');
}

// per-layer vertex colors (normalized cube space is built per-branch)
function layerColors(L, defRGB) {
  const n = L.n, c = new Float32Array(n * 3);
  if (L.constColor) defRGB = hex2rgb(L.constColor);
  if (!L.color) { for (let i=0;i<n;i++){c[i*3]=defRGB[0];c[i*3+1]=defRGB[1];c[i*3+2]=defRGB[2];} return c; }
  const d = L.color.data;
  if (L.color.kind === 'cat') {
    for (let i = 0; i < n; i++) { const p = PAL[d[i] % PAL.length];
      c[i*3]=p[0]; c[i*3+1]=p[1]; c[i*3+2]=p[2]; }
  } else {
    for (let i = 0; i < n; i++) { const p = rampAt(d[i] / 65535);
      c[i*3]=p[0]; c[i*3+1]=p[1]; c[i*3+2]=p[2]; }
  }
  return c;
}

// ── chrome: title + legend ──────────────────────────────────────────────────
const figEl = document.getElementById('fig');
const titleEl = document.getElementById('title');
if (S.labs.title) titleEl.textContent = S.labs.title;
const legEl = document.getElementById('legend');
// legend click-filtering: category index -> three.js objects
const hiddenCats = new Set();
const catObjs = new Map();
function regCat(ci, obj) {
  if (!catObjs.has(ci)) catObjs.set(ci, []);
  catObjs.get(ci).push(obj);
}
let redraw = () => {};   // 2D assigns its draw(); 3D renders continuously
window.__plot3 = { hiddenCats, catObjs };
if (S.legend) {
  legEl.style.display = 'block';
  legEl.style.background = T.surface + 'e6';
  legEl.style.border = '1px solid ' + T.grid;
  legEl.style.color = T.ink2;
  legEl.innerHTML = (S.labs.color ? '<b style="color:'+T.ink+'">' +
      S.labs.color + '</b>' : '') +
    S.legend.map((e, i) => '<div class="lg-e" data-ci="' + i +
      '"><span class="sw" style="background:' + e.color + '"></span>' +
      e.label + '</div>').join('');
  legEl.addEventListener('click', ev => {
    const row = ev.target.closest('.lg-e');
    if (!row) return;
    const ci = +row.dataset.ci;
    if (hiddenCats.has(ci)) hiddenCats.delete(ci); else hiddenCats.add(ci);
    row.style.opacity = hiddenCats.has(ci) ? 0.35 : 1;
    for (const o of (catObjs.get(ci) || [])) o.visible = !hiddenCats.has(ci);
    tip.style.display = 'none';
    redraw();
  });
} else if (S.color.kind === 'num') {
  legEl.style.display = 'block';
  legEl.style.background = T.surface + 'e6';
  legEl.style.border = '1px solid ' + T.grid;
  legEl.style.color = T.ink2;
  legEl.innerHTML = '<b style="color:'+T.ink+'">' + (S.labs.color||'') +
    '</b><div id="ramp" style="background:linear-gradient(90deg,' +
    S.color.ramp.join(',') + ')"></div>' +
    '<span style="float:left">' + (+S.color.lo.toPrecision(3)) + '</span>' +
    '<span style="float:right">' + (+S.color.hi.toPrecision(3)) + '</span>';
}

function fmt(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toPrecision(3);
  return String(+v.toFixed(6));
}
// numeric colour: normalized ramp position -> data value (inverse transform)
function cval(t) {
  const tr = S.color.trans || 'linear';
  const f = tr === 'sqrt' ? Math.sqrt
    : tr === 'log10' ? (v => Math.log10(Math.max(v, 1e-12))) : (v => v);
  const inv = tr === 'sqrt' ? (v => v * v)
    : tr === 'log10' ? (v => Math.pow(10, v)) : (v => v);
  return inv(f(S.color.lo) + t * (f(S.color.hi) - f(S.color.lo)));
}
function fmtAxis(ax, v) {                     // v in data units
  const sc = S.scales[ax];
  if (sc.kind === 'cat') {
    const i = Math.round(v);
    return (i >= 0 && i < sc.cats.length && Math.abs(v - i) < 0.26) ? sc.cats[i] : '';
  }
  if (sc.kind === 'dt') return new Date(v * 1000).toLocaleString();
  return fmt(v);
}

// nice numeric ticks (JS side for pan/zoom)
function niceTicks(lo, hi, n) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  let step = 10 * mag;
  for (const m of [1, 2, 5, 10]) if (raw <= m * mag) { step = m * mag; break; }
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step)
    out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  return out;
}
// ticks for any scale over a visible data range -> [[pos_data, label], ...]
function thin(vis, maxN) {
  if (vis.length <= maxN) return vis;
  const step = Math.ceil(vis.length / maxN);
  return vis.filter((_, i) => i % step === 0);
}
function ticksFor(ax, lo, hi) {
  const sc = S.scales[ax];
  if (sc.kind === 'cat')
    return thin(sc.cats.map((c, i) => [i, c])
      .filter(t => t[0] >= lo && t[0] <= hi), 12);
  if (sc.kind === 'dt') {
    for (const level of sc.ladder) {
      const vis = level.filter(t => t[0] >= lo && t[0] <= hi);
      if (vis.length >= 3 && vis.length <= 14) return vis;
    }
    let vis = sc.ladder[0].filter(t => t[0] >= lo && t[0] <= hi);
    if (vis.length < 3)
      vis = sc.ladder[sc.ladder.length - 1]
        .filter(t => t[0] >= lo && t[0] <= hi);
    return thin(vis, 10);
  }
  return niceTicks(lo, hi, 6).map(t => [t, fmt(t)]);
}
// hover value: enough digits to resolve ~1/300 of the visible span
function fmtSpan(ax, v, spanData) {
  const sc = S.scales[ax];
  if (sc.kind !== 'num') return fmtAxis(ax, v);
  const d = Math.max(0, Math.min(6,
    Math.ceil(-Math.log10(Math.max(1e-12, spanData / 300)))));
  return v.toFixed(d);
}
const dataLo = ax => S.scales[ax].lo, dataHi = ax => S.scales[ax].hi;
const spanOf = ax => (dataHi(ax) - dataLo(ax)) || 1;

const host = document.getElementById('canvas-host');
const svg = document.getElementById('axes');
const tip = document.getElementById('tip');
tip.style.background = T.surface;
tip.style.border = '1px solid ' + T.axis;
tip.style.color = T.ink;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const lineMats = [];

if (!S.is3d) {
  // ═════════════════════════ 2D: ortho + pan/zoom ═════════════════════════
  const M = { l: 58, r: 12, t: 30, b: 40 };
  let W = 100, H = 100;
  const cam = new THREE.OrthographicCamera(-0.03, 1.03, 1.03, -0.03, -10, 10);

  for (const L of S.layers) {
    const n = L.n;
    const cols = layerColors(L, hex2rgb(T.cat[0]));
    const isCat = L.color && L.color.kind === 'cat';
    if (L.kind === 'point') {
      if (isCat) {
        // one Points object per category -> legend click-filtering
        const k = S.color.cats.length;
        const buckets = Array.from({ length: k }, () => []);
        for (let i = 0; i < n; i++) buckets[L.color.data[i] % k].push(i);
        buckets.forEach((idx, ci) => {
          if (!idx.length) return;
          const pos = new Float32Array(idx.length * 3);
          for (let j = 0; j < idx.length; j++) {
            const i = idx[j];
            pos[j*3] = L.x.data[i]; pos[j*3+1] = L.y.data[i]; pos[j*3+2] = 0;
          }
          const g = new THREE.BufferGeometry();
          g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
          const pt = new THREE.Points(g, new THREE.PointsMaterial({
            color: S.color.palette[ci], size: L.size, sizeAttenuation: false,
            transparent: true, opacity: L.alpha }));
          scene.add(pt);
          regCat(ci, pt);
        });
      } else {
        const pos = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {
          pos[i*3] = L.x.data[i]; pos[i*3+1] = L.y.data[i]; pos[i*3+2] = 0;
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
        scene.add(new THREE.Points(g, new THREE.PointsMaterial({
          size: L.size, sizeAttenuation: false, vertexColors: true,
          transparent: true, opacity: L.alpha })));
      }
    } else if (L.kind === 'col') {
      // Axis-aligned bars from baseline y0 to y=height (normalized coords).
      const hw = (L.width || 0.08) * 0.5;
      const y0 = (L.y0 != null) ? L.y0 : 0;
      const pos = new Float32Array(n * 6 * 3);
      const col = new Float32Array(n * 6 * 3);
      let p = 0, c = 0;
      for (let i = 0; i < n; i++) {
        const x = L.x.data[i], y = L.y.data[i];
        const x0 = x - hw, x1 = x + hw;
        // two triangles: (x0,y0)-(x1,y0)-(x1,y) and (x0,y0)-(x1,y)-(x0,y)
        const tri = [x0,y0,0, x1,y0,0, x1,y,0, x0,y0,0, x1,y,0, x0,y,0];
        for (let k = 0; k < 18; k++) pos[p++] = tri[k];
        for (let k = 0; k < 6; k++) {
          col[c++] = cols[i*3]; col[c++] = cols[i*3+1]; col[c++] = cols[i*3+2];
        }
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      scene.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({
        vertexColors: true, transparent: true, opacity: L.alpha,
        side: THREE.DoubleSide, depthWrite: false })));
    } else {
      for (const [s0, cnt] of L.groups) {
        if (cnt < 2) continue;
        const flat = new Float32Array(cnt * 3);
        for (let i = 0; i < cnt; i++) {
          flat[i*3] = L.x.data[s0+i]; flat[i*3+1] = L.y.data[s0+i]; flat[i*3+2]=0;
        }
        const lg = new LineGeometry();
        lg.setPositions(Array.from(flat));
        const rgb = [cols[s0*3], cols[s0*3+1], cols[s0*3+2]];
        const lm = new LineMaterial({
          color: new THREE.Color(rgb[0], rgb[1], rgb[2]).getHex(),
          linewidth: L.linewidth, worldUnits: false,
          transparent: true, opacity: L.alpha });
        lineMats.push(lm);
        const ln = new Line2(lg, lm);
        scene.add(ln);
        if (isCat) regCat(L.color.data[s0] % S.color.cats.length, ln);
      }
    }
  }

  function drawAxes() {
    const x0 = dataLo('x') + cam.left  * spanOf('x');
    const x1 = dataLo('x') + cam.right * spanOf('x');
    const y0 = dataLo('y') + cam.bottom * spanOf('y');
    const y1 = dataLo('y') + cam.top    * spanOf('y');
    const px = v => M.l + (v - x0) / (x1 - x0) * W;
    const py = v => M.t + H - (v - y0) / (y1 - y0) * H;
    let s = '';
    // cap tick count by panel size so labels never collide
    const xt = thin(ticksFor('x', x0, x1), Math.max(3, Math.floor(W / 80)));
    const yt = thin(ticksFor('y', y0, y1), Math.max(3, Math.floor(H / 40)));
    for (const [t, lab] of xt) {
      const X = px(t);
      if (X < M.l - 1 || X > M.l + W + 1) continue;
      s += `<line x1="${X}" y1="${M.t}" x2="${X}" y2="${M.t+H}" stroke="${T.grid}"/>`;
      s += `<text x="${X}" y="${M.t+H+14}" fill="${T.muted}" text-anchor="middle">${lab}</text>`;
    }
    for (const [t, lab] of yt) {
      const Y = py(t);
      if (Y < M.t - 1 || Y > M.t + H + 1) continue;
      s += `<line x1="${M.l}" y1="${Y}" x2="${M.l+W}" y2="${Y}" stroke="${T.grid}"/>`;
      s += `<text x="${M.l-7}" y="${Y+4}" fill="${T.muted}" text-anchor="end">${lab}</text>`;
    }
    s += `<rect x="${M.l}" y="${M.t}" width="${W}" height="${H}" fill="none" stroke="${T.axis}"/>`;
    s += `<text x="${M.l+W/2}" y="${M.t+H+30}" fill="${T.ink2}" text-anchor="middle">${S.labs.x}</text>`;
    s += `<text x="14" y="${M.t+H/2}" fill="${T.ink2}" text-anchor="middle" transform="rotate(-90 14 ${M.t+H/2})">${S.labs.y}</text>`;
    svg.innerHTML = s;
  }

  function layout() {
    W = Math.max(50, figEl.clientWidth - M.l - M.r);
    H = Math.max(50, figEl.clientHeight - M.t - M.b);
    host.style.left = M.l + 'px'; host.style.top = M.t + 'px';
    renderer.setSize(W, H);
    svg.setAttribute('width', figEl.clientWidth);
    svg.setAttribute('height', figEl.clientHeight);
    for (const lm of lineMats) lm.resolution.set(W, H);
    draw();
  }
  let rafPending = false;
  function draw() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      cam.updateProjectionMatrix();
      renderer.render(scene, cam);
      drawAxes();
    });
  }
  redraw = draw;

  // pan / zoom / hover
  const el = renderer.domElement;
  el.style.touchAction = 'none';
  function clampView() {
    const MAX = 2.4, LO = -0.7, HI = 1.7;
    for (const [a, b] of [['left', 'right'], ['bottom', 'top']]) {
      let span = cam[b] - cam[a];
      if (span > MAX) {
        const c = (cam[a] + cam[b]) / 2;
        cam[a] = c - MAX / 2; cam[b] = c + MAX / 2;
      }
      if (cam[a] < LO) { cam[b] += LO - cam[a]; cam[a] = LO; }
      if (cam[b] > HI) { cam[a] -= cam[b] - HI; cam[b] = HI; }
    }
  }
  let dragging = null;
  el.addEventListener('pointerdown', e => {
    dragging = { x: e.clientX, y: e.clientY,
                 l: cam.left, r: cam.right, t: cam.top, b: cam.bottom };
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener('pointerup', () => dragging = null);
  el.addEventListener('pointermove', e => {
    if (dragging) {
      const dx = (e.clientX - dragging.x) / W * (dragging.r - dragging.l);
      const dy = (e.clientY - dragging.y) / H * (dragging.t - dragging.b);
      cam.left = dragging.l - dx; cam.right = dragging.r - dx;
      cam.top = dragging.t + dy;  cam.bottom = dragging.b + dy;
      clampView();
      tip.style.display = 'none';
      draw();
    } else hover(e);
  });
  const hintEl = document.getElementById('hint');
  hintEl.style.background = T.surface + 'e6';
  hintEl.style.border = '1px solid ' + T.axis;
  hintEl.style.color = T.ink2;
  let hintT = 0, hintOff = 0;
  el.addEventListener('wheel', e => {
    if (!e.ctrlKey && !e.metaKey) {
      // let the page scroll; nudge toward the modifier
      const now = performance.now();
      if (now - hintT > 1500) {
        hintT = now;
        hintEl.textContent = (navigator.platform || '').includes('Mac')
          ? 'Use \\u2318 + scroll to zoom' : 'Use Ctrl + scroll to zoom';
        hintEl.style.opacity = '1';
        clearTimeout(hintOff);
        hintOff = setTimeout(() => hintEl.style.opacity = '0', 1200);
      }
      return;
    }
    e.preventDefault();
    const f = Math.exp(e.deltaY * 0.0015);
    const r = el.getBoundingClientRect();
    const cx = cam.left + (e.clientX - r.left) / W * (cam.right - cam.left);
    const cy = cam.top - (e.clientY - r.top) / H * (cam.top - cam.bottom);
    cam.left = cx + (cam.left - cx) * f;   cam.right = cx + (cam.right - cx) * f;
    cam.top = cy + (cam.top - cy) * f;     cam.bottom = cy + (cam.bottom - cy) * f;
    clampView();
    tip.style.display = 'none';
    draw();
  }, { passive: false });
  el.addEventListener('dblclick', () => {
    cam.left = -0.03; cam.right = 1.03; cam.bottom = -0.03; cam.top = 1.03;
    draw();
  });

  let hoverTick = 0;
  function hover(e) {
    const now = performance.now();
    if (now - hoverTick < 33) return;
    hoverTick = now;
    const r = el.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bestD = 12 * 12;
    for (const L of S.layers) {
      const catL = L.color && L.color.kind === 'cat';
      for (let i = 0; i < L.n; i++) {
        if (catL && hiddenCats.has(L.color.data[i] % S.color.cats.length))
          continue;
        const sx = (L.x.data[i] - cam.left) / (cam.right - cam.left) * W;
        const sy = (cam.top - L.y.data[i]) / (cam.top - cam.bottom) * H;
        const d = (sx-mx)*(sx-mx) + (sy-my)*(sy-my);
        if (d < bestD) { bestD = d; best = [L, i, sx, sy]; }
      }
    }
    if (!best) { tip.style.display = 'none'; return; }
    const [L, i, sx, sy] = best;
    const xv = dataLo('x') + L.x.data[i] * spanOf('x');
    const yv = dataLo('y') + L.y.data[i] * spanOf('y');
    let head = '';
    if (L.color && L.color.kind === 'cat')
      head = '<b>' + S.color.cats[L.color.data[i] % S.color.cats.length] + '</b><br>';
    else if (L.color && L.color.kind === 'num')
      head = '<b>' + fmt(cval(L.color.data[i] / 65535)) + '</b><br>';
    tip.innerHTML = head
      + fmtSpan('x', xv, (cam.right - cam.left) * spanOf('x')) + ', '
      + fmtSpan('y', yv, (cam.top - cam.bottom) * spanOf('y'));
    tip.style.left = (M.l + sx + 12) + 'px';
    tip.style.top = (M.t + sy - 10) + 'px';
    tip.style.display = 'block';
  }
  el.addEventListener('pointerleave', () => tip.style.display = 'none');

  new ResizeObserver(layout).observe(figEl);
  layout();

} else {
  // ═════════════════════════ 3D: orbit viewer ═════════════════════════════
  host.style.left = '0'; host.style.top = '0';
  const cam = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
  cam.up.set(0, 0, 1);
  // proportional cube: preserve data aspect across axes
  const spans = axesList.map(a => spanOf(a));
  const maxSpan = Math.max(...spans);
  const ext = axesList.map((a, i) => spans[i] / maxSpan);
  const toCube = (a, i, v) => v * ext[i];

  for (const L of S.layers) {
    const n = L.n;
    const cols = layerColors(L, hex2rgb(T.cat[0]));
    const isCat = L.color && L.color.kind === 'cat';
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i*3]   = L.x.data[i] * ext[0];
      pos[i*3+1] = L.y.data[i] * ext[1];
      pos[i*3+2] = L.z.data[i] * ext[2];
    }
    if (L.kind === 'point') {
      if (isCat) {
        const k = S.color.cats.length;
        const buckets = Array.from({ length: k }, () => []);
        for (let i = 0; i < n; i++) buckets[L.color.data[i] % k].push(i);
        buckets.forEach((idx, ci) => {
          if (!idx.length) return;
          const sub = new Float32Array(idx.length * 3);
          for (let j = 0; j < idx.length; j++) {
            const i = idx[j];
            sub[j*3] = pos[i*3]; sub[j*3+1] = pos[i*3+1]; sub[j*3+2] = pos[i*3+2];
          }
          const g = new THREE.BufferGeometry();
          g.setAttribute('position', new THREE.BufferAttribute(sub, 3));
          const pt = new THREE.Points(g, new THREE.PointsMaterial({
            color: S.color.palette[ci], size: L.size, sizeAttenuation: true,
            transparent: true, opacity: L.alpha }));
          scene.add(pt);
          regCat(ci, pt);
        });
      } else {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
        scene.add(new THREE.Points(g, new THREE.PointsMaterial({
          size: L.size, sizeAttenuation: true, vertexColors: true,
          transparent: true, opacity: L.alpha })));
      }
    } else {
      for (const [s0, cnt] of L.groups) {
        if (cnt < 2) continue;
        const lg = new LineGeometry();
        lg.setPositions(Array.from(pos.subarray(s0*3, (s0+cnt)*3)));
        const lm = new LineMaterial({
          color: new THREE.Color(cols[s0*3], cols[s0*3+1], cols[s0*3+2]).getHex(),
          linewidth: L.linewidth, worldUnits: false,
          transparent: true, opacity: L.alpha });
        lineMats.push(lm);
        const ln = new Line2(lg, lm);
        scene.add(ln);
        if (isCat) regCat(L.color.data[s0] % S.color.cats.length, ln);
      }
    }
  }

  // axes box + static ticks (Python-computed) + labels as sprites
  const boxMat = new THREE.LineBasicMaterial({ color: T.axis });
  const bg = new THREE.BufferGeometry().setFromPoints([
    [0,0,0],[1,0,0],[1,0,0],[1,1,0],[1,1,0],[0,1,0],[0,1,0],[0,0,0],
    [0,0,1],[1,0,1],[1,0,1],[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,0,1],
    [0,0,0],[0,0,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],[0,1,0],[0,1,1],
  ].map(p => new THREE.Vector3(p[0]*ext[0], p[1]*ext[1], p[2]*ext[2])));
  scene.add(new THREE.LineSegments(bg, boxMat));

  function sprite(text, small) {
    const c = document.createElement('canvas');
    const ctx = c.getContext('2d');
    const fs = small ? 22 : 26;
    ctx.font = fs + 'px system-ui';
    c.width = Math.max(2, Math.ceil(ctx.measureText(text).width) + 8);
    c.height = fs + 10;
    const ctx2 = c.getContext('2d');
    ctx2.font = fs + 'px system-ui';
    ctx2.fillStyle = small ? T.muted : T.ink2;
    ctx2.textBaseline = 'middle';
    ctx2.fillText(text, 4, c.height / 2);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({
      map: tex, depthTest: false, transparent: true }));
    const k = small ? 0.0016 : 0.0019;
    sp.scale.set(c.width * k, c.height * k, 1);
    return sp;
  }
  function tickPos(ax, t) {                    // data -> cube coords on min edges
    return (t - dataLo(ax)) / spanOf(ax);
  }
  const off = 0.055;
  axesList.forEach((ax, ai) => {
    const sc = S.scales[ax];
    const ticks = sc.kind === 'cat'
      ? sc.cats.map((c, i) => [i, c])
      : (sc.ticks || (sc.ladder ? sc.ladder[0] : []));
    for (const [t, lab] of ticks) {
      const u = tickPos(ax, t);
      if (u < -0.001 || u > 1.001) continue;
      const p = [[u*ext[0], -off*ext[1], 0], [-off*ext[0], u*ext[1], 0],
                 [-off*ext[0], 0, u*ext[2]]][ai];
      const sp = sprite(String(lab), true);
      sp.position.set(p[0], p[1], p[2]);
      scene.add(sp);
    }
    const lp = [[0.5*ext[0], -2.6*off*ext[1], 0],
                [-2.6*off*ext[0], 0.5*ext[1], 0],
                [-2.6*off*ext[0], 0, 0.55*ext[2]]][ai];
    const tl = sprite(S.labs[ax], false);
    tl.position.set(lp[0], lp[1], lp[2]);
    scene.add(tl);
  });

  const ctr = new THREE.Vector3(ext[0]/2, ext[1]/2, ext[2]/2);
  // fit: place the camera so the cube's bounding sphere fills the frustum
  const rad = Math.sqrt(ext[0]*ext[0] + ext[1]*ext[1] + ext[2]*ext[2]) / 2;
  const dist = rad / Math.tan((cam.fov * Math.PI / 180) / 2) * 1.45;
  const dir = new THREE.Vector3(0.55, -0.85, 0.5).normalize();
  cam.position.copy(ctr.clone().add(dir.multiplyScalar(dist)));
  cam.near = dist / 100;
  cam.far = dist * 20;
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.target.copy(ctr);
  controls.enableDamping = true;

  function layout() {
    const w = Math.max(figEl.clientWidth, 1), h = Math.max(figEl.clientHeight, 1);
    renderer.setSize(w, h);
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
    for (const lm of lineMats) lm.resolution.set(w, h);
  }
  new ResizeObserver(layout).observe(figEl);
  layout();
  (function loop() {
    controls.update();
    renderer.render(scene, cam);
    requestAnimationFrame(loop);
  })();
}
</script>
</body></html>"""


# ═════════════════════════════════════════════════════════════════════════════
# read_bin — point-cloud files (nuScenes .pcd.bin etc.) as DataFrames
# ═════════════════════════════════════════════════════════════════════════════


def _ssh_cfg():
    """CRAFT's SSH config from the IPython user namespace (optional)."""
    nss = [globals()]
    if get_ipython is not None:
        try:
            nss.append(get_ipython().user_ns)
        except Exception:
            pass
    for ns in nss:
        if isinstance(ns, dict) and ns.get("SSH_HOST"):
            return ns["SSH_HOST"], ns.get("SSH_OPTS", "")
    raise RuntimeError("SSH_HOST not found — load CRAFT and run %gpu first.")


def _ssh_bytes(remote_cmd: str) -> bytes:
    host, opts = _ssh_cfg()
    cmd = ["ssh", *shlex.split(opts or ""), host, remote_cmd]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ssh failed (rc={proc.returncode}): {err[:300]}")
    return proc.stdout or b""


def read_bin(path, *, stride=5, columns=None, remote=False, sub=1,
             max_points=500_000) -> pd.DataFrame:
    """Load an (N, stride) float32 point-cloud file as a DataFrame.

    columns: names for the stride columns (default x,y,z,intensity,c4,...).
    remote=True streams (and thins on) the CRAFT GPU host over SSH.
    """
    stride = max(3, int(stride))
    sub = max(1, int(sub or 1))
    if remote:
        thin = (
            "python3 -c " + shlex.quote(
                "import sys,numpy as np;"
                f"a=np.fromfile({str(path)!r},dtype=np.float32);"
                f"s={stride};n=a.size//s;a=a[:n*s].reshape(n,s);"
                f"m={int(max_points)};sb={sub};"
                "sb=max(sb,(n+m-1)//m) if m>0 else sb;"
                "sys.stdout.buffer.write("
                "np.ascontiguousarray(a[::sb],dtype=np.float32).tobytes())"
            )
        )
        raw = _ssh_bytes(thin)
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        arr = np.fromfile(str(path), dtype=np.float32)
    n = arr.size // stride
    arr = arr[: n * stride].reshape(n, stride)
    if not remote:
        if max_points and n > max_points * sub:
            sub = max(sub, (n + max_points - 1) // max_points)
        if sub > 1:
            arr = np.ascontiguousarray(arr[::sub])
    if columns is None:
        base = ["x", "y", "z", "intensity"]
        columns = (base + [f"c{i}" for i in range(4, stride)])[:stride]
    return pd.DataFrame(arr, columns=list(columns)[:stride])


# ═════════════════════════════════════════════════════════════════════════════
# Hide-from-AI (same mechanism as pcviz/sslive; self-contained copy)
# ═════════════════════════════════════════════════════════════════════════════


def _find_caller_msg_id():
    import inspect

    frame = inspect.currentframe()
    try:
        f = frame.f_back if frame is not None else None
        while f is not None:
            for ns in (f.f_locals, f.f_globals):
                mid = ns.get("__msg_id") if isinstance(ns, dict) else None
                if mid:
                    return str(mid)
            f = f.f_back
    finally:
        del frame
    try:
        ip = get_ipython()
        for ns_name in ("user_ns", "user_global_ns"):
            ns = getattr(ip, ns_name, None) or {}
            mid = ns.get("__msg_id") if isinstance(ns, dict) else None
            if mid:
                return str(mid)
    except Exception:
        pass
    try:
        from safepyrun import find_var  # type: ignore

        mid = find_var("__msg_id")
        if mid:
            return str(mid)
    except Exception:
        pass
    return None


def _hide_caller_from_ai(mid=None):
    """Best-effort ``skipped=1`` on the calling cell; no-op outside SolveIt."""
    try:
        from dialoghelper.core import update_msg
    except Exception:
        return

    async def _run():
        import inspect

        m = mid or _find_caller_msg_id()
        if not m:
            try:
                from dialoghelper.core import read_msg

                msg = read_msg(n=0, relative=True)
                if inspect.iscoroutine(msg):
                    msg = await msg
                m = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
            except Exception:
                m = None
        if not m:
            try:
                from dialoghelper.core import find_msgs

                msgs = find_msgs(msg_type="code", re_pattern=r"%plot3",
                                 include_output=False, include_meta=True,
                                 include_skipped=True, use_regex=True)
                if inspect.iscoroutine(msgs):
                    msgs = await msgs
                for msg in msgs or []:
                    m = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
            except Exception:
                m = None
        if not m:
            print("plot3: hide-from-ai failed — could not resolve msg id "
                  "(pass hide=0 to silence)")
            return
        m = str(m)
        err = None
        for cand in (m, m[1:] if m.startswith("_") else "_" + m):
            try:
                res = update_msg(id=cand, skipped=1)
                if inspect.iscoroutine(res):
                    await res
                return
            except Exception as e:
                err = e
        print(f"plot3: hide-from-ai failed — update_msg({m}): {err}")

    try:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            asyncio.run(_run())
            return
        try:
            import nest_asyncio

            nest_asyncio.apply()
            loop.run_until_complete(_run())
        except Exception:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(lambda: asyncio.run(_run())).result()
    except Exception as e:
        print(f"plot3: hide-from-ai failed — {e}")


# ═════════════════════════════════════════════════════════════════════════════
# %plot3 magic — remote (CRAFT) or local DataFrame snapshot -> figure
# ═════════════════════════════════════════════════════════════════════════════


def _remote_df(expr: str, cols: dict, max_points: int) -> pd.DataFrame:
    """Snapshot mapped columns of a remote DataFrame over CRAFT's SSH pipe."""
    import io
    import uuid

    ip = get_ipython()
    rr = (ip.user_ns or {}).get("remote_run_") if ip is not None else None
    if not callable(rr):
        raise RuntimeError("remote_run_ missing — load CRAFT and run %gpu")
    tmp = f"/tmp/plot3_{uuid.uuid4().hex}.npz"
    code = f"""
import numpy as _np, pandas as _pd, json as _json
_df = eval({expr!r})
_m = {int(max_points)}
if _m > 0 and len(_df) > _m:
    _df = _df.iloc[::(len(_df) + _m - 1) // _m]
_out, _meta = {{}}, {{}}
for _a, _c in {cols!r}.items():
    _s = _df[_c]
    if str(_s.dtype).startswith("datetime"):
        _out[_a] = _s.astype("datetime64[ns]").astype("int64").to_numpy()
        _meta[_a] = ["dt", _c]
    elif _s.dtype.kind in "ifub":
        _out[_a] = _s.to_numpy(_np.float32); _meta[_a] = ["num", _c]
    else:
        _codes, _cats = _pd.factorize(_s.astype(str))
        _out[_a] = _codes.astype(_np.int32)
        _meta[_a] = ["cat", _c, [str(x) for x in _cats]]
_np.savez({tmp!r}, **_out)
print(_json.dumps(_meta))
"""
    out = rr(code, max_chars=8000).strip()
    meta = json.loads(out.splitlines()[-1])
    try:
        raw = _ssh_bytes("cat -- " + shlex.quote(tmp))
    finally:
        try:
            _ssh_bytes("rm -f -- " + shlex.quote(tmp))
        except Exception:
            pass
    z = np.load(io.BytesIO(raw))
    data = {}
    for a, m in meta.items():
        v = z[a]
        col = m[1]
        if m[0] == "dt":
            data[col] = pd.to_datetime(v.astype("int64"), unit="ns")
        elif m[0] == "cat":
            data[col] = pd.Categorical.from_codes(
                np.clip(v, 0, len(m[2]) - 1), categories=m[2])
        else:
            data[col] = v
    return pd.DataFrame(data)


def _run_plot3_from_magic(line: str = ""):
    parts = shlex.split(line or "")
    if not parts:
        raise ValueError(
            "usage: %plot3 <df_expr> x=col y=col [z=col] [color=col] "
            "[group=col] [kind=point|line|path|point+line] [size=F] "
            "[max_points=N] [theme=dark|light] [height=Npx] [hide=0|1]"
        )
    expr = parts[0]
    m: dict = {}
    kind, size, hide, theme = "point", None, True, "dark"
    max_points, height = 200_000, "480px"
    for tok in parts[1:]:
        k, _, v = tok.partition("=")
        if k in ("x", "y", "z", "color", "colour", "group"):
            m["color" if k == "colour" else k] = v
        elif k == "kind":
            kind = v
        elif k == "size":
            size = float(v)
        elif k == "max_points":
            max_points = int(v)
        elif k == "theme":
            theme = v
        elif k == "height":
            height = v if v.endswith("px") else f"{int(v)}px"
        elif k == "hide":
            hide = v.lower() in ("1", "true", "yes")
        else:
            raise ValueError(f"unknown option {tok!r}")
    if "x" not in m or "y" not in m:
        raise ValueError("%plot3 needs x= and y=")

    mid = _find_caller_msg_id() if hide else None

    ip = get_ipython() if get_ipython is not None else None
    ns = (ip.user_ns or {}) if ip is not None else {}
    if callable(ns.get("remote_run_")):
        df = _remote_df(expr, m, max_points)
    else:
        df = eval(expr, ns)  # local fallback (plain Jupyter)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if max_points and len(df) > max_points:
            df = df.iloc[:: (len(df) + max_points - 1) // max_points]

    # hide=False: the magic manages the red eye itself (with its own msg id)
    fig = ggplot(df, aes(**m), height=height, hide=False)
    for part in kind.split("+"):
        part = part.strip()
        if part == "point":
            fig = fig + (geom_point(size=size) if size else geom_point())
        elif part == "line":
            fig = fig + geom_line()
        elif part == "path":
            fig = fig + geom_path()
        else:
            raise ValueError(f"unknown kind {part!r}")
    if theme != "dark":
        fig = fig + theme_light()

    from IPython.display import HTML, display

    display(HTML(fig._repr_html_()))
    if hide:
        _hide_caller_from_ai(mid)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Registration (addon contract: register everything via get_ipython)
# ═════════════════════════════════════════════════════════════════════════════


def _register_plot3(*, quiet=True) -> bool:
    if get_ipython is None:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    ok = False
    try:
        ip.magics_manager.register_function(
            _run_plot3_from_magic, magic_kind="line", magic_name="plot3")
        ok = True
    except Exception as e:
        if not quiet:
            print(f"plot3: magic registration failed: {e}")
    # host-local under %gpu (CRAFT hook, when present)
    try:
        reg = (ip.user_ns or {}).get("register_local_magic")
        if callable(reg):
            reg("%plot3")
    except Exception:
        pass
    # public API into user_ns (never rely on %run leaking module globals)
    try:
        ns = ip.user_ns
        for name in __all__:
            ns[name] = globals()[name]
    except Exception:
        pass
    if ok and not quiet:
        print("plot3 ready")
        print("  ggplot(df, aes(x=,y=[,z=][,colour=])) + geom_point()/geom_line()")
        print("  %plot3 df x=a y=b [z=c] [color=d]   read_bin(path)   ggsave()")
    return ok


def load_ipython_extension(ip=None):
    _register_plot3(quiet=True)


try:
    if get_ipython is not None and get_ipython() is not None:
        _register_plot3(quiet=False)
except Exception:
    pass
