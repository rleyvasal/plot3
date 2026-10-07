"""Positional scales, ticks, and column typing."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def resolution(x, *, zero: bool = True) -> float:
    """Smallest non-zero distance between adjacent values (ggplot2 ``resolution``).

    Used to turn a relative bar ``width`` (default 0.9) into data units:
    ``data_width = resolution(x) * width``.

    * Integer-like vectors → ``1`` (ggplot2 treats integers as unit-spaced).
    * Single unique value / zero range → ``1``.
    * Otherwise → minimum positive difference between sorted unique values.
    * If ``zero`` is True (ggplot2 default for some paths), ``0`` is included
      in the unique set before measuring gaps.
    """
    raw = np.asarray(x)
    # Integer storage (int32/Int64/…) matches ggplot2 is.integer → 1.
    if np.issubdtype(raw.dtype, np.integer):
        return 1.0
    arr = np.asarray(raw, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo or not math.isfinite(lo) or not math.isfinite(hi):
        return 1.0
    uniq = np.unique(arr)
    if zero:
        uniq = np.unique(np.concatenate([uniq, np.asarray([0.0])]))
    if uniq.size < 2:
        return 1.0
    d = np.diff(np.sort(uniq))
    tol = math.sqrt(np.finfo(float).eps)
    positive = d[d > tol]
    if positive.size == 0:
        return 1.0
    return float(np.min(positive))


def nice_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
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


def fmt_num(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e6 or a < 1e-4:
        return f"{v:.3g}"
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s


def log_ticks(lo: float, hi: float) -> list[list]:
    """Ticks for a log10 scale.

    ``lo`` and ``hi`` are already log10 values (the scale's stored domain).
    Each tick is ``[log10(value), label]`` so it sits in that same space.
    """
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        val = 10 ** lo if math.isfinite(lo) else 1.0
        return [[lo, fmt_num(val)]]
    span = hi - lo
    mults = (1, 2, 5) if span <= 3 else (1,)
    exp0 = math.floor(lo)
    exp1 = math.ceil(hi)
    out: list[list] = []
    for exp in range(int(exp0), int(exp1) + 1):
        for mult in mults:
            val = mult * 10.0 ** exp
            lv = math.log10(val)
            if lv < lo - 1e-9 or lv > hi + 1e-9:
                continue
            out.append([lv, fmt_num(val)])
    if not out:
        out = [[lo, fmt_num(10 ** lo)], [hi, fmt_num(10 ** hi)]]
    return out


DT_LADDERS = [
    # (span_seconds >, [(pandas freq, strftime fmt), coarse -> fine])
    (2 * 365 * 86400, [("YS", "%Y"), ("QS", "%b %Y"), ("MS", "%b %Y")]),
    (90 * 86400, [("MS", "%b %Y"), ("W", "%b %d"), ("D", "%b %d")]),
    (3 * 86400, [("D", "%b %d"), ("6h", "%d %Hh"), ("h", "%H:%M")]),
    (3 * 3600, [("h", "%H:%M"), ("15min", "%H:%M"), ("min", "%H:%M")]),
    (0, [("min", "%H:%M"), ("15s", "%H:%M:%S"), ("s", "%H:%M:%S")]),
]


def dt_ladder(lo_s: float, hi_s: float) -> list[list[list]]:
    """3-level [position_seconds, label] ladders; JS picks by visible count."""
    span = hi_s - lo_s
    for min_span, freqs in DT_LADDERS:
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


class Scale:
    """Resolved positional scale: numeric, datetime or categorical.

    ``trans`` is ``None`` or ``"log10"``. A log scale stores ``lo`` / ``hi``
    and tick positions in log10 space. Labels stay in the original units.
    """

    def __init__(self, kind: str, trans: str | None = None):
        self.kind = kind  # "num" | "dt" | "cat"
        self.trans = trans
        self.lo = math.inf
        self.hi = -math.inf
        self.cats: list[str] = []

    def widen(self, values: np.ndarray):
        if len(values) == 0:
            return
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        self.lo = min(self.lo, float(finite.min()))
        self.hi = max(self.hi, float(finite.max()))

    def finish(self):
        if self.kind == "cat":
            self.lo, self.hi = -0.5, max(0.5, len(self.cats) - 0.5)
        elif not math.isfinite(self.lo):
            self.lo, self.hi = 0.0, 1.0
        elif self.hi <= self.lo:
            self.lo, self.hi = self.lo - 0.5, self.hi + 0.5

    def spec(self) -> dict:
        d = {"kind": self.kind, "lo": self.lo, "hi": self.hi}
        if self.trans:
            d["trans"] = self.trans
        if self.kind == "cat":
            d["cats"] = self.cats
        elif self.kind == "dt":
            d["ladder"] = dt_ladder(self.lo, self.hi)
        elif self.trans == "log10":
            d["ticks"] = log_ticks(self.lo, self.hi)
        else:
            d["ticks"] = [[t, fmt_num(t)] for t in nice_ticks(self.lo, self.hi)]
        return d


def col_values(s: pd.Series) -> tuple[str, np.ndarray, list[str]]:
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

