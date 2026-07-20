"""Positional scales, ticks, and column typing."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

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
            d["ladder"] = dt_ladder(self.lo, self.hi)
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

