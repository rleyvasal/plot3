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


def _nearest_multiple(value: float, step: float) -> float:
    """The multiple of ``step`` closest to ``value``. A tie takes the higher one."""
    k = value / step
    down = math.floor(k + 1e-10)
    up = math.ceil(k - 1e-10)
    if abs(k - down) < abs(up - k) - 1e-10:
        return down * step
    return up * step


def _ticks_on_step(lo: float, hi: float, step: float) -> list[float] | None:
    start = _nearest_multiple(lo, step)
    end = _nearest_multiple(hi, step)
    if end < start:
        start, end = end, start
    count = int(round((end - start) / step))
    if count < 0 or count > 40:
        return None
    origin = int(round(start / step))
    out = []
    for i in range(count + 1):
        t = (origin + i) * step
        if abs(t) < abs(step) * 1e-8:
            t = 0.0
        else:
            t = float(f"{t:.12g}")
        out.append(t)
    return out


def _classic_ticks(lo: float, hi: float, n: int) -> list[float]:
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(max(raw, 1e-12)))
    step = 10 * mag
    for mult in (1, 2, 5, 10):
        if raw <= mult * mag:
            step = mult * mag
            break
    found = _ticks_on_step(lo, hi, step)
    return found or [lo, hi]


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """About ``n`` ticks on a 1-2-2.5-5 grid inside the range, as ggplot2's
    extended breaks: 0, 2, 4, 6, 8 for 0 to 9.2, not every unit.

    Ticks may sit up to 5% of the span past either end (the view's margin),
    so a sample maximum of 0.9997 still gets a tick at 1. Among steps with
    about ``n`` ticks, the one whose ticks reach nearest both ends wins;
    2.5 steps count as a little less round than 1, 2, and 5.
    """
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    span = hi - lo
    target = max(3, int(n))
    pad = 0.05 * span
    raw = span / max(1, target - 1)
    exp = math.floor(math.log10(max(raw, 1e-300)))
    best: list[float] | None = None
    best_score = math.inf
    best_step = -1.0
    for shift in (-1, 0, 1):
        base = 10.0 ** (exp + shift)
        for mult in (1, 2, 2.5, 5):
            step = mult * base
            first = math.ceil((lo - pad) / step - 1e-9)
            last = math.floor((hi + pad) / step + 1e-9)
            count = last - first + 1
            if count < 2 or count > 40:
                continue
            score = float(abs(count - target))
            if count < 4:
                score += 3.0
            if count > target + 2:
                score += 2.0 * (count - target - 2)
            gap = max(0.0, first * step - lo) + max(0.0, hi - last * step)
            score += 2.0 * gap / span
            if mult == 2.5:
                score += 0.25
            if score < best_score - 1e-9 or (abs(score - best_score) <= 1e-9 and step > best_step):
                # Round to the step, not to significant figures, so ticks
                # near 1e12 stay apart.
                digits = max(0, 3 - math.floor(math.log10(step)))
                best = []
                for k in range(first, last + 1):
                    tick = k * step
                    best.append(0.0 if abs(tick) < abs(step) * 1e-8 else round(tick, digits))
                best_score = score
                best_step = step
    if best:
        return best
    return _classic_ticks(lo, hi, n)


def fmt_num(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e6 or a < 1e-4:
        return f"{v:.3g}"
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s


def fmt_ticks(values) -> list[str]:
    """Tick labels that stay apart: 1000000000001, not five "1e+12"."""
    labels = [fmt_num(v) for v in values]
    if len(set(labels)) == len(labels):
        return labels
    if all(float(v).is_integer() and abs(v) < 1e15 for v in values):
        return [str(int(v)) for v in values]
    for digits in range(4, 16):
        labels = [f"{v:.{digits}g}" for v in values]
        if len(set(labels)) == len(labels):
            return labels
    return labels


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
        lo, hi = min(self.lo, self.hi), max(self.lo, self.hi)  # a reversed axis
        if self.trans:
            d["trans"] = self.trans
        if self.kind == "cat":
            d["cats"] = self.cats
        elif self.kind == "dt":
            d["ladder"] = dt_ladder(lo, hi)
        elif self.trans == "log10":
            d["ticks"] = log_ticks(lo, hi)
        else:
            ticks = nice_ticks(lo, hi)
            d["ticks"] = [list(pair) for pair in zip(ticks, fmt_ticks(ticks))]
        custom = getattr(self, "custom", None)
        if custom is not None:
            _apply_custom(d, self, custom, lo, hi)
        return d


def _is_na(value) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def ordered_levels(values, *, keep_order: bool = False) -> list:
    """Distinct values in ggplot2's level order, missing values last.

    ``keep_order`` (a categorical column) keeps the given order. Otherwise
    booleans run False, True; numbers sort numerically (4, 6, 8, 10, not
    "10" before "4"); anything else sorts as text.
    """
    values = list(values)
    present = [v for v in values if not _is_na(v)]
    missing = len(present) != len(values)
    distinct = list(dict.fromkeys(present))
    if not keep_order:
        if all(isinstance(v, (bool, np.bool_)) for v in distinct):
            distinct.sort(key=bool)
        elif all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_))
            for v in distinct
        ):
            distinct.sort(key=float)
        else:
            distinct.sort(key=str)
    return distinct + ([None] if missing else [])


def col_values(s: pd.Series) -> tuple[str, np.ndarray, list[str]]:
    """Series -> (scale kind, float64 positions, categories)."""
    if pd.api.types.is_bool_dtype(s):
        # True/False are two groups (ggplot2), not a 0-1 colour gradient.
        cats = [str(v) for v in ordered_levels(s.dropna().unique().tolist())]
        idx = {c: i for i, c in enumerate(cats)}
        return "cat", s.map(lambda v: idx.get(str(v), np.nan)).to_numpy(np.float64), cats
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



def _apply_custom(d: dict, scale: "Scale", custom, lo: float, hi: float) -> None:
    """scale_x_continuous(breaks=, labels=), scale_x_discrete(labels=), and
    scale_x_date(date_breaks=, date_labels=) on a resolved scale spec.

    ``fixed`` tells the viewer to keep these ticks when it zooms.
    """
    from plot3.scaling import date_freq, format_label

    labels = custom.labels
    if scale.kind == "cat":
        if isinstance(labels, dict):
            d["cats"] = [str(labels.get(c, c)) for c in scale.cats]
        elif isinstance(labels, (list, tuple)):
            d["cats"] = [str(labels[i]) if i < len(labels) else c for i, c in enumerate(scale.cats)]
        elif callable(labels):
            d["cats"] = [str(labels(c)) for c in scale.cats]
        return
    if scale.kind == "dt":
        fmt = custom.date_labels
        if custom.date_breaks:
            start = pd.Timestamp(lo, unit="s")
            stop = pd.Timestamp(hi, unit="s")
            freq = date_freq(custom.date_breaks)
            stamps = pd.date_range(start.floor("D"), stop, freq=freq)
            stamps = [t for t in stamps if lo <= t.timestamp() <= hi]
            d["ticks"] = [[t.timestamp(), t.strftime(fmt or "%Y-%m-%d")] for t in stamps]
            d["fixed"] = True
        elif fmt:
            d["ladder"] = [
                [[t, pd.Timestamp(t, unit="s").strftime(fmt)] for t, _label in level]
                for level in d.get("ladder") or []
            ]
        return
    log = scale.trans == "log10"
    if custom.breaks is not None:
        ticks = []
        for i, value in enumerate(custom.breaks):
            value = float(value)
            if log and value <= 0:
                continue
            pos = math.log10(value) if log else value
            if not lo - 1e-9 <= pos <= hi + 1e-9:
                continue
            if isinstance(labels, (list, tuple)):
                text = str(labels[i])
            elif labels is not None:
                text = format_label(value, labels)
            else:
                text = fmt_num(value)
            ticks.append([pos, text])
        d["ticks"] = ticks
        d["fixed"] = True
    elif labels is not None and not isinstance(labels, (list, tuple)):
        d["ticks"] = [
            [pos, format_label(10 ** pos if log else pos, labels)] for pos, _text in d.get("ticks") or []
        ]
        d["fixed"] = True
