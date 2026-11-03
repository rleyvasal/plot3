"""Position transformations: log10 and pseudo-log.

A transformation maps data values into the space an axis is drawn in.
Where it happens depends on who asks for it, as in ggplot2:

* ``scale_x_log10()`` / ``scale_x_continuous(trans="pseudo_log")`` transform
  the data *before* statistics run, so a boxplot's quartiles, whiskers and
  outliers, a histogram's bins, or a smoother's fit are computed on the
  transformed values.
* ``coord_transform(x="log10")`` transforms *after* statistics run: every
  statistic is computed on the original values, then drawn on a log axis.

Tick labels stay in the original units either way.
"""

from __future__ import annotations

import math

import numpy as np


class Transform:
    """One position transformation. Values outside its domain become NaN."""

    name = "identity"

    def transform(self, values) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)

    def inverse(self, values) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)

    def outside(self, values) -> np.ndarray:
        """Finite values the transformation has no place for."""
        return np.zeros(np.shape(values), dtype=bool)

    def value(self, number: float) -> float | None:
        """One value in transformed space, or None outside the domain."""
        out = float(self.transform(np.asarray([float(number)]))[0])
        return out if math.isfinite(out) else None

    def ticks(self, lo: float, hi: float) -> list[list]:
        """``[position, label]`` ticks; ``lo`` / ``hi`` are transformed."""
        raise NotImplementedError

    def spec(self) -> dict:
        """What the viewer needs to invert positions and re-tick on zoom."""
        return {"trans": self.name}

    def __eq__(self, other):
        return isinstance(other, Transform) and self.spec() == other.spec()

    def __hash__(self):
        return hash(tuple(sorted(self.spec().items())))

    def __repr__(self):
        return f"{type(self).__name__}()"


class Log10(Transform):
    """Base-10 logarithm; non-positive values are omitted."""

    name = "log10"

    def transform(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        out = np.full(v.shape, np.nan, dtype=np.float64)
        ok = np.isfinite(v) & (v > 0)
        out[ok] = np.log10(v[ok])
        return out

    def inverse(self, values) -> np.ndarray:
        return np.power(10.0, np.asarray(values, dtype=np.float64))

    def outside(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        return np.isfinite(v) & (v <= 0)

    def ticks(self, lo: float, hi: float) -> list[list]:
        from plot3.scales import log_ticks

        return log_ticks(lo, hi)


class PseudoLog(Transform):
    """``asinh(x / (2 sigma)) / log(base)`` (ggplot2 ``transform_pseudo_log``).

    Linear near zero, logarithmic far from it, and defined everywhere: zeros
    and negative values keep their place. ``sigma`` sets the width of the
    linear region around zero.
    """

    name = "pseudo_log"

    def __init__(self, sigma: float = 1.0, base: float = math.e):
        sigma, base = float(sigma), float(base)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("pseudo_log sigma must be a positive number")
        if not math.isfinite(base) or base <= 1:
            raise ValueError("pseudo_log base must be a number above 1")
        self.sigma = sigma
        self.base = base

    def transform(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        return np.arcsinh(v / (2.0 * self.sigma)) / math.log(self.base)

    def inverse(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        return 2.0 * self.sigma * np.sinh(v * math.log(self.base))

    def ticks(self, lo: float, hi: float) -> list[list]:
        from plot3.scales import fmt_ticks

        raw_lo, raw_hi = (float(x) for x in self.inverse([lo, hi]))
        values = pseudo_log_breaks(raw_lo, raw_hi, self.sigma)
        return [[float(self.transform([v])[0]), label] for v, label in zip(values, fmt_ticks(values))]

    def spec(self) -> dict:
        return {"trans": self.name, "sigma": self.sigma, "base": self.base}

    def __repr__(self):
        return f"PseudoLog(sigma={self.sigma:g}, base={self.base:g})"


def pseudo_log_breaks(lo: float, hi: float, sigma: float = 1.0) -> list[float]:
    """Breaks for a pseudo-log axis, in original units.

    Inside the linear region (everything within ten ``sigma`` of zero) these
    are ordinary nice breaks. Beyond it they are 0 and powers of ten on each
    side of zero (with 2 and 5 between them over two decades or fewer).
    """
    from plot3.scales import nice_ticks

    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    reach = max(abs(lo), abs(hi))
    if reach <= 10.0 * sigma:
        return [v for v in nice_ticks(lo, hi, 5) if lo - 1e-9 <= v <= hi + 1e-9] or [lo, hi]
    top = math.ceil(math.log10(reach))
    bottom = max(math.floor(math.log10(sigma)), top - 12)
    mults = (1, 2, 5) if top - bottom <= 2 else (1,)
    magnitudes = [m * 10.0 ** e for e in range(bottom, top + 1) for m in mults]
    out = [0.0] + magnitudes + [-m for m in magnitudes]
    out = sorted(v for v in out if lo - 1e-9 * reach <= v <= hi + 1e-9 * reach)
    while len(out) > 11:
        # Too many decades: keep zero and every other power on each side.
        pos = [v for v in out if v > 0][1::2]
        neg = [v for v in out if v < 0][-2::-2]
        out = sorted(neg + ([0.0] if 0.0 in out else []) + pos)
    return out


LOG10 = Log10()


def pseudo_log_trans(sigma: float = 1.0, base: float = math.e) -> PseudoLog:
    """A pseudo-log transformation for ``scale_x_continuous(trans=)`` or
    ``coord_transform()``: ``trans=pseudo_log_trans(sigma=100)``."""
    return PseudoLog(sigma=sigma, base=base)


transform_pseudo_log = pseudo_log_trans


_NAMED = {"log10": lambda: LOG10, "pseudo_log": PseudoLog}


def as_transform(value) -> Transform | None:
    """``"log10"``, ``"pseudo_log"``, or a Transform; None for no change.

    ``"identity"``, ``"reverse"`` and None map to None: reversing is an axis
    direction, not a change of values.
    """
    if value is None or value in ("identity", "reverse"):
        return None
    if isinstance(value, Transform):
        return value
    if isinstance(value, str) and value in _NAMED:
        return _NAMED[value]()
    raise ValueError(
        "trans is 'log10', 'pseudo_log', pseudo_log_trans(sigma=, base=), "
        "'reverse', or None"
    )
