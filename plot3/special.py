"""Special functions and probability densities a formula may call.

NumPy has no gamma or erf, so these wrap the ``math`` module element-wise.
Names follow R (``dnorm``, ``pnorm``, ``qnorm``), which ggplot users know::

    geom_function("y = dbeta(x, 2, 5)") + area(0.2, 0.5)
    geom_function("y = dnorm(x, mu, sigma)", mu=100, sigma=15)

A density is 0 outside its support, as in R. ``support()`` gives the range
worth drawing, so ``dbeta`` opens on [0, 1] instead of the default (-10, 10).
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable

import numpy as np


def _elementwise(fn: Callable[[float], float]) -> Callable[..., np.ndarray]:
    def safe(*args: float) -> float:
        try:
            return float(fn(*args))
        except (ValueError, OverflowError, ZeroDivisionError):
            return math.nan

    vec = np.vectorize(safe, otypes=[np.float64])

    def apply(*args: Any) -> Any:
        out = vec(*args)
        return out if np.ndim(out) else float(out)

    apply.__name__ = getattr(fn, "__name__", "special")
    return apply


def _gamma(x: float) -> float:
    if x <= 0 and float(x).is_integer():
        return math.inf  # poles at 0, -1, -2, ...
    return math.gamma(x)


gamma = _elementwise(_gamma)
lgamma = _elementwise(math.lgamma)
erf = _elementwise(math.erf)
erfc = _elementwise(math.erfc)


def _lbeta(a, b):
    return lgamma(a) + lgamma(b) - lgamma(np.add(a, b))


def beta(a, b):
    """Euler's Beta function B(a, b) for a, b > 0."""
    return np.exp(_lbeta(a, b))


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def dnorm(x, mean=0.0, sd=1.0):
    x, mean, sd = _f(x), _f(mean), _f(sd)
    with np.errstate(all="ignore"):
        z = (x - mean) / sd
        return np.where(sd > 0, np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi)), np.nan)


def pnorm(q, mean=0.0, sd=1.0):
    q, mean, sd = _f(q), _f(mean), _f(sd)
    with np.errstate(all="ignore"):
        return 0.5 * erfc(-(q - mean) / (sd * math.sqrt(2.0)))


# Acklam's rational approximation to the normal quantile, |error| < 1.2e-9,
# then one Halley step against erfc for full double precision.
_QA = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
       1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_QB = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
       6.680131188771972e01, -1.328068155288572e01)
_QC = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
       -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_QD = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
       3.754408661907416e00)


def _qnorm1(p: float) -> float:
    if not 0.0 <= p <= 1.0 or math.isnan(p):
        return math.nan
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf
    low = 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_QC[0] * q + _QC[1]) * q + _QC[2]) * q + _QC[3]) * q + _QC[4]) * q + _QC[5]) / (
            (((_QD[0] * q + _QD[1]) * q + _QD[2]) * q + _QD[3]) * q + 1.0)
    elif p <= 1.0 - low:
        q = p - 0.5
        r = q * q
        x = (((((_QA[0] * r + _QA[1]) * r + _QA[2]) * r + _QA[3]) * r + _QA[4]) * r + _QA[5]) * q / (
            ((((_QB[0] * r + _QB[1]) * r + _QB[2]) * r + _QB[3]) * r + _QB[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_QC[0] * q + _QC[1]) * q + _QC[2]) * q + _QC[3]) * q + _QC[4]) * q + _QC[5]) / (
            (((_QD[0] * q + _QD[1]) * q + _QD[2]) * q + _QD[3]) * q + 1.0)
    err = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = err * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


_qnorm_vec = _elementwise(_qnorm1)


def qnorm(p, mean=0.0, sd=1.0):
    return _f(mean) + _f(sd) * _f(_qnorm_vec(p))


def dbeta(x, a, b):
    x, a, b = _f(x), _f(a), _f(b)
    with np.errstate(all="ignore"):
        body = np.exp((a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x) - _lbeta(a, b))
        # x^0 at the edge is 1 even where log gives -inf * 0 = nan.
        body = np.where((x == 0.0) & (a == 1.0), np.exp(-_lbeta(a, b)), body)
        body = np.where((x == 1.0) & (b == 1.0), np.exp(-_lbeta(a, b)), body)
        inside = (x >= 0.0) & (x <= 1.0)
        return np.where(inside, body, 0.0)


def dt(x, df):
    x, df = _f(x), _f(df)
    with np.errstate(all="ignore"):
        log_c = lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0) - 0.5 * np.log(df * math.pi)
        return np.exp(log_c - (df + 1.0) / 2.0 * np.log1p(x * x / df))


def dgamma(x, shape, rate=1.0):
    x, shape, rate = _f(x), _f(shape), _f(rate)
    with np.errstate(all="ignore"):
        body = np.exp(
            shape * np.log(rate) + (shape - 1.0) * np.log(x) - rate * x - lgamma(shape)
        )
        body = np.where((x == 0.0) & (shape == 1.0), rate, body)
        return np.where(x >= 0.0, body, 0.0)


def dchisq(x, df):
    return dgamma(x, _f(df) / 2.0, 0.5)


def dexp(x, rate=1.0):
    x, rate = _f(x), _f(rate)
    with np.errstate(all="ignore"):
        return np.where(x >= 0.0, rate * np.exp(-rate * x), 0.0)


def dunif(x, min=0.0, max=1.0):  # noqa: A002 - R's argument names
    x, lo, hi = _f(x), _f(min), _f(max)
    with np.errstate(all="ignore"):
        return np.where((x >= lo) & (x <= hi), 1.0 / (hi - lo), 0.0)


def dlnorm(x, meanlog=0.0, sdlog=1.0):
    x, mu, s = _f(x), _f(meanlog), _f(sdlog)
    with np.errstate(all="ignore"):
        z = (np.log(x) - mu) / s
        body = np.exp(-0.5 * z * z) / (x * s * math.sqrt(2.0 * math.pi))
        return np.where(x > 0.0, body, 0.0)


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "gamma": gamma,
    "lgamma": lgamma,
    "beta": beta,
    "erf": erf,
    "erfc": erfc,
    "dnorm": dnorm,
    "pnorm": pnorm,
    "qnorm": qnorm,
    "dbeta": dbeta,
    "dt": dt,
    "dgamma": dgamma,
    "dchisq": dchisq,
    "dexp": dexp,
    "dunif": dunif,
    "dlnorm": dlnorm,
}

# Densities integrate to 1, so a shaded area under one is a probability.
DENSITIES = frozenset({"dnorm", "dbeta", "dt", "dgamma", "dchisq", "dexp", "dunif", "dlnorm"})


def _support(name: str, args: list[float]) -> tuple[float, float] | None:
    """The x range worth drawing for one density call (args after x)."""
    def arg(i: int, default: float) -> float:
        return float(args[i]) if len(args) > i else default

    if name == "dnorm":
        mu, sd = arg(0, 0.0), arg(1, 1.0)
        return (mu - 4.0 * sd, mu + 4.0 * sd) if sd > 0 else None
    if name == "dbeta":
        return (0.0, 1.0)
    if name == "dt":
        df = arg(0, 1.0)
        half = 5.0 if df >= 3 else 8.0
        return (-half, half)
    if name in {"dgamma", "dchisq"}:
        if name == "dgamma":
            shape, rate = arg(0, 1.0), arg(1, 1.0)
        else:
            shape, rate = arg(0, 1.0) / 2.0, 0.5
        if shape <= 0 or rate <= 0:
            return None
        mean, sd = shape / rate, math.sqrt(shape) / rate
        return (0.0, mean + 5.0 * sd)
    if name == "dexp":
        rate = arg(0, 1.0)
        return (0.0, 6.0 / rate) if rate > 0 else None
    if name == "dunif":
        lo, hi = arg(0, 0.0), arg(1, 1.0)
        pad = 0.1 * (hi - lo)
        return (lo - pad, hi + pad) if hi > lo else None
    if name == "dlnorm":
        mu, s = arg(0, 0.0), arg(1, 1.0)
        return (0.0, math.exp(mu + 3.0 * s)) if s > 0 else None
    return None


def density_calls(tree: ast.AST | None, variable: str) -> list[ast.Call]:
    """Density calls whose first argument is the plot variable itself."""
    if tree is None:
        return []
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in DENSITIES
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == variable
    ]


def support_hint(
    tree: ast.AST | None, variable: str, namespace: dict[str, Any]
) -> tuple[float, float] | None:
    """Union of the supports of every density of ``variable`` in ``tree``.

    None when there is no density, or a parameter is not a plain number
    (a slider or transition changes it frame by frame).
    """
    calls = density_calls(tree, variable)
    if not calls:
        return None
    lo, hi = math.inf, -math.inf
    for call in calls:
        values = []
        for node in call.args[1:]:
            try:
                code = compile(ast.Expression(body=node), "<support>", "eval")
                value = float(eval(code, {"__builtins__": {}}, dict(namespace)))  # noqa: S307
            except Exception:
                return None
            if not math.isfinite(value):
                return None
            values.append(value)
        span = _support(call.func.id, values)
        if span is None:
            return None
        lo, hi = min(lo, span[0]), max(hi, span[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return None
    return lo, hi
