"""Binned Gaussian kernel density sums (linear binning + FFT convolution).

R's ``density()`` bins too. Values are spread onto a fine grid (at least
four points per output interval, at most 1/20 of a bandwidth apart in
1D and 1/4 in 2D),
convolved with the kernel by FFT, and read back at the output points. The
relative error is far below what a plot shows. Values more than
``_REACH`` bandwidths past the output range add nothing in float64 and are
left out. The callers keep the exact sum for small inputs, and these
helpers return None when the fine grid would be too large.
"""

from __future__ import annotations

import numpy as np

# Below this many (observations x output points) the exact sum is cheap.
EXACT_WORK = 2_000_000
# The fine spacing is at most this fraction of the bandwidth (1D, 2D), and
# the fine grid has at least four points per output interval. Linear
# binning's error grows with the square of that fraction.
_MAX_STEP_1D = 0.05
_MAX_STEP_2D = 0.25
_MIN_FINE = 4
# Past this many bandwidths a kernel underflows to zero in float64.
_REACH = 40.0
# Largest fine grid (points, or cells in 2D) worth binning onto.
_MAX_CELLS = 1 << 22


def _fine_axis(values: np.ndarray, lo: float, hi: float, n: int, h: float, max_step: float):
    """``(origin, step, size, first, fine)`` of a fine grid through the output points.

    The output point ``k`` sits at fine index ``first + k * fine``. The grid
    covers every value within ``_REACH`` bandwidths of the output range;
    values farther out add nothing to the sums.
    """
    if n < 2 or not hi > lo or not h > 0 or not values.size:
        return None
    fine = max(_MIN_FINE, int(np.ceil((hi - lo) / ((n - 1) * max_step * h))))
    step = (hi - lo) / ((n - 1) * fine)
    low = max(float(values.min()), lo - _REACH * h)
    high = min(float(values.max()), hi + _REACH * h)
    below = max(0, int(np.ceil((lo - low) / step)))
    above = max(0, int(np.ceil((high - hi) / step)))
    size = (n - 1) * fine + 1 + below + above
    if size > _MAX_CELLS:
        return None
    return lo - below * step, step, size, below, fine


def _within(values: np.ndarray, origin: float, step: float, size: int) -> np.ndarray:
    return (values >= origin) & (values <= origin + (size - 1) * step)


def _linear_bins(values: np.ndarray, origin: float, step: float, size: int):
    """Indices and weights of the two fine points around each value."""
    pos = (values - origin) / step
    left = np.clip(np.floor(pos).astype(np.int64), 0, size - 2)
    frac = np.clip(pos - left, 0.0, 1.0)
    return left, frac


def _kernel_fft(step: float, size: int, h: float, length: int) -> np.ndarray:
    offsets = np.arange(-(size - 1), size) * step
    kernel = np.exp(-0.5 * (offsets / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
    return np.fft.rfft(kernel, length)


def _fft_length(size: int) -> int:
    return 1 << int(np.ceil(np.log2(3 * size - 2)))


def _convolve(weights: np.ndarray, step: float, size: int, h: float, axis: int) -> np.ndarray:
    """``sum_i w[i] K((j - i) step)`` for every fine index ``j`` along *axis*."""
    length = _fft_length(size)
    spectrum = np.fft.rfft(weights, length, axis=axis)
    shape = [1] * weights.ndim
    shape[axis] = -1
    full = np.fft.irfft(spectrum * _kernel_fft(step, size, h, length).reshape(shape), length, axis=axis)
    return np.take(full, np.arange(size - 1, 2 * size - 1), axis=axis)


def gaussian_1d(x: np.ndarray, lo: float, hi: float, n: int, h: float) -> np.ndarray | None:
    """``sum_i phi((g - x_i) / h) / h`` at ``linspace(lo, hi, n)``, or None."""
    axis = _fine_axis(x, lo, hi, n, h, _MAX_STEP_1D)
    if axis is None:
        return None
    origin, step, size, first, fine = axis
    x = x[_within(x, origin, step, size)]
    left, frac = _linear_bins(x, origin, step, size)
    weights = np.bincount(left, 1.0 - frac, size) + np.bincount(left + 1, frac, size)
    sums = _convolve(weights, step, size, h, axis=0)
    return np.maximum(sums[first:first + (n - 1) * fine + 1:fine], 0.0)


def gaussian_2d(
    xs: np.ndarray, ys: np.ndarray, x_lims: tuple[float, float], y_lims: tuple[float, float],
    n: int, hx: float, hy: float,
) -> np.ndarray | None:
    """``sum_i phi_x(gx - x_i) phi_y(gy - y_i)`` as an ``(n, n)`` (y, x) grid, or None."""
    ax = _fine_axis(xs, x_lims[0], x_lims[1], n, hx, _MAX_STEP_2D)
    ay = _fine_axis(ys, y_lims[0], y_lims[1], n, hy, _MAX_STEP_2D)
    if ax is None or ay is None:
        return None
    (x_origin, x_step, x_size, x_first, x_fine), (y_origin, y_step, y_size, y_first, y_fine) = ax, ay
    # Two FFT passes over the fine grid must beat n exponentials per value
    # on each axis; a grid stretched over wide limits does not.
    if x_size * y_size > _MAX_CELLS or x_size * y_size * 16 > xs.size * n:
        return None
    keep = _within(xs, x_origin, x_step, x_size) & _within(ys, y_origin, y_step, y_size)
    xs, ys = xs[keep], ys[keep]
    xl, xf = _linear_bins(xs, x_origin, x_step, x_size)
    yl, yf = _linear_bins(ys, y_origin, y_step, y_size)
    cells = y_size * x_size
    weights = np.zeros(cells)
    for dy, wy in ((0, 1.0 - yf), (1, yf)):
        for dx, wx in ((0, 1.0 - xf), (1, xf)):
            weights += np.bincount((yl + dy) * x_size + xl + dx, wy * wx, cells)
    weights = weights.reshape(y_size, x_size)
    sums = _convolve(_convolve(weights, x_step, x_size, hx, axis=1), y_step, y_size, hy, axis=0)
    rows = slice(y_first, y_first + (n - 1) * y_fine + 1, y_fine)
    cols = slice(x_first, x_first + (n - 1) * x_fine + 1, x_fine)
    return np.maximum(sums[rows, cols], 0.0)
