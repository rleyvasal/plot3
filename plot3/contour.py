"""Implicit-curve contours.

The NumPy backend is the default. ``contourpy`` is used when it is installed
and ``PLOT3_CONTOUR_BACKEND`` asks for it. ``import plot3`` does not import
``contourpy``; the import happens the first time a contour is drawn.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

# One segment per non-saddle case: local edge indices
# 0 bottom, 1 right, 2 top, 3 left. Saddles (5 and 10) are filled below
# from the cell-centre value, same pairing as the original cell loop.
_EDGE_LUT = np.array(
    [
        [-1, -1],  # 0
        [0, 3],  # 1
        [0, 1],  # 2
        [1, 3],  # 3
        [1, 2],  # 4
        [-1, -1],  # 5 saddle
        [0, 2],  # 6
        [2, 3],  # 7
        [2, 3],  # 8
        [0, 2],  # 9
        [-1, -1],  # 10 saddle
        [1, 2],  # 11
        [1, 3],  # 12
        [0, 1],  # 13
        [0, 3],  # 14
        [-1, -1],  # 15
    ],
    dtype=np.int8,
)

# Subdivide each coarse cell the contour crosses. 8× on the default 80-grid
# keeps a unit circle smooth on (-10, 10). A full-window refit is not used:
# it cannot refine a small loop that shares the window with a long curve.
_REFINE_SUB = 8
_REFINE_NODE_BUDGET = 400_000
_REFINE_GROW = 4  # rounds of refining around a curve's stray ends

_contourpy_mod = None
_contourpy_loaded = False
_contourpy_warned = False


def _reset_contour_backend() -> None:
    """Forget the cached contourpy import. Tests use this."""
    global _contourpy_mod, _contourpy_loaded, _contourpy_warned
    _contourpy_mod = None
    _contourpy_loaded = False
    _contourpy_warned = False


def _version_at_least(version: str, major: int, minor: int) -> bool:
    parts: list[int] = []
    for piece in str(version).split("."):
        num = ""
        for ch in piece:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            break
        parts.append(int(num))
        if len(parts) == 2:
            break
    while len(parts) < 2:
        parts.append(0)
    return (parts[0], parts[1]) >= (major, minor)


def _load_contourpy():
    """Return the contourpy module, or None if it is missing or too old."""
    global _contourpy_mod, _contourpy_loaded
    if _contourpy_loaded:
        return _contourpy_mod
    _contourpy_loaded = True
    try:
        import contourpy
    except ImportError:
        _contourpy_mod = None
        return None
    version = getattr(contourpy, "__version__", "0")
    if not _version_at_least(version, 1, 0):
        _contourpy_mod = None
        return None
    _contourpy_mod = contourpy
    return contourpy


def _backend_setting() -> str:
    """Raw ``PLOT3_CONTOUR_BACKEND`` value: auto, numpy, or contourpy."""
    raw = os.environ.get("PLOT3_CONTOUR_BACKEND", "auto").strip().lower()
    if raw in ("", "auto", "numpy", "contourpy"):
        return raw or "auto"
    raise ValueError(
        "PLOT3_CONTOUR_BACKEND must be auto, numpy, or contourpy, "
        f"got {raw!r}"
    )


def _backend_name() -> str:
    """Resolved backend name. ``contourpy`` is an error when it is missing."""
    choice = _backend_setting()
    if choice == "numpy":
        return "numpy"
    if choice == "contourpy":
        if _load_contourpy() is None:
            raise ValueError(
                "PLOT3_CONTOUR_BACKEND=contourpy but contourpy>=1.0 is not "
                "installed. Install it with: pip install contourpy"
            )
        return "contourpy"
    if _load_contourpy() is None:
        return "numpy"
    return "contourpy"


def _interp_t(a: np.ndarray, b: np.ndarray, level: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        span = b - a
        safe = np.where(span == 0.0, 1.0, span)
        t = np.where(span == 0.0, 0.5, (level - a) / safe)
        return np.clip(t, 0.0, 1.0)


_ROOT_STEPS = 14  # bisection steps per crossing: 1/16384 of a fine cell


def _edge_root(sample, fa, fb, xa, ya, xb, yb, level: float) -> np.ndarray:
    """Where the curve crosses each edge, as a fraction from a to b.

    Linear interpolation of the corner values is exact only where the
    function crosses zero like a line. Near a double or triple root (the
    heart ``(x^2 + y^2 - 1)^3 = x^2 y^3``) it can miss by half a cell, which
    shows as kinks. With ``sample``, the crossing is bisected on the
    function itself, then interpolated inside the final bracket.
    """
    t = _interp_t(fa, fb, level)
    if sample is None:
        return t
    fa = np.asarray(fa, dtype=np.float64)
    fb = np.asarray(fb, dtype=np.float64)
    cross = np.isfinite(fa) & np.isfinite(fb) & ((fa >= level) != (fb >= level))
    if not np.any(cross):
        return t
    xa, ya, xb, yb = (np.broadcast_to(v, fa.shape)[cross] for v in (xa, ya, xb, yb))
    lo = np.zeros(xa.shape)
    hi = np.ones(xa.shape)
    flo = fa[cross] - level
    fhi = fb[cross] - level
    for _ in range(_ROOT_STEPS):
        mid = 0.5 * (lo + hi)
        fm = np.asarray(sample(xa + mid * (xb - xa), ya + mid * (yb - ya)), dtype=np.float64)
        fm = np.broadcast_to(fm, mid.shape) - level
        ok = np.isfinite(fm)
        low_side = ok & ((fm >= 0) == (flo >= 0))
        high_side = ok & ~low_side
        lo = np.where(low_side, mid, lo)
        flo = np.where(low_side, fm, flo)
        hi = np.where(high_side, mid, hi)
        fhi = np.where(high_side, fm, fhi)
        if not ok.all():
            break
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = np.where(fhi != flo, -flo / (fhi - flo), 0.5)
    out = np.array(t, dtype=np.float64, copy=True)
    out[cross] = lo + np.clip(inner, 0.0, 1.0) * (hi - lo)
    return out


def _pair_segments(
    local_ids: np.ndarray,
    cases: np.ndarray,
    center_inside: np.ndarray,
) -> np.ndarray:
    """Turn per-cell edge ids and case numbers into segment endpoint pairs."""
    if cases.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    saddle = (cases == 5) | (cases == 10)
    chunks: list[np.ndarray] = []
    regular = ~saddle
    if np.any(regular):
        lut = _EDGE_LUT[cases[regular]]
        loc = local_ids[regular]
        index = np.arange(loc.shape[0])
        chunks.append(np.stack([loc[index, lut[:, 0]], loc[index, lut[:, 1]]], axis=1))
    if np.any(saddle):
        loc = local_ids[saddle]
        inside = center_inside[saddle]
        if np.any(inside):
            chunks.append(np.stack([loc[inside, 0], loc[inside, 1]], axis=1))
            chunks.append(np.stack([loc[inside, 2], loc[inside, 3]], axis=1))
        outside = ~inside
        if np.any(outside):
            chunks.append(np.stack([loc[outside, 0], loc[outside, 3]], axis=1))
            chunks.append(np.stack([loc[outside, 1], loc[outside, 2]], axis=1))
    if not chunks:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(chunks, axis=0).astype(np.int64, copy=False)


def _chain_edge_pairs(pairs: np.ndarray, point_of) -> list[list[tuple[float, float]]]:
    """Join segments that share an edge id. The loop touches segments only."""
    nseg = int(pairs.shape[0])
    if nseg == 0:
        return []
    by_edge: dict[int, list[int]] = {}
    for index in range(nseg):
        for edge in (int(pairs[index, 0]), int(pairs[index, 1])):
            by_edge.setdefault(edge, []).append(index)
    used = np.zeros(nseg, dtype=bool)

    def walk(tip: int) -> list[int]:
        found: list[int] = []
        while True:
            nxt = -1
            for cand in by_edge.get(tip, ()):
                if not used[cand]:
                    nxt = cand
                    break
            if nxt < 0:
                return found
            used[nxt] = True
            left = int(pairs[nxt, 0])
            right = int(pairs[nxt, 1])
            other = right if left == tip else left
            found.append(other)
            tip = other

    polylines: list[list[tuple[float, float]]] = []
    for start in range(nseg):
        if used[start]:
            continue
        used[start] = True
        edge0 = int(pairs[start, 0])
        edge1 = int(pairs[start, 1])
        if edge0 == edge1:
            continue
        forward = walk(edge1)
        backward = walk(edge0)
        edges = list(reversed(backward)) + [edge0, edge1] + forward
        polylines.append([point_of(edge) for edge in edges])
    return polylines


def _cell_cases(
    field: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(case, finite_active, center_inside)`` for each cell."""
    f00 = field[:-1, :-1]
    f10 = field[:-1, 1:]
    f11 = field[1:, 1:]
    f01 = field[1:, :-1]
    finite = (
        np.isfinite(f00) & np.isfinite(f10) & np.isfinite(f11) & np.isfinite(f01)
    )
    case = (
        (f00 >= level).astype(np.uint8)
        | ((f10 >= level).astype(np.uint8) << 1)
        | ((f11 >= level).astype(np.uint8) << 2)
        | ((f01 >= level).astype(np.uint8) << 3)
    )
    active = finite & (case != 0) & (case != 15)
    center_inside = (0.25 * (f00 + f10 + f11 + f01)) >= level
    return case, active, center_inside


def _contour_numpy(
    xs: np.ndarray,
    ys: np.ndarray,
    field: np.ndarray,
    level: float,
) -> list[list[tuple[float, float]]]:
    """Marching squares over the whole grid. Cells are classified in NumPy."""
    field = np.asarray(field, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    ny, nx = field.shape
    if ny < 2 or nx < 2:
        return []
    case, active, center_inside = _cell_cases(field, level)
    if not np.any(active):
        return []
    jj, ii = np.nonzero(active)
    cases = case[jj, ii]
    centers = center_inside[jj, ii]

    h_count = ny * (nx - 1)
    bottom = jj.astype(np.int64) * (nx - 1) + ii.astype(np.int64)
    top = (jj.astype(np.int64) + 1) * (nx - 1) + ii.astype(np.int64)
    left = h_count + jj.astype(np.int64) * nx + ii.astype(np.int64)
    right = h_count + jj.astype(np.int64) * nx + (ii.astype(np.int64) + 1)
    local = np.stack([bottom, right, top, left], axis=1)
    pairs = _pair_segments(local, cases, centers)
    if pairs.size == 0:
        return []

    h_a = field[:, :-1]
    h_b = field[:, 1:]
    h_t = _interp_t(h_a, h_b, level)
    h_x = xs[:-1][None, :] + h_t * (xs[1:] - xs[:-1])[None, :]
    h_y = np.broadcast_to(ys[:, None], h_x.shape)
    v_a = field[:-1, :]
    v_b = field[1:, :]
    v_t = _interp_t(v_a, v_b, level)
    v_y = ys[:-1][:, None] + v_t * (ys[1:] - ys[:-1])[:, None]
    v_x = np.broadcast_to(xs[None, :], v_y.shape)
    point_x = np.concatenate([np.ravel(h_x), np.ravel(v_x)])
    point_y = np.concatenate([np.ravel(h_y), np.ravel(v_y)])

    def point_of(edge: int) -> tuple[float, float]:
        return (float(point_x[edge]), float(point_y[edge]))

    return _chain_edge_pairs(pairs, point_of)


def _as_polyline(line: np.ndarray) -> list[tuple[float, float]] | None:
    arr = np.asarray(line, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    points = [(float(row[0]), float(row[1])) for row in arr]
    if len(points) >= 3:
        x0, y0 = points[0]
        x1, y1 = points[-1]
        # Closed loops repeat the first point. Snap a numeric near-close too.
        if abs(x0 - x1) <= 1e-9 and abs(y0 - y1) <= 1e-9:
            points[-1] = points[0]
    return points


def _contour_contourpy(
    xs: np.ndarray,
    ys: np.ndarray,
    field: np.ndarray,
    level: float,
) -> list[list[tuple[float, float]]]:
    contourpy = _load_contourpy()
    if contourpy is None:
        raise ValueError(
            "PLOT3_CONTOUR_BACKEND=contourpy but contourpy>=1.0 is not "
            "installed. Install it with: pip install contourpy"
        )
    masked = np.ma.masked_invalid(np.asarray(field, dtype=np.float64))
    generator = contourpy.contour_generator(
        x=np.asarray(xs, dtype=np.float64),
        y=np.asarray(ys, dtype=np.float64),
        z=masked,
        line_type=contourpy.LineType.Separate,
    )
    polylines: list[list[tuple[float, float]]] = []
    for line in generator.lines(float(level)):
        poly = _as_polyline(line)
        if poly is not None:
            polylines.append(poly)
    return polylines


def _warn_contourpy_fallback(exc: BaseException) -> None:
    global _contourpy_warned
    if _contourpy_warned:
        return
    _contourpy_warned = True
    warnings.warn(
        f"plot3: contourpy failed ({exc}), using the NumPy contour backend",
        UserWarning,
        stacklevel=4,
    )


def _contour_lines(
    xs: np.ndarray,
    ys: np.ndarray,
    field: np.ndarray,
    level: float = 0.0,
) -> list[list[tuple[float, float]]]:
    """Contour ``field`` at ``level``. Backend comes from the environment."""
    choice = _backend_setting()
    if choice == "numpy" or (choice == "auto" and _load_contourpy() is None):
        return _contour_numpy(xs, ys, field, level)
    try:
        return _contour_contourpy(xs, ys, field, level)
    except Exception as exc:
        if choice == "contourpy":
            raise
        _warn_contourpy_fallback(exc)
        return _contour_numpy(xs, ys, field, level)


def _refine_active_cells(
    xs: np.ndarray,
    ys: np.ndarray,
    field: np.ndarray,
    level: float,
    sample,
    sub: int = _REFINE_SUB,
) -> list[list[tuple[float, float]]] | None:
    """Re-sample only the cells the contour crosses, on a finer sub-grid.

    Returns None when the fine grid would be too large; the caller keeps the
    coarse contour, which is already a correct curve.
    """
    field = np.asarray(field, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    _case, active, _center = _cell_cases(field, level)
    if not np.any(active):
        return []
    lines = _refine_cells(xs, ys, active, level, sample, sub)
    # A curve that stops inside the window left the refined cells through
    # a cell whose corners all share a sign: a cusp, or a wedge thinner
    # than one cell. Refine around each stray end until the curve goes on.
    for _ in range(_REFINE_GROW):
        if lines is None:
            return None
        ends = _stray_ends(lines, xs, ys)
        if not ends:
            break
        grown = active.copy()
        for x_val, y_val in ends:
            i = int(np.clip(np.searchsorted(xs, x_val) - 1, 0, len(xs) - 2))
            j = int(np.clip(np.searchsorted(ys, y_val) - 1, 0, len(ys) - 2))
            grown[max(0, j - 1) : j + 2, max(0, i - 1) : i + 2] = True
        if grown.sum() == active.sum():
            break
        active = grown
        lines = _refine_cells(xs, ys, active, level, sample, sub)
    if lines is None:
        return None
    gap = 2.0 * max(float(np.max(np.diff(xs))), float(np.max(np.diff(ys)))) / max(1, int(sub))
    return _join_close_ends(lines, xs, ys, gap)


def _stray_ends(lines, xs, ys) -> list[tuple[float, float]]:
    """End points of open polylines that are not on the window's edge."""
    tol_x = 1e-9 * max(1.0, abs(float(xs[-1] - xs[0])))
    tol_y = 1e-9 * max(1.0, abs(float(ys[-1] - ys[0])))
    out = []
    for line in lines:
        if len(line) < 2 or line[0] == line[-1]:
            continue
        for x_val, y_val in (line[0], line[-1]):
            on_edge = (
                abs(x_val - xs[0]) <= tol_x or abs(x_val - xs[-1]) <= tol_x
                or abs(y_val - ys[0]) <= tol_y or abs(y_val - ys[-1]) <= tol_y
            )
            if not on_edge:
                out.append((x_val, y_val))
    return out


def _join_close_ends(lines, xs, ys, gap: float):
    """Join open polylines whose stray ends are within ``gap`` (a cusp tip)."""
    lines = [list(line) for line in lines if len(line) >= 2]
    while True:
        stray = set(_stray_ends(lines, xs, ys))
        ends = [
            (a, at_end, line[-1] if at_end else line[0])
            for a, line in enumerate(lines)
            for at_end in (False, True)
            if line[0] != line[-1] and (line[-1] if at_end else line[0]) in stray
        ]
        pair = next(
            (
                (p, q)
                for k, p in enumerate(ends)
                for q in ends[k + 1 :]
                if (p[0], p[1]) != (q[0], q[1])
                and np.hypot(p[2][0] - q[2][0], p[2][1] - q[2][1]) <= gap
            ),
            None,
        )
        if pair is None:
            return lines
        (a, a_end, _pa), (b, b_end, _pb) = pair
        if a == b:
            lines[a].append(lines[a][0])  # both ends of one line: close it
            continue
        first = lines[a] if a_end else lines[a][::-1]  # ends at the joint
        second = lines[b] if not b_end else lines[b][::-1]  # starts at it
        lines[a] = first + second
        del lines[b]


def _refine_cells(xs, ys, active, level, sample, sub):
    jj, ii = np.nonzero(active)
    sub = int(sub)
    nodes = sub + 1
    if len(ii) * nodes * nodes > _REFINE_NODE_BUDGET:
        sub = 4
        nodes = sub + 1
    if len(ii) * nodes * nodes > _REFINE_NODE_BUDGET:
        return None
    step = np.linspace(0.0, 1.0, nodes)
    x0 = xs[ii]
    x1 = xs[ii + 1]
    y0 = ys[jj]
    y1 = ys[jj + 1]
    xf = x0[:, None] + (x1 - x0)[:, None] * step
    yf = y0[:, None] + (y1 - y0)[:, None] * step
    xx = np.empty((len(ii), nodes, nodes), dtype=np.float64)
    yy = np.empty((len(ii), nodes, nodes), dtype=np.float64)
    xx[:] = xf[:, None, :]
    yy[:] = yf[:, :, None]
    fine = np.asarray(sample(xx, yy), dtype=np.float64)
    if fine.shape != xx.shape:
        fine = np.broadcast_to(fine, xx.shape).astype(np.float64, copy=False)
    return _contour_refined(ii, jj, xs, ys, fine, xf, yf, level, sub, sample)


def _contour_refined(
    ii: np.ndarray,
    jj: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    fine: np.ndarray,
    xf: np.ndarray,
    yf: np.ndarray,
    level: float,
    sub: int,
    sample=None,
) -> list[list[tuple[float, float]]]:
    """Marching squares on a batch of refined cells, chained by global edge id."""
    f00 = fine[:, :-1, :-1]
    f10 = fine[:, :-1, 1:]
    f11 = fine[:, 1:, 1:]
    f01 = fine[:, 1:, :-1]
    finite = (
        np.isfinite(f00) & np.isfinite(f10) & np.isfinite(f11) & np.isfinite(f01)
    )
    case = (
        (f00 >= level).astype(np.uint8)
        | ((f10 >= level).astype(np.uint8) << 1)
        | ((f11 >= level).astype(np.uint8) << 2)
        | ((f01 >= level).astype(np.uint8) << 3)
    )
    active = finite & (case != 0) & (case != 15)
    if not np.any(active):
        return []
    aj, lj, li = np.nonzero(active)
    cases = case[aj, lj, li]
    center_inside = (0.25 * (f00 + f10 + f11 + f01))[aj, lj, li]
    gi = ii[aj].astype(np.int64) * sub + li.astype(np.int64)
    gj = jj[aj].astype(np.int64) * sub + lj.astype(np.int64)
    nx = int(xs.shape[0])
    ny = int(ys.shape[0])
    h_stride = (nx - 1) * sub
    v_stride = (nx - 1) * sub + 1
    n_horizontal = h_stride * ((ny - 1) * sub + 1)  # rows of cells + 1 rows of edges
    bottom = gj * h_stride + gi
    top = (gj + 1) * h_stride + gi
    left = n_horizontal + gj * v_stride + gi
    right = n_horizontal + gj * v_stride + (gi + 1)
    local = np.stack([bottom, right, top, left], axis=1)
    pairs = _pair_segments(local, cases, center_inside)
    if pairs.size == 0:
        return []

    points: dict[int, tuple[float, float]] = {}

    def put(edge_ids: np.ndarray, x_vals: np.ndarray, y_vals: np.ndarray) -> None:
        for edge, x_val, y_val in zip(edge_ids.tolist(), x_vals.tolist(), y_vals.tolist()):
            points[int(edge)] = (float(x_val), float(y_val))

    x0, x1 = xf[aj, li], xf[aj, li + 1]
    y0, y1 = yf[aj, lj], yf[aj, lj + 1]
    f00, f10 = fine[aj, lj, li], fine[aj, lj, li + 1]
    f01, f11 = fine[aj, lj + 1, li], fine[aj, lj + 1, li + 1]
    t = _edge_root(sample, f00, f10, x0, y0, x1, y0, level)
    put(bottom, x0 + t * (x1 - x0), y0)
    t = _edge_root(sample, f10, f11, x1, y0, x1, y1, level)
    put(right, x1, y0 + t * (y1 - y0))
    t = _edge_root(sample, f01, f11, x0, y1, x1, y1, level)
    put(top, x0 + t * (x1 - x0), y1)
    t = _edge_root(sample, f00, f01, x0, y0, x0, y1, level)
    put(left, x0, y0 + t * (y1 - y0))

    def point_of(edge: int) -> tuple[float, float]:
        return points[edge]

    return _chain_edge_pairs(pairs, point_of)
