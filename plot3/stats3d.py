"""3D stats helpers: regular-grid surfaces, density grids, isosurfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regular_grid_mesh(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    zcol: str,
    ccol: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    """Build ordered vertices + triangle indices for a regular surface grid.

    Returns vertices (y-slow, x-fast), triangle indices ``(ntri, 3)``, nx, ny.
    """
    for name, col in (("x", xcol), ("y", ycol), ("z", zcol)):
        if col not in df.columns:
            raise KeyError(f"column(s) not in DataFrame: {[col]}")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"geom_surface() {name}={col!r} must be numeric")

    cols = [xcol, ycol, zcol]
    if ccol and ccol not in cols:
        cols.append(ccol)
    work = df.loc[:, cols].dropna(subset=[xcol, ycol, zcol])
    work = work.drop_duplicates(subset=[xcol, ycol], keep="first")

    xs = np.sort(pd.unique(work[xcol].to_numpy()))
    ys = np.sort(pd.unique(work[ycol].to_numpy()))
    nx, ny = int(len(xs)), int(len(ys))
    if nx < 2 or ny < 2:
        raise ValueError(
            "geom_surface() needs at least a 2×2 grid "
            f"(got nx={nx}, ny={ny})"
        )
    expected = nx * ny
    if len(work) != expected:
        raise ValueError(
            "geom_surface() requires a complete regular x–y grid: "
            f"expected {expected} unique (x,y) cells from "
            f"{nx}×{ny} levels, got {len(work)}"
        )

    full = pd.MultiIndex.from_product([ys, xs], names=[ycol, xcol])
    indexed = work.set_index([ycol, xcol]).reindex(full)
    if bool(indexed[zcol].isna().any()):
        raise ValueError(
            "geom_surface() grid has missing z values after alignment"
        )

    z_grid = indexed[zcol].to_numpy(dtype=np.float64).reshape(ny, nx)
    xx, yy = np.meshgrid(xs, ys)
    vertices = pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": z_grid.ravel(),
        }
    )
    if ccol:
        if ccol == zcol:
            vertices["colour"] = z_grid.ravel()
        else:
            c_grid = indexed[ccol].to_numpy(dtype=np.float64).reshape(ny, nx)
            if not np.isfinite(c_grid).all():
                c_grid = np.where(np.isfinite(c_grid), c_grid, z_grid)
            vertices["colour"] = c_grid.ravel()

    tris: list[list[int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = j * nx + (i + 1)
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + (i + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    indices = np.asarray(tris, dtype=np.int32)
    return vertices, indices, nx, ny


def density_grid_3d(
    points: np.ndarray,
    *,
    n: int = 32,
    pad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Histogram density on a cubic grid.

    Parameters
    ----------
    points:
        Array shape ``(N, 3)`` of finite xyz samples.
    n:
        Bins per axis (clamped to 8..64 for HTML size).
    pad:
        Fractional padding of the data range on each side.

    Returns
    -------
    density:
        ``(n, n, n)`` float64 field normalized so ``max == 1`` (or zeros).
    xs, ys, zs:
        1D bin-center coordinates along each axis.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("density_grid_3d() expects an (N, 3) point array")
    pts = pts[np.isfinite(pts).all(axis=1)]
    n = int(max(8, min(64, n)))
    if pts.shape[0] == 0:
        grid = np.linspace(-1, 1, n)
        return np.zeros((n, n, n), dtype=np.float64), grid, grid, grid

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    lo = lo - pad * span
    hi = hi + pad * span
    # histogramdd returns density with shape (nx, ny, nz) matching bins order
    hist, edges = np.histogramdd(pts, bins=n, range=list(zip(lo, hi)))
    dens = hist.astype(np.float64)
    peak = float(dens.max())
    if peak > 0:
        dens /= peak
    xs = 0.5 * (edges[0][:-1] + edges[0][1:])
    ys = 0.5 * (edges[1][:-1] + edges[1][1:])
    zs = 0.5 * (edges[2][:-1] + edges[2][1:])
    # Extra smoothing so isolevels are less blocky.
    dens = _smooth3(dens, passes=2)
    peak = float(dens.max())
    if peak > 0:
        dens /= peak
    return dens, xs, ys, zs


def _smooth3(vol: np.ndarray, passes: int = 1) -> np.ndarray:
    """6-neighbor average (edge-padded), optionally repeated."""
    out = np.asarray(vol, dtype=np.float64)
    for _ in range(max(1, int(passes))):
        p = np.pad(out, 1, mode="edge")
        out = (
            p[1:-1, 1:-1, 1:-1] * 0.4
            + p[:-2, 1:-1, 1:-1] * 0.1
            + p[2:, 1:-1, 1:-1] * 0.1
            + p[1:-1, :-2, 1:-1] * 0.1
            + p[1:-1, 2:, 1:-1] * 0.1
            + p[1:-1, 1:-1, :-2] * 0.1
            + p[1:-1, 1:-1, 2:] * 0.1
        )
    return out


def isosurface_mesh(
    density: np.ndarray,
    level: float,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a triangle mesh where ``density >= level`` meets empty space.

    Dependency-free **voxel face** extraction: for each solid voxel, emit
    quads on faces adjacent to empty (or boundary). Robust for interactive
    EDA; slightly blocky at low ``n``.

    Returns vertices ``(V, 3)`` and triangle indices ``(T, 3)``.
    """
    vol = np.asarray(density, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError("density must be a 3D array")
    nx, ny, nz = vol.shape
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    level = float(level)
    solid = vol >= level
    if not solid.any():
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

    def centers(axis: np.ndarray) -> np.ndarray:
        if len(axis) == 1:
            return axis.copy()
        # half-bin edges for face placement
        return axis

    def half_step(axis: np.ndarray) -> float:
        if len(axis) < 2:
            return 1.0
        return float(np.median(np.diff(axis)) * 0.5)

    hx, hy, hz = half_step(xs), half_step(ys), half_step(zs)
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    def add_quad(corners: list[tuple[float, float, float]], flip: bool = False):
        base = len(verts)
        for c in corners:
            verts.append([c[0], c[1], c[2]])
        if flip:
            faces.append([base, base + 2, base + 1])
            faces.append([base, base + 3, base + 2])
        else:
            faces.append([base, base + 1, base + 2])
            faces.append([base, base + 2, base + 3])

    # 6 neighbor offsets and corresponding face corner templates in local ±half
    neighbors = [
        # (di,dj,dk), four corners relative (sx,sy,sz) in ±1 for the face
        (1, 0, 0, [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)], False),
        (-1, 0, 0, [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)], False),
        (0, 1, 0, [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)], False),
        (0, -1, 0, [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)], False),
        (0, 0, 1, [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], False),
        (0, 0, -1, [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)], False),
    ]

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not solid[i, j, k]:
                    continue
                cx, cy, cz = float(xs[i]), float(ys[j]), float(zs[k])
                for di, dj, dk, corners, flip in neighbors:
                    ii, jj, kk = i + di, j + dj, k + dk
                    outside = (
                        ii < 0
                        or jj < 0
                        or kk < 0
                        or ii >= nx
                        or jj >= ny
                        or kk >= nz
                        or not solid[ii, jj, kk]
                    )
                    if not outside:
                        continue
                    # Interpolate face position toward empty neighbor for less blockiness
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        va = vol[i, j, k]
                        vb = vol[ii, jj, kk]
                        t = 0.5 if abs(vb - va) < 1e-15 else (level - va) / (vb - va)
                        t = float(np.clip(t, 0.0, 1.0))
                    else:
                        t = 0.5
                    # Face center between voxel centers
                    fcx = cx + di * hx * 2 * t
                    fcy = cy + dj * hy * 2 * t
                    fcz = cz + dk * hz * 2 * t
                    # Build quad in the plane perpendicular to (di,dj,dk)
                    world = []
                    for sx, sy, sz in corners:
                        # project local face offsets onto the face plane
                        if di != 0:
                            world.append(
                                (fcx, cy + sy * hy, cz + sz * hz)
                            )
                        elif dj != 0:
                            world.append(
                                (cx + sx * hx, fcy, cz + sz * hz)
                            )
                        else:
                            world.append(
                                (cx + sx * hx, cy + sy * hy, fcz)
                            )
                    add_quad(world, flip=flip)

    if not verts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)
    V = _laplacian_smooth(V, F, iterations=2)
    return V, F


def _laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 2,
    lambda_: float = 0.45,
) -> np.ndarray:
    """Relax mesh vertices toward neighbor averages (boundary-friendly)."""
    if len(vertices) == 0 or len(faces) == 0 or iterations < 1:
        return vertices
    n = len(vertices)
    # undirected adjacency
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if 0 <= u < n and 0 <= v < n and u != v:
                nbrs[u].add(int(v))
                nbrs[v].add(int(u))
    V = vertices.copy()
    for _ in range(int(iterations)):
        new = V.copy()
        for i in range(n):
            if not nbrs[i]:
                continue
            avg = V[list(nbrs[i])].mean(axis=0)
            new[i] = (1.0 - lambda_) * V[i] + lambda_ * avg
        V = new
    return V


def isosurface_levels(
    points: np.ndarray,
    levels: list[float] | tuple[float, ...],
    *,
    n: int = 32,
    absolute: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, list[float]]:
    """Density-grid + multi-level isosurface mesh.

    Parameters
    ----------
    points:
        ``(N, 3)`` xyz samples.
    levels:
        Relative to max density in ``[0, 1]`` unless ``absolute=True``.
    n:
        Grid resolution per axis.
    absolute:
        If True, treat levels as raw density thresholds in ``[0, 1]`` after
        normalization (same scale); kept for API clarity.

    Returns
    -------
    vertices:
        Columns x, y, z, level (numeric level id 0..L-1), colour (= level).
    indices:
        Triangle indices into vertices.
    used_levels:
        Absolute thresholds applied.
    """
    dens, xs, ys, zs = density_grid_3d(points, n=n)
    peak = float(dens.max())
    if peak <= 0:
        empty = pd.DataFrame(columns=["x", "y", "z", "level", "colour"])
        return empty, np.zeros((0, 3), dtype=np.int32), []

    used: list[float] = []
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_level: list[np.ndarray] = []
    v_offset = 0
    for li, raw in enumerate(levels):
        thr = float(raw)
        if not absolute:
            thr = float(np.clip(thr, 0.0, 1.0)) * peak
            # dens already normalized to max 1, so relative level is thr as-is
            thr = float(np.clip(raw, 0.0, 1.0))
        thr = float(np.clip(thr, 1e-6, 1.0 - 1e-9))
        verts, faces = isosurface_mesh(dens, thr, xs, ys, zs)
        if len(verts) == 0 or len(faces) == 0:
            continue
        used.append(thr)
        all_verts.append(verts)
        all_faces.append(faces + v_offset)
        all_level.append(np.full(len(verts), li, dtype=np.float64))
        v_offset += len(verts)

    if not all_verts:
        empty = pd.DataFrame(columns=["x", "y", "z", "level", "colour"])
        return empty, np.zeros((0, 3), dtype=np.int32), used

    V = np.vstack(all_verts)
    F = np.vstack(all_faces)
    L = np.concatenate(all_level)
    vertices = pd.DataFrame(
        {
            "x": V[:, 0],
            "y": V[:, 1],
            "z": V[:, 2],
            "level": L,
            "colour": L,
        }
    )
    return vertices, F.astype(np.int32), used
