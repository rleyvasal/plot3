"""3D stats helpers: regular-grid surfaces, density grids, isosurfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plot3.table import ColumnNotFound


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
            raise ColumnNotFound([col], df)
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


# The cube split into six tetrahedra around its 0-6 diagonal. Every cube
# uses the same split, so neighbouring cubes share faces and the mesh has
# no cracks.
_CUBE_CORNERS = np.array(
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
)
_CUBE_TETS = np.array(
    [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
)


def isosurface_mesh(
    density: np.ndarray,
    level: float,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh of the surface ``density == level`` (marching tetrahedra).

    Vertices sit on grid edges, interpolated linearly, and are shared
    between neighbouring triangles, so the surface is closed and smooth.
    The grid is padded with empty cells, so a blob touching the edge is
    closed too. Triangles face outward, away from the dense side.

    Returns vertices ``(V, 3)`` and triangle indices ``(T, 3)``.
    """
    vol = np.asarray(density, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError("density must be a 3D array")
    level = float(level)
    empty = (np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32))
    if not (vol >= level).any():
        return empty
    axes = [_padded_axis(np.asarray(a, dtype=np.float64)) for a in (xs, ys, zs)]
    vol = np.pad(vol, 1, mode="constant", constant_values=min(0.0, level) - 1.0)
    nx, ny, nz = vol.shape
    flat = vol.ravel()

    # Global grid index of each tetrahedron's four corners.
    ci, cj, ck = np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
    )
    base = np.stack([ci.ravel(), cj.ravel(), ck.ravel()], axis=1)
    corner_ids = []
    for di, dj, dk in _CUBE_CORNERS:
        corner_ids.append(
            ((base[:, 0] + di) * ny + (base[:, 1] + dj)) * nz + (base[:, 2] + dk)
        )
    corner_ids = np.stack(corner_ids, axis=1)
    tets = corner_ids[:, _CUBE_TETS].reshape(-1, 4)
    inside = flat[tets] >= level
    count = inside.sum(axis=1)
    keep = (count > 0) & (count < 4)
    tets, inside, count = tets[keep], inside[keep], count[keep]
    if len(tets) == 0:
        return empty
    # Outside corners first, then inside ones.
    order = np.argsort(inside, axis=1, kind="stable")
    v = np.take_along_axis(tets, order, axis=1)

    edges = []  # (corner p, corner q, inside corner) per triangle, three edges each
    one = count == 1  # [o, o, o, i]
    if one.any():
        t = v[one]
        edges.append((np.stack([t[:, [3, 3, 3]], t[:, [0, 1, 2]]], -1), t[:, 3]))
    three = count == 3  # [o, i, i, i]
    if three.any():
        t = v[three]
        edges.append((np.stack([t[:, [0, 0, 0]], t[:, [1, 2, 3]]], -1), t[:, 1]))
    two = count == 2  # [o, o, i, i]: a quad, two triangles
    if two.any():
        t = v[two]
        ac, ad, bc, bd = t[:, [2, 0]], t[:, [2, 1]], t[:, [3, 0]], t[:, [3, 1]]
        edges.append((np.stack([ac, ad, bd], 1), t[:, 2]))
        edges.append((np.stack([ac, bd, bc], 1), t[:, 2]))
    tri_edges = np.concatenate([e for e, _ in edges])  # (T, 3, 2)
    tri_inside = np.concatenate([i for _, i in edges])

    # One vertex per grid edge, shared by every triangle that crosses it.
    lo = tri_edges.min(axis=2)
    hi = tri_edges.max(axis=2)
    keys = lo.astype(np.int64) * flat.size + hi
    unique, faces = np.unique(keys.ravel(), return_inverse=True)
    faces = faces.reshape(-1, 3)
    p, q = unique // flat.size, unique % flat.size
    fp, fq = flat[p], flat[q]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(fq - fp) < 1e-15, 0.5, (level - fp) / (fq - fp))
    w = np.clip(w, 0.0, 1.0)[:, None]
    pp, qq = _grid_points(p, axes, ny, nz), _grid_points(q, axes, ny, nz)
    verts = pp + w * (qq - pp)

    # Face each triangle away from its tetrahedron's inside corner.
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    toward = _grid_points(tri_inside, axes, ny, nz) - tri.mean(axis=1)
    flip = np.einsum("ij,ij->i", normal, toward) > 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    area = np.linalg.norm(normal, axis=1)
    faces = faces[area > 1e-18]
    return verts, faces.astype(np.int32)


def _padded_axis(axis: np.ndarray) -> np.ndarray:
    """``axis`` with one more step at each end (for the padded grid)."""
    step = float(np.median(np.diff(axis))) if len(axis) > 1 else 1.0
    return np.concatenate([[axis[0] - step], axis, [axis[-1] + step]])


def _grid_points(ids: np.ndarray, axes, ny: int, nz: int) -> np.ndarray:
    k = ids % nz
    j = (ids // nz) % ny
    i = ids // (ny * nz)
    return np.stack([axes[0][i], axes[1][j], axes[2][k]], axis=-1)


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
        Columns x, y, z, level (the threshold, relative to the peak
        density), colour (= level), as ggplot2's contours colour by level.
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
    for raw in levels:
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
        all_level.append(np.full(len(verts), thr, dtype=np.float64))
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


# Box corner order: bottom loop, then top loop (each closed), then the four
# uprights, as six polylines.
_BOX_LOOP = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]


def box3d_layers(geom, mapping: dict, data) -> list:
    """geom_box3d rows as wireframe line layers, one layer per class.

    One colour scale serves the whole figure, and a lidar scene colours its
    points by height. So each box class becomes its own fixed-colour layer
    with a legend entry, and the height scale stays free for the points.
    """
    from plot3.scales import ordered_levels
    from plot3.stat2d import _frame, _layer
    from plot3.table import has_column
    from plot3.themes import _THEMES

    need = ["x", "y", "z", "length", "width", "height"]
    missing = [k for k in need if not mapping.get(k)]
    if missing:
        raise ValueError(
            "geom_box3d() requires aes(x=, y=, z=, length=, width=, height=) "
            f"(missing {', '.join(missing)}); angle= is the heading in radians"
        )
    colour = mapping.get("color")
    colour = colour if colour and has_column(data, colour) else None
    angle = mapping.get("angle")
    cols = [mapping[k] for k in need] + ([angle] if angle else []) + ([colour] if colour else [])
    frame = _frame(data, cols)
    values = {k: pd.to_numeric(frame[mapping[k]], errors="coerce").to_numpy(np.float64) for k in need}
    yaw = (
        pd.to_numeric(frame[angle], errors="coerce").to_numpy(np.float64)
        if angle else np.zeros(len(frame))
    )
    keys = frame[colour].astype(str).to_numpy() if colour else np.full(len(frame), "")
    levels = ordered_levels(list(dict.fromkeys(keys.tolist()))) if colour else [""]
    palette = getattr(geom, "_palette", None) or _THEMES["dark"]["cat"]
    layers = []
    for index, level in enumerate(levels):
        xs, ys, zs, starts = [], [], [], []
        for i in np.flatnonzero(keys == level):
            cx, cy, cz, ln, wd, ht = (values[k][i] for k in need)
            a = yaw[i] if np.isfinite(yaw[i]) else 0.0
            if not all(np.isfinite([cx, cy, cz, ln, wd, ht])):
                continue
            c, s = np.cos(a), np.sin(a)

            def corner(u, v, w):
                dx, dy = u * ln / 2.0, v * wd / 2.0
                return cx + c * dx - s * dy, cy + s * dx + c * dy, cz + w * ht / 2.0

            pieces = [
                [corner(u, v, -1) for u, v in _BOX_LOOP],
                [corner(u, v, 1) for u, v in _BOX_LOOP],
            ] + [[corner(u, v, -1), corner(u, v, 1)] for u, v in _BOX_LOOP[:4]]
            for piece in pieces:
                starts.append([len(xs), len(piece)])
                for px, py, pz in piece:
                    xs.append(px)
                    ys.append(py)
                    zs.append(pz)
        if not starts:
            continue
        out_frame = pd.DataFrame({"x": xs, "y": ys, "z": zs})
        out = _layer("line", out_frame, {"x": "x", "y": "y", "z": "z"}, geom,
                     _groups=starts, linewidth=float(getattr(geom, "linewidth", 1.5)))
        if colour:
            out.const_color = geom.const_color or palette[index % len(palette)]
            out._legend_label = level
            out._entries_title = str(colour)
        else:
            out.const_color = geom.const_color or "#ffffff"
        layers.append(out)
    if not layers:
        raise ValueError("geom_box3d() found no complete boxes (missing sizes or centres)")
    return layers
