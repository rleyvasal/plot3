"""3D stats helpers: regular-grid surfaces (and later density / isosurface)."""

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

    Parameters
    ----------
    df:
        Long-form data with one row per grid cell.
    xcol, ycol, zcol:
        Coordinate / height columns (must be numeric).
    ccol:
        Optional per-vertex colour/fill column.

    Returns
    -------
    vertices:
        DataFrame with columns x, y, z [, colour] in row-major order
        (y slowest, x fastest): vertex index = j * nx + i.
    indices:
        ``int32`` array shape ``(ntri, 3)`` of vertex indices.
    nx, ny:
        Grid dimensions.
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
    if indexed[zcol].isna().any():
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
