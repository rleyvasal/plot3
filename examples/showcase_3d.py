#!/usr/bin/env python3
"""Browser-first 3D showcase for plot3.

VS Code notebooks cannot run the WebGL viewer inline; these figures are written
as standalone HTML and opened in your system browser.

Usage (from plot3 repo root)::

    source .venv/bin/activate
    python examples/showcase_3d.py              # build all, open each
    python examples/showcase_3d.py --list
    python examples/showcase_3d.py helix peaks  # subset
    python examples/showcase_3d.py --no-open    # files only

Artifacts: examples/output/3d/
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from plot3 import (
    aes,
    coord_3d,
    geom_isosurface,
    geom_path,
    geom_point3d,
    geom_surface,
    ggplot,
    labs,
    scale_colour_viridis_c,
    stat_density_3d,
    theme_dark,
    theme_light,
)

OUT = Path(__file__).resolve().parent / "output" / "3d"
OUT.mkdir(parents=True, exist_ok=True)


# ── synthetic datasets ───────────────────────────────────────────────────────


def data_lidar_cloud(n: int = 8_000, seed: int = 0) -> pd.DataFrame:
    """Two-room lidar-ish point cloud with intensity falloff."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []

    def _slab(x0, x1, y0, y1, z0, z1, m, room):
        parts.append(
            pd.DataFrame(
                {
                    "x": rng.uniform(x0, x1, m),
                    "y": rng.uniform(y0, y1, m),
                    "z": rng.uniform(z0, z1, m),
                    "room": room,
                }
            )
        )

    # Room A: floor, ceiling, walls
    for z0, z1, dens in ((0.0, 0.08, 0.18), (2.4, 2.55, 0.12)):
        _slab(-3, 3, -2, 2, z0, z1, max(int(n * dens), 50), "A")
    m = max(int(n * 0.08), 40)
    parts.append(
        pd.DataFrame(
            {
                "x": rng.choice([-3.0, 3.0], m) + rng.normal(0, 0.02, m),
                "y": rng.uniform(-2, 2, m),
                "z": rng.uniform(0, 2.5, m),
                "room": "A",
            }
        )
    )
    # Room B (offset in +x)
    for z0, z1, dens in ((0.0, 0.08, 0.18), (2.4, 2.55, 0.12)):
        _slab(4, 9, -1.5, 2.5, z0, z1, max(int(n * dens), 50), "B")
    m = max(int(n * 0.08), 40)
    parts.append(
        pd.DataFrame(
            {
                "x": rng.uniform(4, 9, m),
                "y": rng.choice([-1.5, 2.5], m) + rng.normal(0, 0.02, m),
                "z": rng.uniform(0, 2.5, m),
                "room": "B",
            }
        )
    )
    # Volumetric clutter
    used = sum(len(p) for p in parts)
    m = max(n - used, 100)
    parts.append(
        pd.DataFrame(
            {
                "x": rng.uniform(-3, 9, m),
                "y": rng.uniform(-2, 2.5, m),
                "z": rng.uniform(0.1, 2.3, m),
                "room": rng.choice(["A", "B"], m),
            }
        )
    )
    df = pd.concat(parts, ignore_index=True)
    # Intensity: brighter near a virtual sensor at (0, 0, 1.5)
    d = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + (df["z"] - 1.5) ** 2)
    df["intensity"] = 1.0 / (1.0 + 0.15 * d**2) + rng.normal(0, 0.02, len(df))
    df["intensity"] = df["intensity"].clip(0.05, 1.0)
    return df


def data_galaxy(n: int = 6_000, seed: int = 1) -> pd.DataFrame:
    """Spiral arms + central bulge (demo point cloud)."""
    rng = np.random.default_rng(seed)
    n_arms = int(n * 0.7)
    n_bulge = n - n_arms
    # logarithmic spiral arms
    arm = rng.integers(0, 2, n_arms)
    t = rng.uniform(0.4, 4.5, n_arms)
    r = np.exp(0.35 * t) * 0.35
    theta = t + arm * np.pi + rng.normal(0, 0.08, n_arms)
    x = r * np.cos(theta) + rng.normal(0, 0.05, n_arms)
    y = r * np.sin(theta) + rng.normal(0, 0.05, n_arms)
    z = rng.normal(0, 0.08 + 0.02 * t, n_arms)
    arms = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "z": z,
            "component": np.where(arm == 0, "arm_a", "arm_b"),
            "radius": r,
        }
    )
    # bulge
    bulge = pd.DataFrame(
        {
            "x": rng.normal(0, 0.35, n_bulge),
            "y": rng.normal(0, 0.35, n_bulge),
            "z": rng.normal(0, 0.25, n_bulge),
            "component": "bulge",
            "radius": np.nan,
        }
    )
    bulge["radius"] = np.sqrt(bulge["x"] ** 2 + bulge["y"] ** 2)
    df = pd.concat([arms, bulge], ignore_index=True)
    df["brightness"] = np.exp(-df["radius"] / 2.2) + rng.uniform(0, 0.15, len(df))
    return df


def data_peaks(nx: int = 80, ny: int = 80) -> pd.DataFrame:
    """Classic multi-peak landscape (MATLAB-style peaks, simplified)."""
    xs = np.linspace(-3, 3, nx)
    ys = np.linspace(-3, 3, ny)
    xx, yy = np.meshgrid(xs, ys)
    zz = (
        3 * (1 - xx) ** 2 * np.exp(-(xx**2) - (yy + 1) ** 2)
        - 10 * (xx / 5 - xx**3 - yy**5) * np.exp(-(xx**2) - yy**2)
        - (1 / 3) * np.exp(-((xx + 1) ** 2) - yy**2)
    )
    return pd.DataFrame(
        {"x": xx.ravel(), "y": yy.ravel(), "height": zz.ravel()}
    )


def data_sombrero(nx: int = 70, ny: int = 70) -> pd.DataFrame:
    """Mexican-hat / sombrero surface."""
    xs = np.linspace(-8, 8, nx)
    ys = np.linspace(-8, 8, ny)
    xx, yy = np.meshgrid(xs, ys)
    r = np.sqrt(xx**2 + yy**2) + 1e-9
    zz = np.sin(r) / r
    return pd.DataFrame(
        {"x": xx.ravel(), "y": yy.ravel(), "height": zz.ravel()}
    )


def data_helix(n: int = 500, turns: int = 5) -> pd.DataFrame:
    """Double helix paths."""
    t = np.linspace(0, turns * 2 * np.pi, n)
    frames = []
    for phase, strand in ((0.0, "A"), (np.pi, "B")):
        frames.append(
            pd.DataFrame(
                {
                    "x": np.cos(t + phase),
                    "y": np.sin(t + phase),
                    "z": t / (turns * 2 * np.pi) * 4,
                    "strand": strand,
                    "t": t,
                }
            )
        )
    # base-pair rungs (sample)
    rung_idx = np.linspace(0, n - 1, turns * 8, dtype=int)
    rungs = []
    for i in rung_idx:
        rungs.append(
            pd.DataFrame(
                {
                    "x": [np.cos(t[i]), np.cos(t[i] + np.pi)],
                    "y": [np.sin(t[i]), np.sin(t[i] + np.pi)],
                    "z": [t[i] / (turns * 2 * np.pi) * 4] * 2,
                    "strand": "rung",
                    "t": [t[i], t[i]],
                }
            )
        )
    return pd.concat(frames + rungs, ignore_index=True)


def data_molten_blobs(n: int = 2_400, seed: int = 2) -> pd.DataFrame:
    """Three Gaussian clusters for density isosurfaces."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.6, 0.4, 0.3],
            [-0.8, 1.2, -0.5],
        ]
    )
    scales = [0.35, 0.28, 0.32]
    parts = []
    for c, s, name in zip(centers, scales, ("core", "arm", "satellite")):
        m = n // 3
        pts = rng.normal(loc=c, scale=s, size=(m, 3))
        parts.append(
            pd.DataFrame(
                {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "cluster": name}
            )
        )
    return pd.concat(parts, ignore_index=True)


def data_terrain_with_samples(nx: int = 50, ny: int = 50, n_pts: int = 120, seed: int = 3):
    """Rolling terrain surface + survey points on top."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-4, 4, nx)
    ys = np.linspace(-4, 4, ny)
    xx, yy = np.meshgrid(xs, ys)
    height = (
        0.6 * np.sin(0.9 * xx) * np.cos(0.7 * yy)
        + 0.25 * np.sin(1.7 * xx + 0.4 * yy)
        + 0.15 * np.cos(0.5 * xx * yy)
    )
    grid = pd.DataFrame(
        {"x": xx.ravel(), "y": yy.ravel(), "height": height.ravel(), "kind": "mesh"}
    )
    px = rng.uniform(-3.5, 3.5, n_pts)
    py = rng.uniform(-3.5, 3.5, n_pts)
    # sample height with same formula
    pz = (
        0.6 * np.sin(0.9 * px) * np.cos(0.7 * py)
        + 0.25 * np.sin(1.7 * px + 0.4 * py)
        + 0.15 * np.cos(0.5 * px * py)
        + 0.05
    )
    pts = pd.DataFrame(
        {"x": px, "y": py, "height": pz, "kind": "sample"}
    )
    return grid, pts


# ── figures ──────────────────────────────────────────────────────────────────


def fig_lidar():
    df = data_lidar_cloud()
    return (
        ggplot(df, aes(x="x", y="y", z="z", colour="intensity"))
        + geom_point3d(size=0.025, alpha=0.9)
        + scale_colour_viridis_c(option="turbo")
        + coord_3d(aspect="data", size_mode="scene")
        + theme_dark()
        + labs(
            title="Indoor scan (synthetic lidar)",
            x="x (m)",
            y="y (m)",
            z="z (m)",
            colour="intensity",
        )
    )


def fig_galaxy():
    df = data_galaxy()
    return (
        ggplot(df, aes(x="x", y="y", z="z", colour="brightness"))
        + geom_point3d(size=0.018, alpha=0.85)
        + scale_colour_viridis_c(option="magma")
        + coord_3d(aspect="equal", size_mode="scene")
        + theme_dark()
        + labs(title="Spiral galaxy (demo cloud)", colour="brightness")
    )


def fig_peaks():
    df = data_peaks()
    return (
        ggplot(df, aes(x="x", y="y", z="height", fill="height"))
        + geom_surface(alpha=0.95)
        + scale_colour_viridis_c(option="viridis")
        + coord_3d(aspect="data")
        + theme_dark()
        + labs(title="Peaks surface", z="height", colour="height")
    )


def fig_sombrero():
    df = data_sombrero()
    return (
        ggplot(df, aes(x="x", y="y", z="height", fill="height"))
        + geom_surface(wireframe=True, alpha=0.9)
        + scale_colour_viridis_c(option="turbo")
        + coord_3d(aspect="data")
        + theme_dark()
        + labs(title="Sombrero (wireframe)", z="sinc(r)", colour="height")
    )


def fig_helix():
    df = data_helix()
    # strands as paths; rungs as short paths (group by strand + discretize rungs)
    # geom_path sorts? path keeps order — split rungs with group
    df = df.copy()
    # unique group per rung segment
    is_rung = df["strand"] == "rung"
    rung_ids = np.zeros(len(df), dtype=int)
    rung_ids[is_rung] = np.arange(is_rung.sum()) // 2
    df["group"] = np.where(is_rung, "rung_" + rung_ids.astype(str), df["strand"])
    return (
        ggplot(df, aes(x="x", y="y", z="z", colour="strand", group="group"))
        + geom_path(linewidth=2.5, alpha=0.95)
        + coord_3d(aspect="equal")
        + theme_dark()
        + labs(title="Double helix", colour="strand")
    )


def fig_isosurface():
    df = data_molten_blobs()
    return (
        ggplot(df, aes(x="x", y="y", z="z"))
        + stat_density_3d(n=28)
        + geom_isosurface(levels=[0.2, 0.45, 0.7], n=28, alpha=0.4)
        + coord_3d(aspect="equal")
        + theme_dark()
        + labs(title="Density isosurfaces (3 clusters)")
    )


def fig_terrain_survey():
    grid, pts = data_terrain_with_samples()
    # surface from grid; points need height as z
    # plot3 multi-layer uses figure data — combine and map carefully
    # Surface uses full grid; for points we use a second figure approach:
    # single dataframe with mesh rows + point rows, surface needs complete grid only.
    # So build surface fig and... multi-layer needs same columns.
    # Use surface alone from grid; overlay points via second layer on combined
    # won't work if surface requires complete grid on all rows.
    # Build surface figure from grid only; open points separately OR
    # use only points on terrain for one fig and surface for another.
    # Better: two-layer if surface expansion uses only mapped cols and complete grid subset.
    # From build.py, surface uses the full data — incomplete fails.
    # So showcase surface and points as one combined visual using just points + a denser path:
    # Actually: save surface as main; document. For true overlay, check if layer-local data exists...
    # geoms don't take data= like ggplot2. Surface-only + separate points file.
    # Compromise: denser terrain as points colored by height (looks good) AND full surface fig.
    return (
        ggplot(grid, aes(x="x", y="y", z="height", fill="height"))
        + geom_surface(alpha=0.88)
        + scale_colour_viridis_c(option="viridis")
        + coord_3d(aspect="data")
        + theme_light()
        + labs(title="Rolling terrain", z="elevation", colour="elevation")
    )


def fig_terrain_samples():
    _, pts = data_terrain_with_samples()
    return (
        ggplot(pts, aes(x="x", y="y", z="height", colour="height"))
        + geom_point3d(size=0.06, alpha=0.95)
        + scale_colour_viridis_c(option="magma")
        + coord_3d(aspect="data")
        + theme_light()
        + labs(title="Survey samples on terrain", z="elevation", colour="elev")
    )


def fig_rgb_cube(n_side: int = 12) -> "ggplot":
    """Discrete RGB lattice — categorical-ish continuous colour via viridis on index."""
    g = np.linspace(0, 1, n_side)
    rr, gg, bb = np.meshgrid(g, g, g, indexing="ij")
    df = pd.DataFrame(
        {
            "r": rr.ravel(),
            "g": gg.ravel(),
            "b": bb.ravel(),
            "luma": 0.2126 * rr.ravel() + 0.7152 * gg.ravel() + 0.0722 * bb.ravel(),
        }
    )
    return (
        ggplot(df, aes(x="r", y="g", z="b", colour="luma"))
        + geom_point3d(size=0.035, alpha=0.9)
        + scale_colour_viridis_c(option="turbo")
        + coord_3d(aspect="equal", size_mode="scene")
        + theme_dark()
        + labs(title="RGB cube (luma colour)", x="R", y="G", z="B", colour="luma")
    )


SHOWCASES: dict[str, callable] = {
    "lidar": fig_lidar,
    "galaxy": fig_galaxy,
    "peaks": fig_peaks,
    "sombrero": fig_sombrero,
    "helix": fig_helix,
    "isosurface": fig_isosurface,
    "terrain": fig_terrain_survey,
    "survey": fig_terrain_samples,
    "rgb": fig_rgb_cube,
}


def build_one(name: str, *, open_browser: bool) -> Path:
    fig = SHOWCASES[name]()
    path = OUT / f"{name}.html"
    fig.save(str(path))
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="plot3 3D showcase (browser HTML)")
    p.add_argument(
        "names",
        nargs="*",
        help=f"subset of: {', '.join(SHOWCASES)} (default: all)",
    )
    p.add_argument("--list", action="store_true", help="list showcase names")
    p.add_argument(
        "--no-open",
        action="store_true",
        help="write HTML only (do not open browser)",
    )
    args = p.parse_args(argv)

    if args.list:
        for k in SHOWCASES:
            print(k)
        return

    names = args.names or list(SHOWCASES)
    unknown = [n for n in names if n not in SHOWCASES]
    if unknown:
        raise SystemExit(f"unknown showcase(s): {unknown}; choose from {list(SHOWCASES)}")

    open_each = (not args.no_open) and len(names) <= 4
    print(f"plot3 3D showcase → {OUT}")
    built: list[tuple[str, Path]] = []
    for name in names:
        path = build_one(name, open_browser=open_each)
        built.append((name, path))
        print(f"  {name:12s}  {path.name}  ({path.stat().st_size // 1024} KB)")

    # Gallery index for full runs (avoid opening 9 browser tabs).
    cards = "".join(
        f'<a class="card" href="{p.name}" target="_blank"><strong>{n}</strong>'
        f"<span>{p.name}</span></a>"
        for n, p in built
    )
    index = OUT / "index.html"
    index.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>plot3 · 3D gallery</title>
<style>
body{{margin:0;font:15px/1.45 system-ui,sans-serif;background:#0b1020;color:#e2e8f0}}
header{{padding:28px 32px 8px}} h1{{margin:0 0 6px;font-size:22px}}
p{{margin:0;color:#94a3b8;max-width:52rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;padding:20px 32px 40px}}
.card{{display:flex;flex-direction:column;gap:4px;padding:14px 16px;border-radius:10px;
background:#1e293b;color:#e2e8f0;text-decoration:none;border:1px solid #334155}}
.card:hover{{border-color:#60a5fa}} .card span{{color:#94a3b8;font-size:12px}}
</style></head><body>
<header><h1>plot3 · 3D gallery</h1>
<p>Drag to orbit · scroll to zoom · double-click reset · hover for values.</p></header>
<div class="grid">{cards}</div></body></html>
""",
        encoding="utf-8",
    )
    print(f"  index       {index.name}")
    if not args.no_open and not open_each:
        webbrowser.open(index.resolve().as_uri())
    print("done — drag to orbit, scroll to zoom, hover for values.")


if __name__ == "__main__":
    main()
