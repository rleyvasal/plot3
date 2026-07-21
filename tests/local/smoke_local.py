#!/usr/bin/env python3
"""Write sample HTML figures for local visual inspection.

Run from repo root::

    python tests/local/smoke_local.py

Artifacts land in tests/local/output/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plot3 import (
    aes,
    coord_3d,
    facet_wrap,
    geom_boxplot,
    geom_col,
    geom_density,
    geom_histogram,
    geom_isosurface,
    geom_point,
    geom_point3d,
    geom_surface,
    geom_violin,
    ggplot,
    labs,
    scale_colour_viridis_c,
    theme_light,
)

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def save(name: str, fig) -> None:
    path = OUT / name
    fig.save(str(path))
    print(f"  wrote {path} ({path.stat().st_size // 1024} KB)")


def main() -> None:
    rng = np.random.default_rng(0)
    cars = pd.DataFrame(
        {
            "mpg": [21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2],
            "cyl": [6, 6, 4, 6, 8, 6, 8, 4, 4, 6],
            "wt": [2.62, 2.88, 2.32, 3.22, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44],
        }
    )
    cloud = pd.DataFrame(
        {
            "x": rng.normal(size=400),
            "y": rng.normal(size=400),
            "z": rng.normal(size=400),
            "intensity": rng.uniform(0, 1, 400),
        }
    )
    xs, ys = np.linspace(-2, 2, 30), np.linspace(-2, 2, 30)
    xx, yy = np.meshgrid(xs, ys)
    grid = pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "height": (np.sin(xx) * np.cos(yy)).ravel(),
        }
    )

    print("plot3 local smoke →", OUT)
    save(
        "01_scatter.html",
        ggplot(cars, aes(x="wt", y="mpg", colour="cyl"))
        + geom_point(size=6)
        + theme_light()
        + labs(title="scatter"),
    )
    save(
        "02_hist_density.html",
        ggplot(cars, aes(x="mpg"))
        + geom_histogram(bins=6, alpha=0.5)
        + labs(title="histogram"),
    )
    save(
        "03_density.html",
        ggplot(cars, aes(x="mpg", colour="cyl"))
        + geom_density(n=128)
        + labs(title="density by cyl"),
    )
    save(
        "04_boxplot.html",
        ggplot(cars, aes(x="cyl", y="mpg"))
        + geom_boxplot()
        + labs(title="boxplot"),
    )
    save(
        "05_violin.html",
        ggplot(cars, aes(x="cyl", y="mpg"))
        + geom_violin(n=64)
        + labs(title="violin"),
    )
    save(
        "06_facet.html",
        ggplot(cars, aes(x="wt", y="mpg"))
        + geom_point()
        + facet_wrap("cyl", ncol=2, scales="fixed")
        + labs(title="facet_wrap"),
    )
    save(
        "07_point3d.html",
        ggplot(cloud, aes(x="x", y="y", z="z", colour="intensity"))
        + geom_point3d(size=0.02)
        + scale_colour_viridis_c(option="turbo")
        + coord_3d()
        + labs(title="point3d"),
    )
    save(
        "08_surface.html",
        ggplot(grid, aes(x="x", y="y", z="height", fill="height"))
        + geom_surface()
        + scale_colour_viridis_c()
        + coord_3d()
        + labs(title="surface"),
    )
    save(
        "09_isosurface.html",
        ggplot(cloud, aes(x="x", y="y", z="z"))
        + geom_isosurface(levels=[0.25, 0.5], n=16, alpha=0.45)
        + coord_3d()
        + labs(title="isosurface"),
    )
    summary = cars.groupby("cyl", as_index=False)["mpg"].mean().rename(
        columns={"mpg": "avg"}
    )
    save(
        "10_col.html",
        ggplot(summary, aes(x="cyl", y="avg"))
        + geom_col()
        + labs(title="col"),
    )
    print("done.")


if __name__ == "__main__":
    main()
