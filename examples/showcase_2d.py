#!/usr/bin/env python3
"""Browser-first 2D showcase for plot3 — ggplot2 gallery style.

Inspired by the R Graph Gallery ggplot2 section
(https://r-graph-gallery.com/ggplot2-package.html): one self-contained
example per common geom / pattern that plot3 supports today.

Usage (from plot3 repo root)::

    source .venv/bin/activate
    python examples/showcase_2d.py                 # all, open each
    python examples/showcase_2d.py scatter boxplot # subset
    python examples/showcase_2d.py --list
    python examples/showcase_2d.py --no-open

Artifacts: examples/output/2d/
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from plot3 import (
    aes,
    facet_wrap,
    geom_bar,
    geom_boxplot,
    geom_col,
    geom_density,
    geom_histogram,
    geom_line,
    geom_path,
    geom_point,
    geom_violin,
    ggplot,
    labs,
    scale_colour_viridis_c,
    theme_dark,
    theme_light,
)

OUT = Path(__file__).resolve().parent / "output" / "2d"
OUT.mkdir(parents=True, exist_ok=True)


# ── datasets (gallery-style fixtures) ────────────────────────────────────────


def data_mtcars(seed: int = 0) -> pd.DataFrame:
    """mtcars-like sample used across many ggplot2 gallery posts."""
    rng = np.random.default_rng(seed)
    rows = []
    for cyl, mpg0, wt0, hp0 in (
        (4, 26.5, 2.2, 85),
        (6, 19.8, 3.1, 120),
        (8, 15.0, 4.0, 190),
    ):
        n = 18 if cyl != 8 else 14
        for _ in range(n):
            mpg = float(rng.normal(mpg0, 2.2))
            wt = float(rng.normal(wt0, 0.28))
            hp = float(max(rng.normal(hp0, 22), 45))
            rows.append(
                {
                    "mpg": mpg,
                    "cyl": str(cyl),  # categorical for legends / facets
                    "cyl_n": cyl,
                    "wt": wt,
                    "hp": hp,
                    "disp": float(rng.normal(80 + 35 * (cyl - 4), 25)),
                    "am": rng.choice(["auto", "manual"]),
                    "gear": str(rng.choice([3, 4, 5])),
                    "qsec": float(rng.normal(18 - 0.4 * (cyl - 4), 1.1)),
                }
            )
    return pd.DataFrame(rows)


def data_timeseries(seed: int = 1) -> pd.DataFrame:
    """Multi-series monthly index (geom_line gallery)."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=48, freq="MS")
    t = np.arange(len(months))
    frames = []
    for name, level, amp, noise in (
        ("North", 100, 12, 3.0),
        ("South", 80, 18, 4.0),
        ("West", 60, 8, 2.5),
    ):
        y = level + amp * np.sin(2 * np.pi * t / 12) + 0.35 * t + rng.normal(
            0, noise, len(t)
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": months,
                    "t": t.astype(float),
                    "value": y,
                    "region": name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def data_gapminder(seed: int = 2) -> pd.DataFrame:
    """One-year snapshot: GDP × lifeExp × continent (scatter mapping)."""
    rng = np.random.default_rng(seed)
    continents = {
        "Africa": (2_500, 58, 40),
        "Americas": (12_000, 74, 25),
        "Asia": (8_000, 72, 45),
        "Europe": (28_000, 79, 30),
        "Oceania": (22_000, 78, 8),
    }
    rows = []
    for cont, (gdp0, life0, n) in continents.items():
        gdp = rng.lognormal(np.log(gdp0), 0.55, n)
        life = rng.normal(life0, 3.5, n)
        pop = rng.lognormal(15.5, 1.2, n)
        for i in range(n):
            rows.append(
                {
                    "continent": cont,
                    "gdpPercap": float(gdp[i]),
                    "lifeExp": float(np.clip(life[i], 40, 90)),
                    "pop": float(pop[i]),
                    "log_gdp": float(np.log10(gdp[i])),
                }
            )
    return pd.DataFrame(rows)


def data_connected(seed: int = 3) -> pd.DataFrame:
    """Two variables over time for a connected scatterplot."""
    rng = np.random.default_rng(seed)
    n = 40
    t = np.arange(n)
    x = np.cumsum(rng.normal(0.4, 1.0, n))
    y = np.cumsum(rng.normal(0.2, 0.9, n))
    return pd.DataFrame({"year": 1980 + t, "x": x, "y": y, "t": t.astype(float)})


def data_categories(seed: int = 4) -> pd.DataFrame:
    """Discrete counts / bar chart inputs."""
    rng = np.random.default_rng(seed)
    cats = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(
        {
            "category": cats,
            "sales": rng.integers(40, 180, len(cats)).astype(float),
            "cost": rng.integers(20, 100, len(cats)).astype(float),
        }
    )


def data_long_counts(seed: int = 5) -> pd.DataFrame:
    """Raw observations for geom_bar (count)."""
    rng = np.random.default_rng(seed)
    species = rng.choice(
        ["setosa", "versicolor", "virginica"],
        size=150,
        p=[0.33, 0.40, 0.27],
    )
    return pd.DataFrame(
        {
            "species": species,
            "island": rng.choice(["Biscoe", "Dream", "Torgersen"], size=150),
        }
    )


# ── figures (one per gallery pattern) ───────────────────────────────────────


def fig_scatter():
    """Basic scatter — geom_point (gallery #272)."""
    df = data_mtcars()
    return (
        ggplot(df, aes(x="wt", y="mpg"))
        + geom_point(size=6, alpha=0.85)
        + theme_light()
        + labs(title="Basic scatterplot", x="weight", y="mpg")
    )


def fig_scatter_colour():
    """Map a categorical variable to colour (gallery #274)."""
    df = data_mtcars()
    return (
        ggplot(df, aes(x="wt", y="mpg", colour="cyl"))
        + geom_point(size=6, alpha=0.9)
        + theme_light()
        + labs(
            title="Scatter — colour by cylinders",
            x="weight",
            y="mpg",
            colour="cyl",
        )
    )


def fig_scatter_continuous():
    """Continuous colour ramp (viridis / turbo)."""
    df = data_gapminder()
    return (
        ggplot(df, aes(x="log_gdp", y="lifeExp", colour="pop"))
        + geom_point(size=5, alpha=0.85)
        + scale_colour_viridis_c(option="turbo", trans="log10")
        + theme_light()
        + labs(
            title="Scatter — continuous colour (log pop)",
            x="log10 GDP per cap",
            y="life expectancy",
            colour="population",
        )
    )


def fig_scatter_groups():
    """Gapminder-style continent colouring."""
    df = data_gapminder()
    return (
        ggplot(df, aes(x="log_gdp", y="lifeExp", colour="continent"))
        + geom_point(size=5, alpha=0.85)
        + theme_light()
        + labs(
            title="GDP vs life expectancy by continent",
            x="log10 GDP per cap",
            y="life expectancy",
            colour="continent",
        )
    )


def fig_line():
    """Single time series — geom_line (gallery line plots)."""
    df = data_timeseries()
    one = df[df["region"] == "North"]
    return (
        ggplot(one, aes(x="t", y="value"))
        + geom_line(linewidth=2.5)
        + geom_point(size=3, alpha=0.7)
        + theme_light()
        + labs(title="Line chart (single series)", x="month index", y="index")
    )


def fig_line_groups():
    """Multi-series lines with colour = group."""
    df = data_timeseries()
    return (
        ggplot(df, aes(x="t", y="value", colour="region"))
        + geom_line(linewidth=2.2)
        + theme_light()
        + labs(
            title="Multi-series line chart",
            x="month index",
            y="index",
            colour="region",
        )
    )


def fig_path():
    """Connected scatterplot — geom_path keeps data order (gallery connected)."""
    df = data_connected()
    return (
        ggplot(df, aes(x="x", y="y", colour="t"))
        + geom_path(linewidth=2.0)
        + geom_point(size=4, alpha=0.8)
        + scale_colour_viridis_c(option="viridis")
        + theme_light()
        + labs(
            title="Connected scatterplot (geom_path)",
            x="series A",
            y="series B",
            colour="time",
        )
    )


def fig_col():
    """Bars from values — geom_col (gallery barplots)."""
    df = data_categories()
    return (
        ggplot(df, aes(x="category", y="sales", colour="category"))
        + geom_col(width=0.75)
        + theme_light()
        + labs(title="Bar chart (geom_col)", x="category", y="sales")
    )


def fig_bar():
    """Count bars — geom_bar (gallery #218)."""
    df = data_long_counts()
    return (
        ggplot(df, aes(x="species", colour="species"))
        + geom_bar(width=0.7)
        + theme_light()
        + labs(title="Count bars (geom_bar)", x="species", y="count")
    )


def fig_histogram():
    """Histogram — geom_histogram (gallery #220)."""
    df = data_mtcars()
    return (
        ggplot(df, aes(x="mpg"))
        + geom_histogram(bins=12, alpha=0.85)
        + theme_light()
        + labs(title="Histogram", x="mpg", y="count")
    )


def fig_histogram_fill():
    """Histogram with colour groups (faceted view is clearer; here overlay colour)."""
    df = data_mtcars()
    # plot3 histogram is single-series; show overall + note facet alternative
    return (
        ggplot(df, aes(x="hp"))
        + geom_histogram(bins=14, alpha=0.85, colour="#3b82f6")
        + theme_light()
        + labs(title="Histogram of horsepower", x="hp", y="count")
    )


def fig_density():
    """Kernel density — geom_density (gallery #21)."""
    df = data_mtcars()
    return (
        ggplot(df, aes(x="mpg"))
        + geom_density(n=256, adjust=1.0, fill=True, linewidth=1.8)
        + theme_light()
        + labs(title="Density estimate", x="mpg", y="density")
    )


def fig_density_groups():
    """Density by group (gallery distribution plots)."""
    df = data_mtcars()
    return (
        ggplot(df, aes(x="mpg", colour="cyl"))
        + geom_density(n=256, adjust=1.1, fill=True, alpha=0.35, linewidth=1.6)
        + theme_light()
        + labs(
            title="Density by cylinders",
            x="mpg",
            y="density",
            colour="cyl",
        )
    )


def fig_boxplot():
    """Boxplot — geom_boxplot (gallery #262)."""
    df = data_mtcars()
    return (
        ggplot(df, aes(x="cyl", y="mpg", colour="cyl"))
        + geom_boxplot(width=0.55, outlier_size=4)
        + theme_light()
        + labs(title="Boxplot by cylinders", x="cyl", y="mpg")
    )


def fig_violin():
    """Violin — geom_violin (gallery #95)."""
    df = data_mtcars()
    return (
        ggplot(df, aes(x="cyl", y="mpg", colour="cyl"))
        + geom_violin(n=128, width=0.9, linewidth=1.2)
        + theme_light()
        + labs(title="Violin by cylinders", x="cyl", y="mpg")
    )


def fig_facet():
    """Small multiples — facet_wrap (gallery #223)."""
    df = data_mtcars()
    return (
        ggplot(df, aes(x="wt", y="mpg", colour="am"))
        + geom_point(size=5, alpha=0.9)
        + facet_wrap("cyl", ncol=3, scales="fixed")
        + theme_light()
        + labs(
            title="Facet wrap by cylinders",
            x="weight",
            y="mpg",
            colour="trans",
        )
    )


def fig_facet_density():
    """Faceted densities — distribution small multiples."""
    df = data_mtcars()
    return (
        ggplot(df, aes(x="hp", colour="cyl"))
        + geom_density(n=200, fill=True, alpha=0.4)
        + facet_wrap("am", ncol=2, scales="fixed")
        + theme_light()
        + labs(
            title="HP density faceted by transmission",
            x="hp",
            y="density",
            colour="cyl",
        )
    )


def fig_col_grouped():
    """Side-by-side style via long data + colour (two metrics as points/cols)."""
    df = data_categories().melt(
        id_vars=["category"],
        value_vars=["sales", "cost"],
        var_name="metric",
        value_name="amount",
    )
    # geom_col with colour metric — categories share x; plot3 may overplot.
    # Use facet for a clean grouped comparison.
    return (
        ggplot(df, aes(x="category", y="amount", colour="metric"))
        + geom_col(width=0.7)
        + facet_wrap("metric", ncol=2, scales="free")
        + theme_light()
        + labs(title="Sales vs cost (faceted cols)", x="category", y="amount")
    )


def fig_dark_scatter():
    """Dark theme showcase (theme_dark default aesthetic)."""
    df = data_gapminder()
    return (
        ggplot(df, aes(x="log_gdp", y="lifeExp", colour="continent"))
        + geom_point(size=6, alpha=0.9)
        + theme_dark()
        + labs(
            title="Dark theme scatter",
            x="log10 GDP per cap",
            y="life expectancy",
            colour="continent",
        )
    )


def fig_line_dark():
    """Dark multi-line for dashboards / slides."""
    df = data_timeseries()
    return (
        ggplot(df, aes(x="t", y="value", colour="region"))
        + geom_line(linewidth=2.4)
        + theme_dark()
        + labs(
            title="Regional index (dark)",
            x="month index",
            y="index",
            colour="region",
        )
    )


SHOWCASES: dict[str, callable] = {
    # points
    "scatter": fig_scatter,
    "scatter_colour": fig_scatter_colour,
    "scatter_continuous": fig_scatter_continuous,
    "scatter_groups": fig_scatter_groups,
    "dark_scatter": fig_dark_scatter,
    # lines / paths
    "line": fig_line,
    "line_groups": fig_line_groups,
    "line_dark": fig_line_dark,
    "path": fig_path,
    # bars
    "col": fig_col,
    "bar": fig_bar,
    "col_grouped": fig_col_grouped,
    # distributions
    "histogram": fig_histogram,
    "histogram_hp": fig_histogram_fill,
    "density": fig_density,
    "density_groups": fig_density_groups,
    "boxplot": fig_boxplot,
    "violin": fig_violin,
    # small multiples
    "facet": fig_facet,
    "facet_density": fig_facet_density,
}


def build_one(name: str, *, open_browser: bool) -> Path:
    fig = SHOWCASES[name]()
    path = OUT / f"{name}.html"
    fig.save(str(path))
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


def write_index(built: list[tuple[str, Path]]) -> Path:
    """Simple HTML index linking every figure (gallery landing page)."""
    cards = []
    for name, path in built:
        rel = path.name
        cards.append(
            f'<a class="card" href="{rel}" target="_blank">'
            f"<strong>{name}</strong>"
            f"<span>{rel}</span></a>"
        )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>plot3 · 2D gallery</title>
<style>
  body {{ margin:0; font:15px/1.45 system-ui,sans-serif; background:#0b1020; color:#e2e8f0; }}
  header {{ padding:28px 32px 8px; }}
  header h1 {{ margin:0 0 6px; font-size:22px; }}
  header p {{ margin:0; color:#94a3b8; max-width:52rem; }}
  header a {{ color:#93c5fd; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
           gap:12px; padding:20px 32px 40px; }}
  .card {{ display:flex; flex-direction:column; gap:4px; padding:14px 16px;
           border-radius:10px; background:#1e293b; color:#e2e8f0; text-decoration:none;
           border:1px solid #334155; transition:border-color .15s, transform .15s; }}
  .card:hover {{ border-color:#60a5fa; transform:translateY(-1px); }}
  .card span {{ color:#94a3b8; font-size:12px; }}
</style></head><body>
<header>
  <h1>plot3 · 2D gallery</h1>
  <p>
    ggplot2-style examples inspired by the
    <a href="https://r-graph-gallery.com/ggplot2-package.html">R Graph Gallery</a>.
    Each card opens a self-contained interactive HTML figure (pan / zoom / hover).
  </p>
</header>
<div class="grid">
{"".join(cards)}
</div>
</body></html>
"""
    index = OUT / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="plot3 2D showcase (browser HTML)")
    p.add_argument(
        "names",
        nargs="*",
        help=f"subset of: {', '.join(SHOWCASES)} (default: all)",
    )
    p.add_argument("--list", action="store_true", help="list showcase names")
    p.add_argument(
        "--no-open",
        action="store_true",
        help="write HTML only (do not open browser tabs)",
    )
    p.add_argument(
        "--open-index",
        action="store_true",
        help="open the gallery index page only (after build)",
    )
    args = p.parse_args(argv)

    if args.list:
        for k in SHOWCASES:
            print(k)
        return

    names = args.names or list(SHOWCASES)
    unknown = [n for n in names if n not in SHOWCASES]
    if unknown:
        raise SystemExit(
            f"unknown showcase(s): {unknown}; choose from {list(SHOWCASES)}"
        )

    # Open individual tabs only for small subsets; full gallery → index page.
    open_each = (not args.no_open) and (not args.open_index) and len(names) <= 4
    open_index = (not args.no_open) and (args.open_index or len(names) > 4)

    print(f"plot3 2D showcase → {OUT}")
    built: list[tuple[str, Path]] = []
    for name in names:
        path = build_one(name, open_browser=open_each)
        built.append((name, path))
        print(f"  {name:20s}  {path.name}  ({path.stat().st_size // 1024} KB)")

    # Prefer a complete index when all known figures exist on disk.
    index_entries = [
        (n, OUT / f"{n}.html")
        for n in SHOWCASES
        if (OUT / f"{n}.html").exists()
    ]
    if not index_entries:
        index_entries = built
    index = write_index(index_entries)
    print(f"  index               {index.name}")

    if open_index:
        webbrowser.open(index.resolve().as_uri())

    print("done — pan/zoom in the browser; click legend entries to hide series.")


if __name__ == "__main__":
    main()
