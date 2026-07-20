# plot3

Grammar-of-graphics plotting on **three.js** — 2D and 3D figures from pandas
DataFrames that stay small, stay smooth, and survive
[sslive](https://github.com/rleyvasal/sslive) slide export.

Works **locally** (VS Code, terminal, JupyterLab) or under CRAFT / SolveIt.

## Install (local / VS Code)

```bash
cd /path/to/plot3
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,jupyter]"
```

In VS Code: **Python: Select Interpreter** → this `.venv`. No CRAFT required.

```python
from plot3 import ggplot, aes, geom_point, geom_line, geom_col, geom_histogram, labs, theme_light

(ggplot(df, aes(x="time", y="temp", colour="sensor"))
 + geom_point(size=4)
 + geom_line(linewidth=2)
 + labs(title="Sensor temps", y="degC"))
```

The mapping-first form is pipeable from tidy3, pandas, or Polars. Layers bind
before ``>>``, so the expression needs no extra parentheses:

```python
plot = tidy(df) >> ggplot(aes(x="time", y="temp")) + geom_point()
```

Follows [ggplot2](https://github.com/tidyverse/ggplot2) conventions: `+`
layering, `colour`/`color` both accepted, `geom_line(linewidth=)`, `geom_path`
preserves data order while `geom_line` sorts by x, `ggsave("fig.html", p)`.

## Why

- **Portable**: each figure is one self-contained iframe — data uint16-quantized,
  delta + byte-shuffle + gzip compressed (browser `DecompressionStream`),
  three.js from CDN. Typical figures are 30–150 KB vs multi-MB Plotly HTML.
- **Smooth**: WebGL marks; 100k+ point scatters at 60 fps, first frame instant.
- **2D interactivity**: drag to pan, Ctrl/⌘+scroll (or trackpad pinch) to zoom,
  double-click to reset, hover for values. `aes(z=...)` switches to a 3D orbit
  viewer with axes box. **Click a legend entry to hide/show that category.**
- **Validated colors**: fixed-order categorical palette and single-hue
  sequential ramp, CVD-validated for both the dark (`#0b1020`) and light
  themes. More than 8 categories is an error, not an eleventh hue.
  Numeric colour uses **robust 2–98 percentile limits by default** (skewed
  data like lidar intensity stays readable); control with
  `scale_colour_continuous(trans="sqrt"|"log10", limits=(lo,hi)|"full")`.
  For lidar/heatmap-style data add `scale_colour_viridis_c()` —
  `option="viridis"|"magma"|"turbo"` (turbo ≈ the nuScenes look), exact
  matplotlib stops, perceptually ordered and CVD-safe unlike jet/hsv.
- **Context-safe in SolveIt**: displaying a figure red-eyes its cell
  (`skipped=1`) so viewer HTML never eats LLM context. Opt out per figure
  with `ggplot(..., hide=False)` or globally with `autohide(False)`.

## API (v1)

| Piece | Notes |
|---|---|
| `ggplot(df, aes(...))` / `data >> ggplot(aes(...))` | figure; `quantize=False` for deep-zoom float32 payloads |
| `aes(x, y, z, colour, group)` | `z` → 3D; colour numeric → ramp, categorical → palette + legend |
| `geom_point(size=, alpha=)` | 2D size in px, 3D in scene units |
| `geom_line(linewidth=)` / `geom_path()` | line sorts by x; path keeps data order |
| `geom_col(width=)` | bars from `y` heights (ggplot2 `geom_col`) |
| `geom_bar(width=)` | count bars for discrete `x` |
| `geom_histogram(bins=, width=)` | continuous `x` histogram → bars |
| `geom_boxplot(width=, coef=, outlier_size=)` | Tukey box-and-whisker of `y` by `x` |
| `geom_density(n=, adjust=, fill=)` | KDE of continuous `x` (optional `colour` groups) |
| `geom_violin(n=, adjust=, width=)` | Mirrored KDE of `y` by `x` |
| `facet_wrap("col", ncol=, nrow=)` | Panel grid by a discrete column |
| `labs(title=, x=, y=, z=, colour=)` | labels |
| `scale_colour_continuous(trans=, limits=, palette=)` | numeric colour: `sqrt`/`log10`/`linear`; limits tuple or `"full"` |
| `scale_colour_viridis_c(option=)` | viridis / magma / turbo colormaps for numeric colour |
| `theme_dark()` / `theme_light()` | dark is default |
| `ggsave(filename, p)` / `p.save(path)` | standalone HTML file |
| `read_bin(path, stride=5)` | point-cloud `.bin` → DataFrame; `remote=True` streams via CRAFT |

## Package layout

```text
plot3/
  __init__.py      # public API
  geoms.py         # aes, geoms, labs, colour scales, themes
  ggplot.py        # ggplot, ggsave, autohide
  build.py         # stats expand + layer/spec encoding
  viewer.py        # three.js HTML template
  encode.py        # quantize / gzip payloads
  scales.py        # positional scales & ticks
  themes.py        # palettes / theme tokens
  io.py            # read_bin (+ optional CRAFT SSH)
  jupyter.py       # %plot3 magic, SolveIt hide-from-AI
load.py            # CRAFT %run entry (optional)
```

Public imports stay the same: `from plot3 import ggplot, aes, geom_point, ...`.

## SolveIt / CRAFT

Prefer install + extension load:

```text
%local
# once per environment: pip install -e /path/to/plot3
%load_ext plot3
%plot3 df x=time y=temp color=sensor kind=line
```

Or without pip, run the package entrypoint (replaces the old monolith
`%run .../plot3.py`):

```text
%local
%run /path/to/plot3/load.py
%plot3 df x=time y=temp color=sensor kind=line
```

With CRAFT loaded, `%plot3` evaluates the DataFrame expression **on the GPU
kernel** and ships only the mapped columns over SSH; without CRAFT it evaluates
locally (plain Jupyter works). The cell is red-eyed out of LLM context
(`hide=0` to keep it). Point clouds:

```python
(ggplot(read_bin("/path/scan.pcd.bin", remote=True),
        aes(x="x", y="y", z="z", color="intensity"))
 + geom_point(size=0.006))
```

```python
# Density / violin / facets
ggplot(df, aes(x="mpg", colour="cyl")) + geom_density()
ggplot(df, aes(x="cyl", y="mpg")) + geom_violin()
ggplot(df, aes(x="wt", y="mpg")) + geom_point() + facet_wrap("cyl", ncol=2)
```

## Roadmap

Shared fixed facet scales, 3D hover picking, more `scale_*` overrides, diverging
scales, express-style wrappers.
