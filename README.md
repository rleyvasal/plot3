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
| `aes(x, y, z, colour, group)` | `z` on every layer → 3D orbit view |
| `geom_point(size=, alpha=)` | 2D: size in **px**; 3D: size in **scene units** |
| `geom_point3d(size=, alpha=)` | Explicit 3D points (same engine as `geom_point`) |
| `geom_surface(wireframe=, alpha=)` | 3D mesh from a complete regular `x`×`y` grid (`z` height) |
| `coord_3d(aspect=, size_mode=, max_points=)` | 3D aspect (`data`/`equal`), point size mode, optional subsample |
| `geom_line(linewidth=)` / `geom_path()` | line sorts by x; path keeps data order; work in 3D too |
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

## 3D and point clouds

3D mode is the same grammar: map **`z`**, add a 3D-capable geom. There is no
separate `ggplot3d` class.

```python
from plot3 import (
    ggplot, aes, geom_point, geom_point3d, coord_3d,
    scale_colour_viridis_c, read_bin, labs,
)

# Core form (always supported)
ggplot(df, aes(x="x", y="y", z="z", colour="intensity")) + geom_point(size=0.008)

# Explicit 3D + coordinate options (recommended for lidar)
(
    ggplot(df, aes(x="x", y="y", z="z", colour="intensity"))
    + geom_point3d(size=0.008)
    + coord_3d(aspect="data", size_mode="scene", max_points=300_000)
    + scale_colour_viridis_c(option="turbo")
    + labs(title="LIDAR", colour="intensity")
)
```

### nuScenes / CRAFT lidar

```python
cloud = read_bin(
    "/home/gpudev/nuscenes_data/samples/LIDAR_TOP/"
    "n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin",
    remote=True,   # stream + thin on the GPU host under CRAFT
)
(
    ggplot(cloud, aes(x="x", y="y", z="z", colour="intensity"))
    + geom_point3d(size=0.001)
    + scale_colour_viridis_c(option="turbo")
    + coord_3d()
)
```

`read_bin` defaults to columns `x, y, z, intensity` (stride 5 float32). Use
`remote=True` only after CRAFT `%gpu` so `SSH_HOST` / `remote_run_` exist.

### Surfaces from a grid

Long-form rectangular grid (every `x`×`y` cell present once):

```python
import numpy as np
import pandas as pd
from plot3 import ggplot, aes, geom_surface, coord_3d, scale_colour_viridis_c

xs, ys = np.linspace(-2, 2, 40), np.linspace(-2, 2, 40)
xx, yy = np.meshgrid(xs, ys)
grid = pd.DataFrame({
    "x": xx.ravel(),
    "y": yy.ravel(),
    "height": np.sin(xx).ravel() * np.cos(yy).ravel(),
})
(
    ggplot(grid, aes(x="x", y="y", z="height", fill="height"))
    + geom_surface(alpha=0.95)
    + scale_colour_viridis_c()
    + coord_3d(aspect="data")
)
# wireframe only:
# + geom_surface(wireframe=True)
```

| | 2D | 3D |
|--|----|----|
| Switch | no `z` | `aes(z=...)` on **all** layers |
| Point `size` | pixels | scene units (`size_mode="scene"`) |
| Interaction | pan / zoom | orbit |
| Not allowed | — | `facet_wrap`, bar/box/density/violin/… |

## Roadmap

Density isosurfaces (`geom_isosurface`), shared fixed facet scales, 3D hover
picking, more `scale_*` overrides, diverging
scales, express-style wrappers.
