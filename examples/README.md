# plot3 examples

Browser-first galleries (VS Code notebooks cannot run the WebGL viewer inline).

Inspired by the [R Graph Gallery · ggplot2](https://r-graph-gallery.com/ggplot2-package.html).

```bash
cd /path/to/plot3
source .venv/bin/activate

python examples/showcase_2d.py              # 2D gallery + index.html
python examples/showcase_3d.py              # 3D gallery + index.html

python examples/showcase_2d.py boxplot violin facet
python examples/showcase_3d.py galaxy peaks
python examples/showcase_2d.py --list
python examples/showcase_2d.py --no-open    # write files only
```

| Gallery | Output | Notes |
|---------|--------|--------|
| 2D | `examples/output/2d/` | Opens **index.html** for full runs |
| 3D | `examples/output/3d/` | Opens **index.html** for full runs |

Subsets of ≤4 names open those figures directly in the browser.

---

## 2D gallery (ggplot2-style)

| File | ggplot2 analogue | Geom / pattern |
|------|------------------|----------------|
| `scatter.html` | basic scatter | `geom_point` |
| `scatter_colour.html` | map colour to factor | `geom_point` + `colour=` |
| `scatter_continuous.html` | continuous colour | `scale_colour_viridis_c` |
| `scatter_groups.html` | gapminder-style | continent colour |
| `dark_scatter.html` | dark theme | `theme_dark` |
| `line.html` | line chart | `geom_line` + points |
| `line_groups.html` | multi-series lines | `colour=group` |
| `line_dark.html` | dark multi-line | dashboard style |
| `path.html` | connected scatter | `geom_path` |
| `col.html` | barplot from values | `geom_col` |
| `bar.html` | count bars | `geom_bar` |
| `col_grouped.html` | grouped comparison | `geom_col` + `facet_wrap` |
| `histogram.html` | histogram | `geom_histogram` |
| `histogram_hp.html` | histogram | continuous `x` |
| `density.html` | density | `geom_density` |
| `density_groups.html` | density by group | `colour=` |
| `boxplot.html` | boxplot | `geom_boxplot` |
| `violin.html` | violin | `geom_violin` |
| `facet.html` | small multiples | `facet_wrap` |
| `facet_density.html` | faceted densities | `facet_wrap` |

**Controls:** drag pan · Ctrl/⌘+scroll zoom · double-click reset · legend click hide/show

### Snippet

```python
from plot3 import aes, geom_point, ggplot, labs, theme_light
import examples.showcase_2d as g

fig = (
    ggplot(g.data_mtcars(), aes(x="wt", y="mpg", colour="cyl"))
    + geom_point(size=6)
    + theme_light()
    + labs(title="Weight vs MPG", colour="cyl")
)
fig.show(browser=True)
```

---

## 3D gallery

| File | Technique |
|------|-----------|
| `lidar.html` | synthetic indoor scan, turbo intensity |
| `galaxy.html` | spiral arms + bulge |
| `peaks.html` | multi-peak `geom_surface` |
| `sombrero.html` | wireframe surface |
| `helix.html` | double helix `geom_path` |
| `isosurface.html` | density shells |
| `terrain.html` | elevation mesh |
| `survey.html` | sparse survey points |
| `rgb.html` | RGB lattice |

**Controls:** drag orbit · scroll zoom · double-click reset · hover values

```python
from plot3 import aes, coord_3d, geom_point3d, ggplot, labs, scale_colour_viridis_c
import examples.showcase_3d as s

fig = (
    ggplot(s.data_galaxy(), aes(x="x", y="y", z="z", colour="brightness"))
    + geom_point3d(size=0.018, alpha=0.85)
    + scale_colour_viridis_c(option="magma")
    + coord_3d(aspect="equal")
    + labs(title="Galaxy")
)
fig.show(browser=True)
```

---

## Not yet in plot3 (gallery gaps)

Useful R Graph Gallery patterns package does **not** cover yet (candidates for later geoms):

- `geom_smooth` / loess ribbons  
- `geom_tile` / heatmaps  
- `geom_text` / `geom_label`  
- `geom_hline` / `geom_vline` / segments  
- `geom_ribbon` / area  
- `geom_jitter`, hex bins, rug  

The 2D showcase sticks to geoms that work today so every card is a real interactive figure.
