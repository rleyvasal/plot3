# plot3

ggplot2's grammar of graphics for Python, drawn with WebGL (three.js) in the
notebook and saved as journal-ready PNG, SVG, or PDF.

```python notest
ggplot(df, aes(x="dose", y="response", colour="arm")) + geom_point() + geom_smooth(method="lm")
```

- **ggplot2 grammar**: `+` layers, `aes()`, geoms, stats, scales, facets,
  themes, and `ggsave()`, with ggplot2's defaults and names.
- **Interactive 2D and 3D**: pan, zoom, hover, click a legend entry to hide
  it, orbit 3D point clouds and surfaces, play animations, drag sliders.
- **Publication output**: `ggsave("fig.pdf", p, width=3.5, height=2.6,
  units="in")` with real fonts, 300 dpi metadata, and `theme_bw` by default.
- **Maths built in**: plot `"y = x^2 + 1"`, implicit equations, surfaces,
  densities (`dbeta`, `dnorm`), probability areas, tangents, and LaTeX.
- **Your data as it is**: pandas, Polars, tidy3, or NumPy arrays, locally
  or on a remote GPU kernel (CRAFT / SolveIt).

Contents: [Install](#install) · [Quick start](#quick-start) ·
[Statistical plots](#statistical-plots) · [Annotation](#annotation) ·
[Scales](#scales) · [Themes and titles](#themes-and-titles) ·
[Facets and multi-panel figures](#facets-and-multi-panel-figures) ·
[Saving for a paper](#saving-for-a-paper) · [Functions and maths](#functions-and-maths) ·
[Animation and sliders](#animation-and-sliders) · [3D](#3d-and-point-clouds) ·
[Notebooks, SolveIt, CRAFT](#notebooks-solveit-and-craft) ·
[API reference](#api-reference) · [Differences from ggplot2](#differences-from-ggplot2)

## Install

```bash
git clone https://github.com/rleyvasal/plot3 && cd plot3
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[jupyter,export]"
```

| Extra | Adds |
|---|---|
| `export` | `cairosvg`: PNG with real fonts and PDF. Needs the Cairo C library (`brew install cairo`, `apt install libcairo2`). SVG needs nothing. |
| `fast` | `contourpy`: faster implicit-curve contours (matplotlib users have it) |
| `jupyter` | IPython integration (bare column names, `%plot3`) |
| `polars` | Polars tables |

Python 3.10 or newer; NumPy and pandas are the only required packages.

## Quick start

The examples below share this data:

```python
import numpy as np
import pandas as pd
from plot3 import *

rng = np.random.default_rng(1)
trial = pd.DataFrame({
    "arm": rng.choice(["placebo", "low", "high"], 150),
    "sex": rng.choice(["F", "M"], 150),
    "dose": rng.uniform(0, 10, 150),
})
trial["response"] = (2 + 0.6 * trial.dose + 1.5 * (trial.arm == "high")
                     + rng.normal(0, 1.2, 150))
sales = pd.DataFrame({"month": pd.date_range("2021-01-01", periods=36, freq="MS")})
sales["units"] = 100 + np.cumsum(rng.normal(1, 4, 36))
sales["lo"], sales["hi"] = sales.units - 8, sales.units + 8
```

A plot is data, an aesthetic mapping, and layers:

```python
p = (ggplot(trial, aes(x="dose", y="response", colour="arm"))
     + geom_point(alpha=0.7)
     + geom_smooth(method="lm")
     + labs(title="Response by dose", x="Dose (mg)", y="Response"))
```

In a notebook, `p` on its own line displays the interactive figure. Save it:

```python notest
ggsave("response.pdf", p, width=3.5, height=2.6, units="in")   # vector, journal column
ggsave("response.png", p, width=7, height=4, units="in", dpi=300)
ggsave("response.html", p)                                      # interactive page
```

In notebooks, bare column names work too: `aes(x=dose, y=response)`.
Plain `.py` files use strings, as above.

## Statistical plots

```python
# Distributions
ggplot(trial, aes(x="response", fill="sex")) + geom_histogram(bins=20)            # stacked by group
ggplot(trial, aes(x="response", fill="arm")) + geom_density(alpha=0.4)
ggplot(trial, aes(x="arm", y="response")) + geom_boxplot() + geom_jitter(width=0.15, height=0)
ggplot(trial, aes(x="arm", y="response", fill="arm")) + geom_violin()
ggplot(trial, aes(x="response", colour="arm")) + stat_ecdf()
ggplot(trial, aes(sample="response")) + geom_qq() + geom_qq_line()

# Counts and proportions
ggplot(trial, aes(x="arm", fill="sex")) + geom_bar()                     # stacked
ggplot(trial, aes(x="arm", fill="sex")) + geom_bar(position="dodge")     # side by side
ggplot(trial, aes(x="arm", fill="sex")) + geom_bar(position="fill")      # shares

# Means with uncertainty
(ggplot(trial, aes(x="arm", y="response", fill="sex"))
 + stat_summary(fun_data="mean_se", geom="col", position="dodge")
 + stat_summary(fun_data="mean_cl_normal", geom="errorbar", position="dodge", width=0.3))

# Trends and time series
ggplot(trial, aes(x="dose", y="response")) + geom_point() + geom_smooth()        # loess + 95% band
(ggplot(sales, aes(x="month", y="units"))
 + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.25) + geom_line())
```

| Need | Use |
|---|---|
| Error bars from your own columns | `geom_errorbar(aes(ymin=, ymax=))`, `geom_pointrange`, `geom_linerange` |
| Summaries | `stat_summary(fun_data="mean_se" / "mean_cl_normal" / "mean_sdl" / "median_hilow")` |
| Heatmaps | `geom_tile(aes(x=, y=, fill=))` (or `geom_raster`), with `geom_text(aes(label=))` |
| Stacked areas | `geom_area(aes(fill=))`, `position="fill"` for shares |
| Steps, segments, rectangles | `geom_step()`, `geom_segment(aes(xend=, yend=), arrow=arrow())`, `geom_rect(aes(xmin=, xmax=, ymin=, ymax=))` |
| Horizontal layout | `+ coord_flip()` (bars, boxplots, densities, error bars) |
| Several datasets | `geom_rect(aes(...), data=periods)`: any layer can bring its own data |

Rows with missing values are dropped, and plot3 says so: *Removed 3 rows
containing missing values (geom_point)*.

## Annotation

```python
(ggplot(trial, aes(x="dose", y="response"))
 + geom_point()
 + geom_hline(yintercept=5, linetype="dashed")
 + geom_vline(xintercept=[2, 8], colour="firebrick")
 + geom_abline(slope=0.6, intercept=2)
 + annotate("rect", xmin=2, xmax=8, ymin=0, ymax=12)
 + annotate("label", x=9, y=1, label="safe range")
 + annotate("segment", x=1, y=10, xend=3, yend=8, arrow=arrow()))

means = trial.groupby("arm", as_index=False).response.mean()
(ggplot(means, aes(x="arm", y="response", label="response"))
 + geom_col() + geom_text(nudge_y=0.4))     # numbers print to 4 significant figures
```

`geom_text` / `geom_label` take `size` in millimetres (ggplot2's 3.88 mm
default), `hjust`, `vjust`, `nudge_x`, `nudge_y`, `fontface`, and
`check_overlap`. Line types are `"solid"`, `"dashed"`, `"dotted"`,
`"dotdash"`, `"longdash"`, `"twodash"`, R's numbers, or hex such as `"44"`.

Map more aesthetics: `aes(shape=)` (circle, triangle, square, diamond, plus,
cross), `aes(linetype=)`, `aes(size=)` (area), `aes(fill=)` for filled shapes
and `aes(colour=)` for points and lines.

## Scales

```python
(ggplot(trial, aes(x="dose", y="response", colour="arm"))
 + geom_point()
 + scale_x_continuous("Dose (mg)", breaks=[0, 2.5, 5, 7.5, 10])
 + scale_y_continuous(limits=(0, None))
 + scale_colour_manual(values={"placebo": "grey50", "low": "#56B4E9", "high": "#D55E00"},
                       breaks=["placebo", "low", "high"],
                       labels=["Placebo", "Low dose", "High dose"], name="Arm"))

(ggplot(sales, aes(x="month", y="units"))
 + geom_line()
 + scale_x_date(date_breaks="6 months", date_labels="%b %Y")
 + scale_y_continuous(labels="dollar"))
```

| Scale | Functions |
|---|---|
| Position | `scale_x_continuous(name, limits, breaks, labels, trans="log10"/"reverse")`, `scale_x_discrete(limits, labels)`, `scale_x_date(date_breaks, date_labels)`, `scale_x_log10`, `scale_x_reverse`, `xlim`, `ylim`, `lims` (and the `y` versions) |
| Label formats | `"percent"`, `"comma"`, `"dollar"`, `"scientific"`, `"{:.1f} kg"`, a list, or a function |
| Discrete colour / fill | `scale_colour_manual`, `scale_colour_brewer(palette="Set2")`, `scale_colour_viridis_d`, `scale_colour_grey`, `scale_colour_okabe_ito` (colour-blind safe) |
| Continuous colour / fill | `scale_colour_gradient(low, high)`, `scale_colour_gradient2(low, mid, high, midpoint)`, `scale_colour_viridis_c`, `scale_colour_continuous(trans="log10")` |
| Shape / linetype | `scale_shape_manual`, `scale_linetype_manual` |

Every colour scale has `scale_fill_*` and `scale_color_*` names. Colours can
be CSS names (`"steelblue"`), R greys (`"grey50"`), `"rgb(1,2,3)"`, or hex.
Limits on a continuous axis drop the rows outside them and say how many.

## Themes and titles

```python
(ggplot(trial, aes(x="arm", y="response", fill="arm"))
 + geom_boxplot()
 + labs(title="Response", subtitle="150 patients", caption="Source: simulated", tag="A")
 + theme_classic()
 + theme(legend_position="none", axis_text_x_angle=45, plot_title_hjust=0.5))
```

Themes: `theme_bw` (default for saved files), `theme_classic`,
`theme_minimal`, `theme_light`, `theme_dark` (default in the interactive
viewer). Each takes `base_size` (points) and `base_family`. `theme()` sets
`legend_position` (`"right"`, `"bottom"`, `"none"`, or `(x, y)` inside the
panel), `legend_title=False`, `panel_grid=False`, `axis_text_x_angle`,
`plot_title_hjust`, `base_size`, and `base_family`.

## Facets and multi-panel figures

```python
ggplot(trial, aes(x="dose", y="response")) + geom_point() + facet_wrap("arm")
(ggplot(trial, aes(x="dose", y="response", colour="arm"))
 + geom_point() + facet_grid("sex ~ arm"))                # rows ~ columns

a = ggplot(trial, aes(x="dose", y="response", colour="arm")) + geom_point()
b = ggplot(trial, aes(x="arm", y="response", fill="arm")) + geom_boxplot() + theme(legend_position="none")
c = ggplot(sales, aes(x="month", y="units")) + geom_line()
fig = ((a | b) / c) + plot_annotation(title="Overview", tag_levels="A")
fig2 = (a | b) + plot_layout(widths=[2, 1])
```

Facet panels share scales, colours, one legend, and one pair of axis titles,
as in ggplot2. `|` puts plots side by side, `/` stacks them, and
`tag_levels` is `"A"`, `"a"`, `"1"`, or `"I"`. Save a composed figure with
`ggsave` like any plot.

## Saving for a paper

```python notest
ggsave("fig1.pdf", p, width=3.5, height=2.6, units="in")             # one column
ggsave("fig1.png", p, width=7, height=4.5, units="in", dpi=300)       # two columns
ggsave("fig1.svg", p, width=18, height=12, units="cm", fontsize=9, family="Arial")
```

- PNG, SVG, and PDF draw the same picture with real fonts. PNGs carry their
  DPI, so Word and journal portals size them correctly.
- Saved files use `theme_bw` unless you add a theme.
- Crowded category labels turn or thin automatically; set
  `theme(axis_text_x_angle=)` to choose.
- Non-Latin text (東京, 서울, ✓) uses an installed font that has the glyphs.
- Notes such as *Removed 3 rows…* are printed, not drawn; `notes=True` draws them.
- Without the `export` extra, `.svg` works everywhere and `.png` falls back to
  a built-in bitmap font.

The interactive viewer also has a **Save** button (HTML, SVG, PNG, video for
animations, copy to clipboard).

## Functions and maths

`geom_function` plots a formula with the same grammar as data. `^` is power
and `2x` means `2*x`.

```python
ggplot() + geom_function("y = 2x + 2")
ggplot() + geom_function("y = a x^2 + b x + c", a=2, b=-3, c=1)    # coefficients at the end
ggplot() + geom_function("x^2 + y^2 = 1")                          # implicit: a round circle
ggplot() + geom_function("z = sin(x) cos(y)", xlim=(-3, 3), ylim=(-3, 3))   # 3D surface, coloured by height
ggplot() + geom_function(r"y = \frac{\sin x}{x}")                  # LaTeX input
ggplot() + geom_function("r = 1 + cos(theta)") + coord_polar()
ggplot() + geom_function("y > x^2")                                # shaded region
ggplot(trial, aes(x="dose", y="response")) + geom_point() + geom_function("y = 2 + 0.6x")
```

Distributions and probability areas (no SciPy needed):

```python
inf = float("inf")
ggplot() + geom_function("y = dbeta(x, 2, 5)") + area(0.2, 0.5)            # P(0.2 ≤ X ≤ 0.5) = 0.546
ggplot() + geom_function("y = dnorm(x)") + area(-inf, -1.96) + area(1.96, inf)
ggplot() + geom_function("y = dt(x, 3)") + area(2.353, inf)               # P(X ≥ 2.353) = 0.05
ggplot() + geom_function("y = x^3 - 3x") + tangent(at=1) + derivative()
```

Formulas know `sin cos tan exp log ln sqrt abs floor ceil gamma lgamma beta
erf erfc`, the densities `dnorm dbeta dt dchisq dgamma dexp dunif dlnorm`, and
`pnorm qnorm pt qt pbeta pexp punif`. A density opens on its own support
(`dbeta` on [0, 1]). Poles (`1/x`, `tan x`) are clipped with a note; steep but
finite curves are not. Pass your own functions as keywords:
`geom_function("y = damp(x) sin(3x)", damp=my_damp)`, or a lambda.

## Animation and sliders

```python
ggplot() + geom_function("y = sin(x - t)") + transition_time(t=(0, 6.28))   # travelling wave
ggplot() + geom_function("y = dbeta(x, a, b)") + slider(a=(0.5, 5), b=(0.5, 5))
```

Data animations follow `gganimate`:

```python notest
(ggplot(gapminder, aes(x="gdp", y="life", size="pop", colour="continent", group="country"))
 + geom_point() + scale_x_log10() + transition_time("year") + labs(title="{frame_time}"))
```

The viewer interpolates between frames in the browser, with play, pause,
scrubbing, speed, and video recording.

## 3D and point clouds

Map `z` on every layer for an orbit view. A point cloud with no colour of
its own is coloured by height (viridis), as lidar viewers draw it; map
`colour=` or set `colour="steelblue"` to change that.

```python notest
ggplot(lidar, aes(x="x", y="y", z="z")) + geom_point3d()      # coloured by height

(ggplot(cloud, aes(x="x", y="y", z="z", colour="intensity"))
 + geom_point3d(size=0.008)
 + coord_3d(aspect="data", max_points=300_000)
 + scale_colour_viridis_c(option="turbo"))

ggplot(grid, aes(x="x", y="y", z="height", fill="height")) + geom_surface()
ggplot(points, aes(x="x", y="y", z="z")) + geom_isosurface(levels=[0.2, 0.5, 0.8])
read_bin("scan.pcd.bin")          # nuScenes-style point clouds; remote=True under CRAFT
```

NumPy arrays use column positions: `ggplot(pts, aes(x=0, y=1, z=2, colour=3))`.

The box keeps the data's proportions, except that a tall cloud (a helix, a
tree) is shortened to twice its width so it does not become a thin column;
`coord_3d(aspect="data")` keeps true proportions always, and
`aspect="equal"` draws a cube.

## Notebooks, SolveIt, and CRAFT

- **Jupyter / SolveIt**: bare column names and backticks work in `aes()`,
  `facet_wrap()`, and `facet_grid()` (`aes(x=`First Name`)`). Toggle with
  `enable_r_style()` / `disable_r_style()`. `.py` files keep quoted strings.
- **SolveIt** draws figures inline and hides their HTML from the model's
  context (`autohide(False)` to opt out).
- **VS Code notebooks** block WebGL in output cells, so plot3 opens figures
  in your browser (`PLOT3_DISPLAY=browser|iframe` to force a mode).
- **CRAFT / `%gpu`**: the same `ggplot(...)` code runs on the remote kernel;
  stats run where the data lives and only a compact payload comes back.

```text
%run /path/to/plot3/plot3.py        # loads plot3 locally and seeds the GPU kernel
%plot3 df x=wt y=mpg color=cyl       # optional shortcut magic
```

## API reference

| Area | Functions |
|---|---|
| Figure | `ggplot(data, aes(...))`, `data >> ggplot(aes(...))`, `+`, `p.show()`, `p.save()`, `ggsave()` |
| Aesthetics | `aes(x, y, z, colour, fill, size, shape, linetype, group, label, ymin, ymax, xmin, xmax, xend, yend, sample)` |
| Points and lines | `geom_point`, `geom_jitter`, `geom_line`, `geom_path`, `geom_step`, `geom_segment`, `geom_text`, `geom_label` |
| Bars and areas | `geom_col`, `geom_bar`, `geom_histogram`, `geom_area`, `geom_ribbon`, `geom_rect`, `geom_tile`/`geom_raster` |
| Distributions | `geom_boxplot`, `geom_violin`, `geom_density`, `geom_qq`, `geom_qq_line`, `stat_ecdf`, `stat_summary` |
| Uncertainty and fits | `geom_errorbar`, `geom_pointrange`, `geom_linerange`, `geom_smooth(method="loess"/"lm")` |
| Reference | `geom_hline`, `geom_vline`, `geom_abline`, `annotate("text"/"label"/"rect"/"segment"/"point")` |
| Positions | `position="stack"/"dodge"/"fill"/"identity"`, `position_dodge(width)`, `position_stack()`, `position_fill()` |
| Functions | `geom_function`, `geom_vector_field`, `area`, `tangent`, `derivative` |
| Scales | see [Scales](#scales) |
| Coordinates | `coord_flip`, `coord_equal`, `coord_polar`, `coord_3d` |
| Facets and layout | `facet_wrap`, `facet_grid`, `p1 | p2`, `p1 / p2`, `plot_layout`, `plot_annotation` |
| Labels and themes | `labs(title, subtitle, caption, tag, x, y, colour, fill)`, `theme_*`, `theme()` |
| Animation | `transition_time`, `transition_states`, `slider` |
| 3D | `geom_point3d`, `geom_surface`, `geom_isosurface`, `stat_density_3d`, `read_bin` |

## Differences from ggplot2

- Python needs quotes outside notebooks: `aes(x="wt")`. In Jupyter and
  SolveIt, `aes(x=wt)` works.
- `labs(x="")` removes a title (ggplot2's `labs(x = NULL)`).
- `fill` colours filled shapes; points and lines use `colour`, as in ggplot2.
- Categories sort alphabetically (numbers numerically); use
  `pd.Categorical` or `scale_x_discrete(limits=)` for your own order.
- Text `size` is in millimetres, as in ggplot2; `geom_point(size=)` is in
  pixels in 2D.
- Saved files default to `theme_bw`, the interactive viewer to `theme_dark`.

## Development

```bash
pytest -q                       # ~600 tests, including every example in this README
python examples/showcase_2d.py  # 2D gallery in the browser
python examples/showcase_3d.py  # 3D gallery
```

The code is in `plot3/`: `geoms.py` (grammar objects), `scaling.py`
(scale functions), `build.py` (stats to a figure spec), `stat2d.py` and
`flip.py` (statistical layers), `static.py` (PNG/SVG/PDF), `viewer.py` (the
WebGL viewer), `expr.py` / `function.py` / `calculus.py` (formulas),
`compose.py` (multi-panel figures). See [CHANGELOG.md](CHANGELOG.md) for
changes between versions.

## License

MIT. See [LICENSE](LICENSE).
