# plot3

**ggplot2 for Python, plus what ggplot2 can't do: interactive 3D, animated
data, and plots typed straight from an equation.** Figures render with WebGL
in the notebook and save as journal-ready PNG, SVG, or PDF.

<table>
<tr>
<td width="50%"><img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/gapminder.gif" alt="Gapminder: life expectancy against GDP per capita, animated from 1952 to 2007"></td>
<td width="50%"><img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/lidar.gif" alt="A lidar driving scene with detection boxes, orbiting"></td>
</tr>
<tr>
<td><sub><code>+ transition_time("year")</code>: gapminder in one line</sub></td>
<td><sub><code>geom_point3d()</code> + <code>geom_box3d()</code>: a 220k-point lidar sweep with detection boxes</sub></td>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/ripple.gif" alt="An animated ripple surface z = sin(r - t)"></td>
<td><img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/beta_slider.gif" alt="A beta density reshaping as two sliders move"></td>
</tr>
<tr>
<td><sub><code>geom_function("z = sin(sqrt(x^2 + y^2) - t)") + transition_time(t=(0, 6.28))</code></sub></td>
<td><sub><code>geom_function("y = dbeta(x, a, b)") + slider(a=…, b=…)</code></sub></td>
</tr>
</table>

```bash
pip install plot3
```

Contents: [Plot an equation](#plot-an-equation) ·
[Animate your data](#animate-your-data) · [3D and point clouds](#3d-and-point-clouds) ·
[One figure, two outputs](#one-figure-two-outputs) · [It's ggplot2](#its-ggplot2) ·
[Notebooks, SolveIt, CRAFT](#notebooks-solveit-and-craft) · [Install](#install) ·
[Full reference](docs/reference.md)

## Plot an equation

Write the maths as you would on paper. `geom_function` works out whether
it's a curve, an implicit shape, a region, or a 3D surface, and picks a
sensible window.

```python
from plot3 import *

(ggplot() + geom_function("y = sin(x)", xlim=(-4, 4))                        # a function and
 + geom_function("y = x - x^3/6", xlim=(-4, 4)))                            # its Taylor polynomial
ggplot() + geom_function("(x^2 + y^2 - 1)^3 = x^2 y^3")                     # implicit: a heart
ggplot() + geom_function("y > x^2", xlim=(-2, 2))                           # shaded region
ggplot() + geom_function("r = 1 + cos(theta)") + coord_polar()              # polar cardioid
ggplot() + geom_function(r"z = \frac{\sin\sqrt{x^2+y^2}}{\sqrt{x^2+y^2}}",  # LaTeX in, 3D sombrero out
                         xlim=(-10, 10), ylim=(-10, 10))
ggplot() + geom_vector_field("dx = y, dy = -sin(x) - 0.3y",                # a damped pendulum's
                          xlim=(-6, 6), ylim=(-3, 3), stream=True)          # phase portrait
```

<p>
<img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/fn_taylor.png" width="15%" alt="sin(x) and its Taylor polynomial">
<img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/fn_heart.png" width="15%" alt="Implicit heart curve">
<img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/fn_region.png" width="15%" alt="Shaded region y > x^2">
<img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/fn_cardioid.png" width="15%" alt="Polar cardioid">
<img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/fn_sombrero.png" width="15%" alt="3D sombrero surface">
<img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/fn_streamlines.png" width="15%" alt="Streamlines of a damped pendulum">
</p>

**Probability and calculus, with the answer printed.** No SciPy needed:

```python
inf = float("inf")
ggplot() + geom_function("y = dbeta(x, 2, 5)") + area(0.2, 0.5)          # labels P(0.2 ≤ X ≤ 0.5) = 0.546
ggplot() + geom_function("y = dnorm(x)") + area(-inf, -1.96) + area(1.96, inf)
ggplot() + geom_function("y = x^3 - 3x") + tangent(at=1) + derivative()
```

**Turn any coefficient into a slider or an animation:**

```python
ggplot() + geom_function("y = dbeta(x, a, b)") + slider(a=(0.5, 5), b=(0.5, 5))
ggplot() + geom_function("y = sin(x - t)") + transition_time(t=(0, 6.28))   # travelling wave
```

Equations mix freely with data:
`ggplot(df, aes(x="dose", y="response")) + geom_point() + geom_function("y = 2 + 0.6x")`.
`^` is power and `2x` means `2*x`. All functions, densities, and rules are in
the [reference](docs/reference.md#functions-and-maths).

## Animate your data

[gganimate](https://gganimate.com)'s grammar: add a transition and the plot
plays. The viewer interpolates between frames in the browser, with play,
pause, scrubbing, speed control, and one-click video export.

```python notest
import pandas as pd
gapminder = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/master/gapminder.csv")

(ggplot(gapminder, aes(x="gdpPercap", y="lifeExp", size="pop", colour="continent", group="country"))
 + geom_point(alpha=0.7)
 + scale_x_log10()
 + transition_time("year")
 + labs(title="{frame_time}", x="GDP per capita", y="Life expectancy"))
```

`{frame_time}` in the title shows the current year as it plays.
`transition_states("phase")` steps through categories instead of time.

Click a bubble to trace it: its path through every frame appears, with
the years along it and its name (from `group=`) riding beside it as it
plays. Click it again, or empty space, to clear it. This works in 3D too.

<details>
<summary>Run this without downloading anything (made-up gapminder-shaped data)</summary>

```python
import pandas as pd

countries = {"Brazil": "Americas", "China": "Asia", "Egypt": "Africa",
             "France": "Europe", "India": "Asia", "Mexico": "Americas"}
gapminder = pd.DataFrame([
    {"country": country, "continent": continent, "year": year,
     "gdpPercap": 800 * (1.03 + 0.01 * i) ** (year - 1952),
     "lifeExp": 45 + 0.35 * (year - 1952) + 2 * i,
     "pop": 2e7 * (1.02 + 0.002 * i) ** (year - 1952)}
    for i, (country, continent) in enumerate(countries.items())
    for year in range(1952, 2008, 5)
])

(ggplot(gapminder, aes(x="gdpPercap", y="lifeExp", size="pop", colour="continent", group="country"))
 + geom_point() + scale_x_log10() + transition_time("year") + labs(title="{frame_time}"))
```

</details>

## 3D and point clouds

Map `z` and the plot becomes a 3D scene you can orbit, zoom, and hover.
Hundreds of thousands of points stay smooth because they're drawn on the
GPU.

```python
import numpy as np

rng = np.random.default_rng(0)
arm = rng.integers(0, 2, 20_000)
r = rng.gamma(2, 1.2, 20_000)
theta = r * 0.9 + arm * np.pi + rng.normal(0, 0.25, 20_000)
galaxy = pd.DataFrame({"x": r * np.cos(theta), "y": r * np.sin(theta),
                       "z": rng.normal(0, 0.15, 20_000), "r": r})

(ggplot(galaxy, aes(x="x", y="y", z="z", colour="r"))
 + geom_point3d(size=0.01) + scale_colour_viridis_c(option="magma")
 + coord_3d(elev=40) + theme_lidar() + guides(colour="none"))
```

<p>
<img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/galaxy.gif" width="32%" alt="A 20,000-point spiral galaxy, orbiting">
<img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/peaks.png" width="32%" alt="The peaks surface coloured by height">
<img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/isosurface.png" width="32%" alt="Nested density isosurfaces of three clusters">
</p>

Surfaces, isosurfaces, and self-driving scenes use the same grammar:

```python notest
ggplot(grid, aes(x="x", y="y", z="height", fill="height")) + geom_surface()
ggplot(points, aes(x="x", y="y", z="z")) + geom_isosurface(levels=[0.2, 0.5, 0.8])

(ggplot(sweep, aes(x="x", y="y", z="z"))                 # a nuScenes / KITTI lidar sweep
 + geom_point3d()                                         # coloured by height, as lidar viewers do
 + geom_box3d(aes(length="l", width="w", height="h", angle="yaw", colour="class"), data=boxes)
 + coord_3d(azim=180, elev=28, zoom=1.6)                 # chase camera behind the car
 + theme_lidar())
```

`read_bin("scan.pcd.bin")` loads nuScenes-style point clouds, and NumPy
arrays work by column position: `aes(x=0, y=1, z=2, colour=3)`. Camera,
aspect, and point-budget options are in the
[reference](docs/reference.md#api-reference) under `coord_3d`.

## One figure, two outputs

The same object is an interactive figure in the notebook and a publication
figure on disk, with real fonts, correct DPI metadata, and `theme_bw` by
default.

```python notest
ggsave("fig1.pdf", p, width=3.5, height=2.6, units="in")         # vector, one journal column
ggsave("fig1.png", p, width=7, height=4.5, units="in", dpi=300)   # two columns
ggsave("fig1.html", p)                                            # interactive page to share
```

## It's ggplot2

If you know ggplot2, you already know plot3: `+` layers, `aes()`, geoms,
stats, scales, facets, themes, `labs()`, and patchwork's `|` and `/`, with
ggplot2's names and defaults.

```python
rng = np.random.default_rng(1)
arms = ["placebo", "low", "high"]
trial = pd.DataFrame({"arm": pd.Categorical(rng.choice(arms, 150), categories=arms),
                      "dose": rng.uniform(0, 10, 150)})
slope = trial.arm.map({"placebo": 0.05, "low": 0.35, "high": 0.7}).astype(float)
trial["response"] = 2 + slope * trial.dose + rng.normal(0, 0.9, 150)

a = (ggplot(trial, aes(x="dose", y="response", colour="arm"))
     + geom_point(alpha=0.7) + geom_smooth(method="lm", se=False)
     + scale_colour_hue() + theme_grey() + theme(legend_position="bottom"))
b = (ggplot(trial, aes(x="arm", y="response", fill="arm"))
     + geom_violin() + geom_jitter(width=0.1, height=0)
     + scale_fill_hue() + theme_grey() + theme(legend_position="none"))
fig = ((a | b) + plot_layout(widths=[3, 2])
       + plot_annotation(title="Response by dose and arm", tag_levels="A"))
```

`scale_colour_hue()` and `theme_grey()` are ggplot2's own defaults, so the
figure looks as it would from R. plot3's defaults are `theme_bw()` for saved
files and `theme_dark()` in the viewer.

<img src="https://raw.githubusercontent.com/rleyvasal/plot3/main/docs/img/ggplot2_look.png" alt="Two-panel figure in ggplot2's grey theme: response against dose with one fitted line per arm, and violins of response by arm">

Everything else you'd expect is there, including histograms, densities,
boxplots, bar positions, `stat_summary`, error bars, heatmaps, hexbins, 2D
densities, `facet_wrap` / `facet_grid`, Brewer, viridis, and Okabe–Ito
scales, date axes, `theme()` with `element_*()`, and `coord_flip` /
`coord_polar`. See the **[full reference](docs/reference.md)** for what
plot3 supports and how it differs from ggplot2, or use the
[ggplot2 docs](https://ggplot2.tidyverse.org/reference/) directly; the
names match.

In notebooks, bare column names work as in R: `aes(x=dose, y=response)`.

## Notebooks, SolveIt, and CRAFT

- **Jupyter / SolveIt**: bare column names and backticks work in `aes()`,
  `facet_wrap()`, and `facet_grid()` (`aes(x=`First Name`)`). Toggle with
  `enable_r_style()` / `disable_r_style()`. `.py` files keep quoted strings.
- **SolveIt** draws figures inline and hides their HTML from the model's
  context (`autohide(False)` to opt out).
- **VS Code notebooks** block WebGL in output cells, so plot3 opens figures
  in your browser (`PLOT3_DISPLAY=browser|iframe` to force a mode). The page
  is written to `.plot3_preview/latest.html` in the notebook's folder, or to
  the system temp folder when that folder is read-only.
- **CRAFT / `%gpu`**: the same `ggplot(...)` code runs on the remote kernel;
  stats run where the data lives and only a compact payload comes back.

```text
%run /path/to/plot3/plot3.py        # loads plot3 locally and seeds the GPU kernel
%plot3 df x=wt y=mpg color=cyl       # optional shortcut magic
```

## Install

```bash
pip install plot3
```

Python 3.10 or newer; NumPy and pandas are the only required packages.
Notebooks, pandas, and Polars tables work with no extras.

For PDF, and PNG with real fonts, add cairosvg and the Cairo library. SVG
needs neither.

```bash
pip install cairosvg
```

```bash
brew install cairo
```

(`apt install libcairo2` on Debian and Ubuntu.) `pip install "plot3[pdf]"`
installs plot3 and cairosvg together.

The latest unreleased code installs from GitHub with
`pip install "plot3 @ git+https://github.com/rleyvasal/plot3"`.

## Development

```bash
git clone https://github.com/rleyvasal/plot3 && cd plot3
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                       # includes every example in this README and docs/reference.md
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
