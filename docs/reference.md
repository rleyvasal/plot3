# plot3 reference

The ggplot2 side of plot3: statistical plots, annotation, scales, themes,
facets, and saving. Names and defaults follow ggplot2, so the
[ggplot2 reference](https://ggplot2.tidyverse.org/reference/) applies too;
this page lists what plot3 supports and where it differs. Back to the
[README](../README.md).

Contents: [Statistical plots](#statistical-plots) · [Annotation](#annotation) ·
[Scales](#scales) · [Themes and titles](#themes-and-titles) ·
[Facets and multi-panel figures](#facets-and-multi-panel-figures) ·
[Saving for a paper](#saving-for-a-paper) · [Functions and maths](#functions-and-maths) ·
[API reference](#api-reference) · [Differences from ggplot2](#differences-from-ggplot2)

The examples share this data:

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

## Statistical plots

```python
# Distributions
ggplot(trial, aes(x="response", fill="sex")) + geom_histogram(bins=20)            # stacked by group
(ggplot(trial, aes(x="response", y="after_stat(density)"))                         # density scale
 + geom_histogram(bins=20) + geom_density())
ggplot(trial, aes(x="response", fill="arm")) + geom_density(alpha=0.4)
ggplot(trial, aes(x="arm", y="response")) + geom_boxplot(outliers=False) + geom_jitter(width=0.15, height=0)
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
| Error bars from your own columns | `geom_errorbar(aes(ymin="mean - se", ymax="mean + se"))`, `geom_pointrange`, `geom_linerange`, `geom_crossbar`, `geom_errorbarh(aes(xmin=, xmax=))` |
| Summaries | `stat_summary(fun_data="mean_se" / "mean_cl_normal" / "mean_sdl" / "median_hilow")` |
| Heatmaps | `geom_tile(aes(x=, y=, fill=))` (or `geom_raster`), with `geom_text(aes(label=))` |
| Stacked areas | `geom_area(aes(fill=))`, `position="fill"` for shares |
| Steps, segments, rectangles | `geom_step()`, `geom_segment(aes(xend=, yend=), arrow=arrow())`, `geom_rect(aes(xmin=, xmax=, ymin=, ymax=))` |
| Horizontal layout | `+ coord_flip()` (bars, boxplots, densities, error bars) |
| Several datasets | `geom_rect(aes(...), data=periods)`: any layer can bring its own data |
| Points over grouped boxes | `geom_boxplot(aes(colour="sex"), outliers=False) + geom_point(position=position_jitterdodge())` |
| Labels beside points | `geom_text(position=position_nudge(y=0.3))`, or `nudge_y=` |
| Shapes and maps | `geom_polygon(aes(group="id", fill="region"))`, concave shapes included |
| Frequency lines | `geom_freqpoly(aes(colour="arm"), binwidth=0.5)` |
| Big scatters, 2D distributions | `geom_hex()`, `geom_bin_2d()`, `geom_count()`, `geom_density_2d()`, `geom_density_2d_filled()`, `stat_ellipse()` (95% by group) |
| Contours of a grid | `geom_contour(aes(x=, y=, z=))` |

`aes()` reads expressions over your columns, as ggplot2 does:
`aes(ymin="mean - se")`, `aes(y="log10(count)")`, `aes(colour="factor(cyl)")`,
`aes(colour="dose > 5")`, `aes(label="round(estimate, 2)")`. They use `+ - * /
^`, comparisons, and `log`, `log10`, `log2`, `exp`, `sqrt`, `abs`, `round`,
`floor`, `ceiling`, `factor`, `as.numeric`, `ifelse`, `pmin`, `pmax`, `mean`,
`median`, `sd`, `min`, `max`, `sum`; nothing else runs. The expression names
the axis or legend.

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
| Discrete colour / fill | `scale_colour_hue` (ggplot2's default colours), `scale_colour_manual`, `scale_colour_brewer(palette="Set2")`, `scale_colour_viridis_d`, `scale_colour_grey`, `scale_colour_okabe_ito` (colour-blind safe), `scale_colour_identity` (the column holds colours) |
| Continuous colour / fill | `scale_colour_gradient(low, high)`, `scale_colour_gradient2(low, mid, high, midpoint)`, `scale_colour_gradientn(colours, values)`, `scale_colour_distiller(palette="RdBu")`, `scale_colour_viridis_c`, `scale_colour_continuous(trans="log10")` |
| Size / alpha | `scale_size(range=(4, 23))` (by area across the data's range, as ggplot2), `scale_size_area(max_size=)` (area from zero), `scale_alpha(range=(0.1, 1))` for `aes(alpha=)` |
| Shape / linetype | `scale_shape_manual`, `scale_linetype_manual` |
| Limits | `expand_limits(y=0)` makes an axis reach a value with no data there |

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
`theme_minimal`, `theme_void` (the data alone), `theme_light`, `theme_dark`
(default in the interactive viewer), and `theme_lidar` (driving scenes). Each takes `base_size` (points) and `base_family`. `theme()` sets
`legend_position` (`"right"`, `"bottom"`, `"none"`, or `(x, y)` inside the
panel), `legend_title=False`, `panel_grid=False`, `axis_text_x_angle`,
`plot_title_hjust`, `base_size`, and `base_family`. `theme_grey` (ggplot2's
grey panel) and `theme_linedraw` are there too.

ggplot2's elements work, with R's dotted names or underscores:

```python
(ggplot(trial, aes(x="arm", y="response")) + geom_boxplot() + theme_bw()
 + theme(**{"axis.text.x": element_text(angle=45), "panel.grid": element_blank(),
            "plot.title": element_text(hjust=0.5),
            "panel.background": element_rect(fill="grey95")}))
```

`element_blank()` hides axis text, axis titles, the grid, the panel border,
or the legend title; colours in `element_text`, `element_line`, and
`element_rect` recolour text, grid, border, and backgrounds. Parts plot3
does not draw (minor grid, tick length) are accepted, and anything else it
cannot draw warns.

`ggtitle("Response", subtitle=)`, `xlab()`, and `ylab()` are shortcuts for
`labs()`. `guides(colour="none")` hides one legend (also `fill`, `size`,
`shape`, `linetype`) and keeps the others;
`guides(colour=guide_legend(title="Arm", reverse=True))` retitles or
reorders it. The `stat_*` spellings (`stat_smooth`, `stat_bin`,
`stat_count`, `stat_density`, `stat_function(fun=, args=)`) work too.

Zoom without dropping data with `coord_cartesian`: a smoother or boxplot is
still computed from every row, while `xlim()` and `scale_x_continuous(limits=)`
remove the rows outside first.

```python
(ggplot(trial, aes(x="dose", y="response"))
 + geom_point() + geom_smooth(method="lm") + geom_rug(alpha=0.4)
 + coord_cartesian(xlim=(2, 6)))
```

## Facets and multi-panel figures

```python
ggplot(trial, aes(x="dose", y="response")) + geom_point() + facet_wrap("arm")
ggplot(trial, aes(x="dose", y="response")) + geom_point() + facet_wrap("arm", labeller="label_both")
ggplot(trial, aes(x="dose", y="response")) + geom_point() + facet_wrap(vars("arm"))
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

## API reference

| Area | Functions |
|---|---|
| Figure | `ggplot(data, aes(...))`, `data >> ggplot(aes(...))`, `+`, `p.show()`, `p.save()`, `ggsave()` |
| Aesthetics | `aes(x, y, z, colour, fill, size, shape, linetype, group, label, ymin, ymax, xmin, xmax, xend, yend, sample)` |
| Points and lines | `geom_point`, `geom_jitter`, `geom_line`, `geom_path`, `geom_step`, `geom_segment`, `geom_text`, `geom_label` |
| Bars and areas | `geom_col`, `geom_bar`, `geom_histogram`, `geom_freqpoly`, `geom_area`, `geom_ribbon`, `geom_rect`, `geom_tile`/`geom_raster`, `geom_polygon` |
| Distributions | `geom_boxplot`, `geom_violin`, `geom_density`, `geom_qq`, `geom_qq_line`, `stat_ecdf`, `stat_summary` |
| 2D distributions | `geom_bin_2d`, `geom_hex`, `geom_count`, `geom_density_2d` / `stat_density_2d`, `geom_density_2d_filled`, `geom_contour`, `stat_ellipse` |
| Uncertainty and fits | `geom_errorbar`, `geom_errorbarh`, `geom_crossbar`, `geom_pointrange`, `geom_linerange`, `geom_smooth(method="loess"/"lm")` |
| Reference | `geom_hline`, `geom_vline`, `geom_abline`, `geom_rug`, `annotate("text"/"label"/"rect"/"segment"/"point")` |
| Positions | `position="stack"/"dodge"/"fill"/"identity"/"jitter"`, `position_dodge(width)`, `position_dodge2(padding)`, `position_stack()`, `position_fill()`, `position_jitter()`, `position_jitterdodge()`, `position_nudge()` |
| Functions | `geom_function`, `geom_vector_field`, `area`, `tangent`, `derivative` |
| Scales | see [Scales](#scales) |
| Coordinates | `coord_cartesian`, `coord_flip`, `coord_equal` / `coord_fixed`, `coord_polar`, `coord_3d` |
| Facets and layout | `facet_wrap`, `facet_grid`, `labeller="label_both"` / `labeller(var=dict)`, `p1 \| p2`, `p1 / p2`, `plot_layout(widths, heights, height)`, `plot_annotation` |
| Labels and themes | `labs(title, subtitle, caption, tag, x, y, colour, fill, alpha)`, `ggtitle`, `xlab`, `ylab`, `guides`, `theme_*`, `theme()`, `element_text`, `element_line`, `element_rect`, `element_blank` |
| Animation | `transition_time`, `transition_states`, `slider` |
| 3D | `geom_point3d`, `geom_surface`, `geom_isosurface`, `geom_box3d`, `stat_density_3d`, `read_bin` |

## Differences from ggplot2

- Python needs quotes outside notebooks: `aes(x="wt")`. In Jupyter and
  SolveIt, `aes(x=wt)` works.
- `labs(x=None)` (or `labs(x="")`) removes a title, as ggplot2's `labs(x = NULL)`.
- `fill` colours filled shapes; points and lines use `colour`, as in ggplot2.
- Categories sort alphabetically (numbers numerically); use
  `pd.Categorical` or `scale_x_discrete(limits=)` for your own order.
- Text `size` is in millimetres, as in ggplot2; `geom_point(size=)` is in
  pixels in 2D.
- Saved files default to `theme_bw`, the interactive viewer to `theme_dark`.
  On light themes, a layer with no colour of its own is drawn as ggplot2
  draws it: black points and lines, grey bars, white boxes and violins. The
  dark viewer uses its own blue instead.
- Building a figure prints nothing. `PLOT3_VERBOSE=1` prints each figure's
  size in KB, for embedding in slides or pages with a size cap.

