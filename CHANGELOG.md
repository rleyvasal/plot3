# Changelog

All notable changes to plot3. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may change behaviour, and those changes are listed
under **Changed**.

## 0.6.6 — 2026-10-10

Plot layers on their own lines in notebooks.

### Added

- In notebooks, a plot's layers can go on their own lines with no
  parentheses, as in R: `ggplot(df, aes(...))` then `+ geom_point()` and
  `+ scale_x_log10()` on the lines below, indented or not, with comments
  between. Only a `+` followed by a plot3 function joins, so other Python
  (`y = +x`) is never rewritten. `join_layer_lines()` does the same for
  tools that export notebooks.

## 0.6.5 — 2026-10-10

Labelled paths for clicked bubbles in animations.

### Changed

- Clicking a bubble in an animation traces its path, as before, and now
  labels it: the years along the path, and the bubble's name (its `group=`
  value) beside it as it moves; in 3D the path's ends are labelled. The
  tooltip says "click to show its path" (or hide), and only a click on the
  bubble itself counts, so a stray click near a small dot draws nothing.

## 0.6.4 — 2026-10-09

Legends no longer cover short plots.

### Fixed

- In a short output area (a SolveIt or notebook cell), a legend taller than
  the plot no longer covers the bottom of the plot and the x axis. It runs
  in rows under the axes, and the plot (2D or 3D) shrinks to make room. In
  a taller window the legend stays at the top right as before.

## 0.6.3 — 2026-10-09

A plain `pip install plot3`.

### Changed

- `pip install plot3` is the whole install for notebooks, pandas, and
  Polars. The `jupyter`, `polars`, `fast`, and `export` extras are gone:
  notebooks and Polars users already have IPython and Polars, and
  contourpy saved only milliseconds on large contour grids.
- PDF and real-font PNG need cairosvg, now offered as the `pdf` extra
  (`pip install "plot3[pdf]"`) or directly (`pip install cairosvg`).
  Messages name the package to install instead of an extra.

## 0.6.2 — 2026-10-09

Categorical colour order in every geom, and frameless ggplot2 legends.

### Fixed

- A categorical colour column (`pd.Categorical(..., categories=[...])`)
  keeps its order, and so its colours, in every geom. Violins, boxplots,
  densities, histograms, bars, smoothers, and summaries sorted the levels
  as text, so a violin and a scatter of the same column in one figure gave
  a group different colours.
- Legends under `theme_grey()` and `theme_classic()` have no frame, as in
  ggplot2; they had a dark grey one.

## 0.6.1 — 2026-10-09

ggplot2's grey facet strips.

### Fixed

- Facet strips are ggplot2's grey85 in `theme_grey()` and `theme_bw()`
  (saved files' default), in the viewer and in saved files. Under
  `theme_grey()` they took the white grid colour and disappeared.

## 0.6.0 — 2026-10-09

Free-axis facets, dashed smoothers, and column expressions in notebook aes().

### Added

- `facet_wrap()` and `facet_grid()` take `scales="free_x"` and
  `scales="free_y"`, freeing one axis per panel as in ggplot2.
- `geom_smooth(linetype=)` dashes the fitted line.

### Fixed

- `theme(legend_position="none")` hides the legend of a faceted figure, in
  the viewer and in saved files. It used to be drawn anyway.

- In notebooks, an expression over columns in `aes()` works bare:
  `aes(colour = factor(cyl))`, `aes(x = log10(pop))`, `aes(colour = cyl > 4)`.
  Each column name used to be quoted on its own (`factor("cyl")`), which
  failed; the expression now reaches `aes()` whole, as `"factor(cyl)"`.

## 0.5.1 — 2026-10-09

A fix for layers with a set colour.

### Fixed

- A colour set on a layer replaces the plot's colour mapping for that layer,
  as in ggplot2. `geom_point(data=centres, colour="black")` under
  `ggplot(aes(colour="cluster"))` draws black points (they were coloured by
  the mapping), and its data no longer needs a `cluster` column. Filled
  layers keep their mapped fill when `colour=` sets the outline.

## 0.5.0 — 2026-10-09

Cleaner maths and 3D figures, and a README that shows them.

### Fixed

- `geom_isosurface` draws closed, smooth surfaces. It used to emit a loose
  tile per voxel face, which showed as scattered specks in the viewer and
  in saved files. The mesh now comes from marching tetrahedra, with shared
  vertices and outward-facing triangles.
- `geom_vector_field(stream=True)` draws evenly spaced streamlines with an
  arrowhead each. Closed orbits close once instead of wrapping over
  themselves, and lines no longer pile up.
- Implicit curves with a cusp (`(x^2 + y^2 - 1)^3 = x^2 y^3`, `y^2 = x^3`)
  no longer stop short of the cusp, and a curve leaving through the top of
  the window is no longer joined to one at the bottom by a stray line.
- Saved 3D figures keep a long axis title (`labs(z="elevation")`) on the
  canvas, making room beside the cube for it.
- Implicit curves sit on the true curve where the equation has a repeated
  root (the sides of the heart at y = 0 were off by half a grid cell, which
  showed as kinks). Each crossing is now solved on the function itself.
- `geom_isosurface` colours each surface by its level (0.2, 0.45, …) under
  a "level" legend, not by its position in the list (0, 1, 2).
- Saved figures with equal aspect (`coord_equal`, implicit curves) tick the
  whole visible axis, as the viewer does, instead of bunching the data's
  ticks in a strip.

### Changed

- `scale_x_log10()` and `scale_y_log10()` take `name`, `limits`, `breaks`,
  and `labels`, as `scale_x_continuous` does.
- A surface or implicit curve can sweep a third letter without a stand-in
  value: `geom_function("z = sin(x - t)") + transition_time(t=(0, 6.28))`
  and `geom_function("x^2 + y^2 = r^2") + slider(r=(0.5, 2))` work as
  written. A letter nothing sweeps still asks for a value.
- The README leads with equation plots, animation, and 3D, with images of
  each; the full ggplot2-side reference moved to `docs/reference.md`.

## 0.4.2 — 2026-10-08

Steadier animation labels.

### Fixed

- The big frame label in animations (`t = 1.53`, the year) no longer shifts
  sideways or changes size as its value changes: digits have a fixed width,
  the label keeps one size for the whole animation, and it sits in a box as
  wide as its widest value.
- A formula animated with `transition_time(t = ...)` no longer shows the first
  frame's value in its legend and tooltip (`y = sin(x − t)  (t = 0)`) while
  the animation plays.

## 0.4.1 — 2026-10-08

A fix for showing figures from VS Code notebooks.

### Fixed

- Showing a figure from a notebook whose kernel runs in a read-only folder
  (VS Code starts notebooks outside a workspace in `/`) failed with
  `OSError: Read-only file system: '/.plot3_preview'`. The browser preview
  now falls back to the system temp folder.
- The README's gapminder animation runs as written, with gapminder's column
  names.

## 0.4.0 — 2026-10-08

ggplot2's everyday grammar, its defaults, and lidar scenes: R code ports
with fewer changes and saves the way ggplot2 draws it.

### Added

- 2D distributions: `geom_bin_2d`, `geom_hex`, `geom_density_2d`
  (`stat_density_2d`), `geom_density_2d_filled`, `geom_contour` for gridded
  `z`, and `stat_ellipse(type="t" / "norm" / "euclid", level=)`.
- `geom_count`: one point per distinct (x, y), sized by how many rows share
  it, with a size legend titled `n`.
- ggplot2 spellings: `stat_smooth`, `stat_bin`, `stat_count`, `stat_density`,
  and `stat_function(fun=dnorm, args={"mean": 2})`.
- `geom_blank()`: its data reach the scales, nothing is drawn.
- `guides(colour=guide_legend(title=, reverse=))`, `guide_colourbar(title=)`,
  and `guide_none()`.
- `vars()` for facets: `facet_wrap(vars(cyl))`,
  `facet_grid(rows=vars(drv), cols=vars(cyl))`. Python's own `vars(obj)`
  and `vars()` still work after `from plot3 import *`.
- `position_dodge2(padding=)`: gaps between dodged bars; boxplots use it.
- `scale_colour_hue()`: ggplot2's default colours (#F8766D, #00BA38,
  #619CFF for three groups).
- `plot_layout(height=)`; multi-panel figures in the viewer are as tall as
  their rows of plots (400 px per row, 480 px at least) instead of 560 px.
- `arrow(angle, length, ends, type)` for `geom_segment`, `geom_path`,
  `geom_line`, and `annotate("segment")`, in the viewer and saved files.
- 3D figures draw their three far walls with grid lines, as matplotlib and
  plotly do, and leave out the box edges at the near corner. The viewer
  updates them as you orbit.
- `coord_3d(aspect="auto")`, the new default: a tall cloud is shortened to
  twice its width. `aspect="data"` keeps true proportions.
- `geom_box3d()`: wireframe boxes from centre, size, and heading (nuScenes /
  KITTI detections). Each class gets its own colour and legend entry, so a
  cloud coloured by height and boxes coloured by class share one figure;
  the legend shows the classes and the height colour bar together.
- `theme_lidar()`: a driving-scene look, with a black page, no box, grid, or
  tick labels, a green-cyan-violet height ramp, and class colours that read
  on black.
- `coord_3d(elev=, azim=, zoom=)` sets the starting camera, in the viewer
  and in saved files.
- `coord_cartesian(xlim=, ylim=, expand=)`: zoom without dropping rows, so
  statistics use all the data. Marks past the panel are clipped.
- `geom_rug(sides=, length=)`, with group colours, under `coord_flip` too.
- `ggtitle()`, `xlab()`, `ylab()`, `guides(colour="none")` (and `fill`,
  `size`, `shape`, `linetype`), and `theme_void()`.
- Histograms on the density scale: `aes(y="after_stat(density)")` (also
  `"..density.."`), per group when grouped.
- `geom_boxplot(outliers=False)` and `outlier_shape=None` hide outlier points.
- Expressions in `aes()`: `aes(ymin="mean - se")`, `aes(colour="factor(cyl)")`,
  `aes(y="log10(count)")`, comparisons, and a short list of functions,
  evaluated over the data's columns (no other code runs), on pandas and
  polars data and per-layer `data=`.
- `geom_freqpoly`, `geom_crossbar`, `geom_errorbarh`, and `geom_polygon`
  (concave shapes fill correctly, in the viewer and saved files).
- Scales: `scale_*_gradientn`, `scale_*_distiller`, `scale_*_identity`,
  `scale_fill_viridis_c`, `scale_size(range=)` (by area over the data's
  range, as ggplot2), `scale_size_area(max_size=)`, and `aes(alpha=)` with
  `scale_alpha(range=)`. A column mapped to both size and alpha gets one
  legend.
- `coord_fixed()`, `expand_limits(x=, y=)`, `theme_grey()` / `theme_gray()`
  (ggplot2's grey panel), and `theme_linedraw()`.
- ggplot2's theme elements: `theme(axis_text_x=element_text(angle=45),
  panel_grid=element_blank(), panel_background=element_rect(fill=...))`,
  with R's dotted names too. Elements plot3 cannot draw warn.
- Facet labellers: `labeller="label_both"`, `labeller(arm=label_both)`,
  `labeller(sex={"F": "Female"})`, `as_labeller(dict)`, or a function.
- `position_jitter()`, `position_jitterdodge()`, and `position_nudge()`
  (`geom_point(position=...)`; `geom_text(position=position_nudge(...))`).
- An unknown parameter warns, with the closest known one:
  *Ignoring unknown parameter in geom_point(): colr (did you mean color?)*.

### Changed

- Axes get about five round ticks, as ggplot2's: 0, 2, 4, 6, 8 for 0 to
  9, not every unit. Labels stay distinct for large numbers (1000000000001,
  not "1e+12" five times), and thinned labels keep round values.
- On light themes (saved files), layers with no colour of their own are
  black, bars grey35, and boxes and violins white with a dark outline, as in
  ggplot2. The dark viewer keeps its blue.
- `labs(x=None)` removes a title, as `labs(x = NULL)` does in R.
- `facet_wrap` lays panels out as ggplot2 does: 3 in a row, 4 in 2 x 2, 5 or
  6 in 2 rows of 3.
- A missing column names the closest match and lists the data's columns.
- Building a figure prints nothing; `PLOT3_VERBOSE=1` prints sizes.
- 3D surfaces from `geom_function` are coloured by height (viridis) with a
  colour bar, unless you set a colour.
- Saved files draw continuous colour bars vertically, labelled with round
  values and white ticks inside the bar, as in ggplot2.
- Point clouds with no colour of their own are coloured by height (viridis),
  and numeric colour on any 3D figure defaults to viridis.
- Default 3D point sizes are visible: about 5 px for a few hundred points,
  down to a fine grain for millions (they were about 1 px at any density).
- Size legends show round breaks (5, 10, 15) instead of the data maximum.
- `geom_boxplot(aes(fill=))` draws filled boxes with dark outlines, whiskers,
  and medians, as ggplot2 does; `aes(colour=)` keeps coloured outlines.
- 2D marks are clipped to the panel in saved files, as in ggplot2.
- Saved 3D figures use the viewer's perspective camera instead of an
  orthographic one, so what you save looks like what you orbit. The box is
  centred by its outline, and gets a top margin when an axis is labelled
  along its top edge.

### Fixed

- Security: text from the data or labels could end the viewer page's
  script (`</script>` in a title or category) and run as HTML, and
  category names were inserted into the viewer's axes and tooltips as
  markup. All of it is now escaped, so a plot3 HTML file made from
  untrusted data shows that text instead of running it.
- Plain points in the viewer were drawn as squares (WebGL's default); they
  are round, as in saved files, in 2D and 3D.
- Size legends for whole-number data (counts) showed 0.5 and 1.5; their
  breaks are whole numbers now.
- In the viewer, clicking a legend row hid the wrong group when the legend
  was reordered (`scale_colour_manual(breaks=)`, a reversed guide).
- Saved 3D figures: tick labels no longer collide at the cube's corners, the
  cube uses the space above it, and surfaces show no seams between triangles.
- 3D tick labels on a short or foreshortened edge are thinned to every
  second (or third) round value instead of overlapping, in saved files and
  in the viewer.
- The viewer drew per-point colours (colour gradients, height colours) paler
  than their colour bar; colours now match it exactly.
- The viewer's size legend overlapped the colour bar's numbers.
- Faceted viewer figures showed only colour squares in the legend: shapes,
  line types, colour bars, and size legends now appear.
- `geom_function` drew narrow peaks short when they fell between samples
  (`exp(-2000 x^2)` topped out at 0.71); it now samples more finely around
  peaks and valleys, unless you set `n=`.
- Infinite values were dropped silently; they now get a note, *Removed 1 row
  containing non-finite values*.
- Boxplots grouped by a second variable (`aes(x="arm", colour="sex")`)
  overlapped; they now sit side by side, as ggplot2 dodges them.
- Nudged labels could fall outside the panel; the scales now make room.
- Sized points in the viewer were half size on high-density screens and a
  little smaller than in saved files; they now match saved files.
- The viewer drew its 2D grid over the data, hiding whisker stems on
  category lines; the grid is now behind the data. Box outlines in the
  viewer were 1 px and missing their right, top, and bottom sides.

## 0.3.0 — 2026-10-07

A grammar of graphics complete enough for most journal figures, plus
functions, distributions, and animation.

### Added

- **Statistical layers**: `geom_jitter`, `geom_errorbar`, `geom_linerange`,
  `geom_pointrange`, `geom_ribbon`, `geom_smooth` (loess and lm with t-based
  bands), `stat_summary` (`mean_se`, `mean_cl_normal`, `mean_sdl`,
  `median_hilow`), `geom_tile` / `geom_raster`, `geom_area`, `geom_step`,
  `geom_segment`, `geom_rect`, `geom_qq`, `geom_qq_line`, `stat_ecdf`.
- **Positions**: `position="stack"`, `"dodge"`, `"fill"`, and
  `position_dodge()` for bars, histograms, areas, and error bars.
- **Annotation**: `geom_text`, `geom_label`, `annotate()`, `geom_hline`,
  `geom_vline`, `geom_abline`.
- **Aesthetics**: `fill` (separate from `colour`), `shape`, `linetype`,
  `label`, `ymin`/`ymax`, `xmin`/`xmax`, `xend`/`yend`, `sample`.
- **Scales**: `scale_x_continuous` / `scale_y_continuous` (`limits`,
  `breaks`, `labels`, `trans`), `scale_x_discrete`, `scale_x_date`,
  `scale_*_reverse`, `xlim`, `ylim`, `lims`; `scale_colour_manual`,
  `_brewer`, `_viridis_d`, `_grey`, `_okabe_ito`, `_gradient`, `_gradient2`
  (with `fill` and `color` names); `scale_shape_manual`,
  `scale_linetype_manual`; label formats `"percent"`, `"comma"`, `"dollar"`.
- **Layout**: `facet_grid`, `coord_flip`, multi-plot figures with `|`, `/`,
  `plot_layout()`, and `plot_annotation(tag_levels=)`.
- **Titles and themes**: `labs(subtitle, caption, tag)`, `theme_bw`,
  `theme_classic`, `theme_minimal`, and `theme()` options (`legend_position`,
  `legend_title`, `panel_grid`, `axis_text_x_angle`, `plot_title_hjust`,
  `base_size`, `base_family`).
- **Export**: `ggsave` writes PNG, SVG, and PDF with real fonts, physical
  sizes (`units="in"`, `"cm"`, `"mm"`, `dpi=`), DPI metadata, and fonts for
  non-Latin text. The viewer has a Save menu.
- **Functions**: `geom_function` for curves, implicit equations, inequalities,
  parametric and polar curves, surfaces, and vector fields; LaTeX input;
  `area()` with probability labels and infinite limits; `tangent()`,
  `derivative()`; special functions and R-style densities (`dnorm`, `dbeta`,
  `dt`, …) and CDFs (`pnorm`, `qt`, `pbeta`, …).
- **Animation**: `transition_time`, `transition_states`, and `slider()`.
- **Data**: per-layer `data=`; booleans as categories; more than eight colour
  groups; CSS and R colour names (`"steelblue"`, `"grey50"`).
- Messages for dropped rows (*Removed 3 rows containing missing values*).
- `LICENSE`, this changelog, and a test that runs every README example.

### Changed

- Saved files (PNG, SVG, PDF) default to `theme_bw`. The interactive viewer
  keeps `theme_dark`.
- `aes(fill=)` is no longer an alias for `colour`: it colours filled shapes,
  and points and lines ignore it, as in ggplot2.
- Categories sort as in ggplot2 (alphabetically, numbers numerically) for
  bars, violins, and facets, instead of first appearance.
- Facets draw one legend, shared axis titles, and strip labels instead of
  repeating the title in every panel.
- Computed axes are titled `count` and `density`.
- Error bars, reference lines, and text default to the theme's ink colour
  (black on light themes).
- Layer colours in the figure spec are `#rrggbb`.

### Fixed

- Category edges at the outermost half-step lost their position (heatmaps,
  dodged bars).
- Named colours were drawn black in saved files.
- Date axes in saved files printed a label for every day.
- Lines drew under translucent layers in the viewer.
- `theme(legend_position="none")` was ignored by the viewer.

## 0.2.0 — 2026-07-29

- Legend click-filtering, robust colour limits, `scale_colour_continuous`,
  `scale_colour_viridis_c`.
- `geom_col`, `geom_bar`, `geom_histogram` with ggplot2 width rules,
  `geom_boxplot`, `geom_density`, `geom_violin`, `facet_wrap`.
- 3D: `geom_point3d`, `coord_3d`, `geom_surface`, `geom_isosurface`,
  `stat_density_3d`, `read_bin`.
- pandas, Polars, tidy3, and NumPy tables; R-style bare column names in
  notebooks; SolveIt and CRAFT integration; the PlotPayload contract.

## 0.1.0 — 2026-07-18

- First release: ggplot grammar for 2D and 3D WebGL figures from DataFrames.
