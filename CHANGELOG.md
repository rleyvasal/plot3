# Changelog

All notable changes to plot3. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may change behaviour, and those changes are listed
under **Changed**.

## Unreleased

### Added

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
- Infinite values were dropped silently; they now get a note, *Removed 1 row
  containing non-finite values*.
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
