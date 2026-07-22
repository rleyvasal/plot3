"""plot3 — grammar-of-graphics plotting on three.js."""

from __future__ import annotations

from plot3.__version__ import __version__
from plot3.geoms import (
    aes,
    coord_3d,
    facet_wrap,
    geom_bar,
    geom_boxplot,
    geom_col,
    geom_density,
    geom_histogram,
    geom_isosurface,
    geom_line,
    geom_path,
    geom_point,
    geom_point3d,
    geom_surface,
    geom_violin,
    labs,
    stat_density_3d,
    scale_color_continuous,
    scale_color_viridis_c,
    scale_colour_continuous,
    scale_colour_viridis_c,
    theme_dark,
    theme_light,
)
from plot3.ggplot import autohide, ggsave, ggplot  # show via ggplot.show
from plot3.io import read_bin
from plot3.jupyter import (
    disable_r_style,
    enable_r_style,
    load_ipython_extension,
    register_plot3,
)

__all__ = [
    "ggplot",
    "aes",
    "geom_point",
    "geom_point3d",
    "geom_surface",
    "geom_isosurface",
    "stat_density_3d",
    "geom_line",
    "geom_path",
    "geom_col",
    "geom_bar",
    "geom_histogram",
    "geom_boxplot",
    "geom_density",
    "geom_violin",
    "facet_wrap",
    "coord_3d",
    "labs",
    "scale_colour_continuous",
    "scale_color_continuous",
    "scale_colour_viridis_c",
    "scale_color_viridis_c",
    "theme_dark",
    "theme_light",
    "ggsave",
    "read_bin",
    "autohide",
    "enable_r_style",
    "disable_r_style",
    "load_ipython_extension",
    "register_plot3",
    "__version__",
]

# Auto-register magics when loaded inside IPython (CRAFT / Jupyter).
try:
    from IPython import get_ipython as _get_ipython

    if _get_ipython is not None and _get_ipython() is not None:
        register_plot3(quiet=False)
except Exception:
    pass
