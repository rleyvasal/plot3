"""plot3 — grammar-of-graphics plotting on three.js."""

from __future__ import annotations

from plot3.__version__ import __version__
from plot3.geoms import (
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
    labs,
    scale_color_continuous,
    scale_color_viridis_c,
    scale_colour_continuous,
    scale_colour_viridis_c,
    theme_dark,
    theme_light,
)
from plot3.ggplot import autohide, ggsave, ggplot
from plot3.io import read_bin
from plot3.jupyter import load_ipython_extension, register_plot3

__all__ = [
    "ggplot",
    "aes",
    "geom_point",
    "geom_line",
    "geom_path",
    "geom_col",
    "geom_bar",
    "geom_histogram",
    "geom_boxplot",
    "geom_density",
    "geom_violin",
    "facet_wrap",
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
