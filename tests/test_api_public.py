"""Public API surface — exports, imports, version."""

from __future__ import annotations

import plot3


REQUIRED = {
    "ggplot",
    "aes",
    "geom_point",
    "geom_point3d",
    "geom_line",
    "geom_path",
    "geom_col",
    "geom_bar",
    "geom_histogram",
    "geom_boxplot",
    "geom_density",
    "geom_violin",
    "geom_surface",
    "geom_isosurface",
    "stat_density_3d",
    "facet_wrap",
    "coord_3d",
    "labs",
    "scale_colour_continuous",
    "scale_colour_viridis_c",
    "theme_dark",
    "theme_light",
    "ggsave",
    "read_bin",
    "autohide",
}


def test_version_present():
    assert isinstance(plot3.__version__, str)
    assert plot3.__version__


def test_required_exports():
    missing = sorted(name for name in REQUIRED if not hasattr(plot3, name))
    assert missing == []
    for name in REQUIRED:
        assert name in plot3.__all__


def test_from_plot3_star_symbols():
    # Ensure __all__ is importable without error.
    ns = {}
    exec("from plot3 import *", ns)  # noqa: S102
    for name in REQUIRED:
        assert name in ns
