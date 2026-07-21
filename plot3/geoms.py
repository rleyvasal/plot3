"""Grammar objects: aes, geoms, labs, colour scales, theme helpers."""

from __future__ import annotations

from plot3.themes import _CONT_PALETTES

class aes(dict):
    """Aesthetic mapping: aes(x=, y=, z=, colour=/color=, fill=, group=).

    ``fill`` is accepted as an alias of ``colour`` when colour is omitted
    (useful for surfaces).
    """

    def __init__(
        self,
        x=None,
        y=None,
        z=None,
        color=None,
        colour=None,
        fill=None,
        group=None,
    ):
        super().__init__()
        colour_value = (
            color
            if color is not None
            else colour
            if colour is not None
            else fill
        )
        for k, v in (("x", x), ("y", y), ("z", z),
                     ("color", colour_value),
                     ("group", group)):
            if v is not None:
                self[k] = v


class _Geom:
    kind = ""
    sort_x = False

    def __init__(self, mapping: aes | None = None, *, color=None, colour=None,
                 alpha=None, **params):
        self.mapping = mapping or aes()
        self.const_color = color if color is not None else colour
        self.alpha = alpha
        self.params = params


class geom_point(_Geom):
    """Scatter points.

    In **2D**, ``size`` is pixels. In **3D** (when ``aes(z=...)`` is set),
    ``size`` is scene units with distance attenuation — lidar clouds typically
    use ``0.001``–``0.02``. Prefer :class:`geom_point3d` for explicit 3D intent.
    """

    kind = "point"

    def __init__(self, mapping=None, *, size=None, **kw):
        super().__init__(mapping, **kw)
        self.size = size


class geom_point3d(geom_point):
    """3D scatter / point-cloud marks (same ``kind`` as :class:`geom_point`).

    Use with ``aes(x=, y=, z=)``. Size is in **scene units** under the default
    :class:`coord_3d` (``size_mode="scene"``).
    """

    def __init__(self, mapping=None, *, size=0.01, **kw):
        super().__init__(mapping, size=size, **kw)


class coord_3d:
    """3D coordinate system options for orbit-view figures.

    Parameters
    ----------
    aspect:
        ``"data"`` preserves relative axis spans (default). ``"equal"`` forces
        a unit cube (equal scale on x/y/z).
    size_mode:
        ``"scene"`` — point size attenuates with distance (lidar).
        ``"screen"`` — constant pixel size.
    max_points:
        If set, deterministically stride-subsample rows when building so huge
        clouds stay interactive in HTML.
    """

    def __init__(
        self,
        *,
        aspect: str = "data",
        size_mode: str = "scene",
        max_points: int | None = None,
    ):
        if aspect not in {"data", "equal"}:
            raise ValueError("aspect must be 'data' or 'equal'")
        if size_mode not in {"scene", "screen"}:
            raise ValueError("size_mode must be 'scene' or 'screen'")
        if max_points is not None and int(max_points) < 1:
            raise ValueError("max_points must be positive")
        self.aspect = aspect
        self.size_mode = size_mode
        self.max_points = None if max_points is None else int(max_points)

    def to_spec(self) -> dict:
        return {
            "aspect": self.aspect,
            "sizeMode": self.size_mode,
            "maxPoints": self.max_points,
        }


class geom_surface(_Geom):
    """3D surface from a regular x–y grid (height in ``z``).

    Requires ``aes(x=, y=, z=)`` on a **complete rectangular grid** in long
    form (one row per cell). Optional ``colour``/``fill`` colours vertices.
    Forces 3D mode.

    Parameters
    ----------
    wireframe:
        If True, draw only mesh edges.
    alpha:
        Face opacity (default 0.95).
    """

    kind = "surface"

    def __init__(self, mapping=None, *, wireframe: bool = False, alpha=0.95, **kw):
        super().__init__(mapping, alpha=alpha, **kw)
        self.wireframe = bool(wireframe)


class stat_density_3d:
    """3D density-grid options for :class:`geom_isosurface`.

    Not drawable alone. Add before ``geom_isosurface`` to set the histogram
    resolution (and keep a ggplot2-shaped call site)::

        ggplot(df, aes(x, y, z)) + stat_density_3d(n=24) + geom_isosurface(levels=[0.3, 0.7])
    """

    kind = "density_3d_stat"

    def __init__(self, *, n: int = 32):
        self.n = int(max(8, min(64, n)))


class geom_isosurface(_Geom):
    """Isosurface of a 3D density estimate from point samples.

    Requires ``aes(x=, y=, z=)`` on scatter-like data. v1 **embeds** density
    estimation (histogram grid + light smoothing) then extracts surfaces at
    the given levels. Levels are fractions of peak density in ``[0, 1]``.

    Parameters
    ----------
    levels:
        One or more relative thresholds (default ``[0.25, 0.5, 0.75]``).
    n:
        Density grid bins per axis (8–64). Overridden by a preceding
        :class:`stat_density_3d` if present on the figure.
    colour_by:
        ``"level"`` colours mesh vertices by isolevel index (default).
    wireframe, alpha:
        Same idea as :class:`geom_surface`.
    """

    kind = "isosurface"

    def __init__(
        self,
        mapping=None,
        *,
        levels: list[float] | tuple[float, ...] | None = None,
        n: int = 32,
        colour_by: str = "level",
        wireframe: bool = False,
        alpha: float = 0.55,
        **kw,
    ):
        super().__init__(mapping, alpha=alpha, **kw)
        if levels is None:
            levels = (0.25, 0.5, 0.75)
        self.levels = tuple(float(x) for x in levels)
        if not self.levels:
            raise ValueError("geom_isosurface() needs at least one level")
        self.n = int(max(8, min(64, n)))
        if colour_by not in {"level", "none"}:
            raise ValueError("colour_by must be 'level' or 'none'")
        self.colour_by = colour_by
        self.wireframe = bool(wireframe)


class geom_path(_Geom):
    kind = "line"
    sort_x = False  # ggplot2 geom_path: connect in data order

    def __init__(self, mapping=None, *, linewidth=None, width=None, **kw):
        super().__init__(mapping, **kw)
        self.linewidth = linewidth if linewidth is not None else (width or 2.0)


class geom_line(geom_path):
    sort_x = True  # ggplot2 geom_line: connect in order of x


class geom_col(_Geom):
    """Bars with heights from ``y`` (ggplot2 ``geom_col``).

    Requires ``aes(x=, y=)``. ``x`` may be categorical or numeric. Optional
    ``width`` is the bar width as a fraction of the median x-spacing (default
    0.9). 2D only.
    """

    kind = "col"

    def __init__(self, mapping=None, *, width=0.9, **kw):
        super().__init__(mapping, **kw)
        self.width = float(width)
        self.data_override = None  # optional layer-local frame (stats)


class geom_bar(_Geom):
    """Count bars for a discrete ``x`` (ggplot2 ``geom_bar``).

    Only ``aes(x=)`` is required; counts become ``y``. Expanded to
    ``geom_col`` at build time. 2D only.
    """

    kind = "bar"

    def __init__(self, mapping=None, *, width=0.9, **kw):
        super().__init__(mapping, **kw)
        self.width = float(width)


class geom_histogram(_Geom):
    """Histogram of a continuous ``x`` (ggplot2 ``geom_histogram``).

    Only ``aes(x=)`` is required. Bins are computed in Python and drawn as
    ``geom_col``. 2D only.
    """

    kind = "histogram"

    def __init__(self, mapping=None, *, bins=30, width=1.0, **kw):
        super().__init__(mapping, **kw)
        self.bins = int(bins)
        self.width = float(width)


class geom_boxplot(_Geom):
    """Box-and-whisker summary of ``y`` by ``x`` (ggplot2 ``geom_boxplot``).

    Requires ``aes(x=, y=)``. ``x`` is usually categorical; ``y`` is numeric.
    Whiskers use the Tukey rule (``coef`` × IQR, default 1.5). Outliers are
    drawn as points. 2D only.

    Parameters
    ----------
    width:
        Box width as a fraction of category spacing (default 0.75).
    outlier_size:
        Outlier point size in pixels (default 3).
    coef:
        Whisker fence multiplier on IQR (default 1.5). Set ``0`` to extend
        whiskers to the data min/max with no outliers.
    """

    kind = "boxplot"

    def __init__(
        self,
        mapping=None,
        *,
        width=0.75,
        outlier_size=3.0,
        coef=1.5,
        **kw,
    ):
        super().__init__(mapping, **kw)
        self.width = float(width)
        self.outlier_size = float(outlier_size)
        self.coef = float(coef)


class geom_density(_Geom):
    """Kernel density estimate of a continuous variable (ggplot2 ``geom_density``).

    Requires ``aes(x=)``. Optional ``colour``/``color`` draws one curve per
    group. Set ``fill=True`` (default) to shade under the curve. 2D only.
    """

    kind = "density"

    def __init__(
        self,
        mapping=None,
        *,
        n=512,
        adjust=1.0,
        fill=True,
        linewidth=1.5,
        **kw,
    ):
        super().__init__(mapping, **kw)
        self.n = int(n)
        self.adjust = float(adjust)
        self.fill = bool(fill)
        self.linewidth = float(linewidth)


class geom_violin(_Geom):
    """Violin plot of ``y`` by ``x`` (ggplot2 ``geom_violin``).

    Requires ``aes(x=, y=)``. Density is mirrored about each ``x`` category.
    2D only.
    """

    kind = "violin"

    def __init__(
        self,
        mapping=None,
        *,
        n=128,
        adjust=1.0,
        width=0.9,
        linewidth=1.0,
        **kw,
    ):
        super().__init__(mapping, **kw)
        self.n = int(n)
        self.adjust = float(adjust)
        self.width = float(width)
        self.linewidth = float(linewidth)


class facet_wrap:
    """Wrap panels by a discrete column (ggplot2 ``facet_wrap``).

    Parameters
    ----------
    facets:
        Column name, or a formula-like string ``"~col"`` / ``". ~ col"``.
    ncol, nrow:
        Panel grid size. If both omitted, ``ncol`` is chosen near ``sqrt(n)``.
    scales:
        ``"fixed"`` (shared domains across panels) or ``"free"`` (per-panel).
    """

    def __init__(
        self,
        facets: str,
        *,
        ncol: int | None = None,
        nrow: int | None = None,
        scales: str = "fixed",
    ):
        if not isinstance(facets, str) or not facets.strip():
            raise TypeError("facet_wrap() facets must be a column name string")
        name = facets.strip()
        if "~" in name:
            # Accept "~cyl", ". ~ cyl", "cyl ~ ."
            parts = [p.strip() for p in name.split("~")]
            candidates = [p for p in parts if p and p != "."]
            if len(candidates) != 1:
                raise ValueError(
                    "facet_wrap() currently accepts a single facet column "
                    f"(got {facets!r})"
                )
            name = candidates[0]
        if scales not in {"fixed", "free"}:
            raise ValueError("scales must be 'fixed' or 'free'")
        if ncol is not None and ncol < 1:
            raise ValueError("ncol must be positive")
        if nrow is not None and nrow < 1:
            raise ValueError("nrow must be positive")
        self.variable = name
        self.ncol = ncol
        self.nrow = nrow
        self.scales = scales


class labs(dict):
    def __init__(self, title=None, x=None, y=None, z=None, color=None,
                 colour=None):
        super().__init__()
        for k, v in (("title", title), ("x", x), ("y", y), ("z", z),
                     ("color", color if color is not None else colour)):
            if v is not None:
                self[k] = v


class scale_colour_continuous:
    """Numeric colour scale control.

    trans:   "linear" | "sqrt" | "log10"
    limits:  (lo, hi) tuple, or "full" for the data min/max.
             Default (no scale added) is robust 2nd-98th percentile limits —
             skewed data (lidar intensity!) stays readable; values outside
             the limits clamp to the ramp ends.
    palette: "blue" (theme single-hue default) | "viridis" | "magma" | "turbo"
    """

    def __init__(self, trans="linear", limits=None, palette="blue"):
        if trans not in ("linear", "sqrt", "log10"):
            raise ValueError("trans must be linear, sqrt or log10")
        if palette != "blue" and palette not in _CONT_PALETTES:
            raise ValueError(
                f"palette must be blue or one of {sorted(_CONT_PALETTES)}")
        self.trans = trans
        self.limits = limits
        self.palette = palette


scale_color_continuous = scale_colour_continuous


class scale_colour_viridis_c(scale_colour_continuous):
    """ggplot2-style viridis continuous scale: option viridis|magma|turbo."""

    def __init__(self, option="viridis", trans="linear", limits=None):
        super().__init__(trans=trans, limits=limits, palette=option)


scale_color_viridis_c = scale_colour_viridis_c


class _Theme:
    def __init__(self, name: str):
        self.name = name


def theme_dark() -> _Theme:
    return _Theme("dark")


def theme_light() -> _Theme:
    return _Theme("light")

