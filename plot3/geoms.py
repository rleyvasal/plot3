"""Grammar objects: aes, geoms, labs, colour scales, theme helpers."""

from __future__ import annotations

import math

import numpy as np

from plot3.themes import _CONT_PALETTES


def _as_column_name(value):
    """Coerce aesthetic values to column-name strings when possible.

    Jupyter R-style masking already turns bare names / backticks into strings.
    This is a small runtime safety net for escaped sentinels and objects that
    expose a column name (e.g. tidy3 ``col("x")``).

    Integer aesthetics (``0``, ``1``, …) are kept as decimal strings so NumPy
    array columns can be addressed with ``aes(x=0, y=1, z=2)``.
    """
    if value is None or isinstance(value, str):
        return value
    # Positional columns for ArrayTable / integer-named frames (not bool).
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    # tidy3 / polars expr sometimes use meta_output_name or similar
    for attr in ("meta_output_name", "column", "col_name"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, str) and out:
                    return out
            except Exception:
                pass
        elif isinstance(fn, str) and fn:
            return fn
    return value


class aes(dict):
    """Aesthetic mapping: aes(x=, y=, z=, colour=/color=, fill=, group=).

    ``fill`` is accepted as an alias of ``colour`` when colour is omitted
    (useful for surfaces).

    In Jupyter / SolveIt with R-style masking (default), bare names and
    backticks work like ggplot2::

        aes(x=wt, y=mpg, colour=cyl)
        aes(x=`First Name`, y=`Age (%)`)

    In plain ``.py`` files, use strings: ``aes(x="wt", y="mpg")``.

    For 2D NumPy arrays, use integer positions (stored as ``\"0\"``, ``\"1\"``, …)::

        ggplot(points, aes(x=0, y=1, z=2, colour=3)) + geom_point3d()
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
                self[k] = _as_column_name(v)


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
    ``size`` is scene units with distance attenuation (unit-cube space after
    encoding). Prefer :class:`geom_point3d` for explicit 3D intent.

    When ``size`` is omitted in 3D, a density-aware default is chosen
    (pcviz-like fine points on dense clouds).
    """

    kind = "point"

    def __init__(self, mapping=None, *, size=None, **kw):
        super().__init__(mapping, **kw)
        self.size = size


class geom_point3d(geom_point):
    """3D scatter / point-cloud marks (same ``kind`` as :class:`geom_point`).

    Use with ``aes(x=, y=, z=)``. Under default :class:`coord_3d`
    (``size_mode="scene"``), size is in **unit-cube scene units** with
    distance attenuation — comparable to pcviz's metric sizing after
    plot3 normalizes axes into a unit cube.

    Parameters
    ----------
    size:
        Point diameter in scene units. Default ``None`` picks a small,
        density-aware size (roughly pcviz ``size=0.06`` m on a ~50 m cloud).
        Override for artistic control, e.g. ``size=0.002``.
    """

    def __init__(self, mapping=None, *, size=None, **kw):
        # None → build_spec density-aware default (not a hard-coded 0.01 blob).
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


class coord_equal:
    """Lock 2D axis units so shapes are not stretched to the panel.

    ``ratio`` is the ggplot2 ``coord_fixed`` ratio: one unit on x has the
    same on-screen length as ``ratio`` units on y. ``coord_equal()`` is
    ``ratio=1``. The panel keeps its size; the camera shows extra range on
    the looser axis instead of stretching the data.

    Implicit-only figures (a circle, for example) use this automatically.
    """

    def __init__(self, ratio: float = 1.0):
        try:
            value = float(ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "coord_equal() ratio must be a positive number"
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("coord_equal() ratio must be a positive number")
        self.ratio = value

    def to_spec(self) -> dict:
        return {"aspect": "equal", "ratio": self.ratio}


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


class geom_function(_Geom):
    """Draw a formula or callable as a curve or surface.

    The quoted string is the form that works in scripts and notebooks::

        ggplot() + geom_function("y = 2x + 2")
        ggplot() + geom_function("z = sin(x) cos(y)", xlim=(-4, 4), ylim=(-4, 4))
        ggplot() + geom_function("y = a x^2 + b", a=1, b=-2)

    In a notebook, the same expression can be written without quotes
    (``geom_function(y = 2*x + 2)``). A callable is full Python::

        ggplot() + geom_function(lambda x: np.where(x < 0, 0, x**2))

    ``xlim`` / ``ylim`` / ``zlim`` are axis limits. On a curve, ``xlim`` is
    the domain and ``ylim`` clips the view. On a surface, ``xlim`` and
    ``ylim`` are the domain and ``zlim`` clips the view. ``n`` is the sample
    count (default 501 on a curve, 80 per axis on a surface or implicit curve).
    An implicit curve that covers only part of that window is resampled
    around the contour so the line stays smooth. A figure made only of
    implicit equations uses equal axis units, so a circle stays round.
    """

    kind = "function"

    def __init__(
        self,
        expr=None,
        mapping=None,
        *,
        y=None,
        z=None,
        f=None,
        xlim=None,
        ylim=None,
        zlim=None,
        n=None,
        linewidth=None,
        wireframe: bool = False,
        color=None,
        colour=None,
        alpha=None,
        **params,
    ):
        from plot3.expr import parse_formula

        super().__init__(
            mapping, color=color, colour=colour, alpha=alpha, **params
        )
        bound = dict(params)
        formula = expr
        lhs_name = None
        if formula is not None:
            if y is not None:
                bound["y"] = y
            if z is not None:
                bound["z"] = z
            if f is not None:
                bound["f"] = f
        elif y is not None:
            formula = y
            lhs_name = "y"
        elif z is not None:
            formula = z
            lhs_name = "z"
        elif f is not None:
            formula = f
            lhs_name = "f"
        # Notebook form ``geom_function(y = 2*x + 2)`` arrives as the keyword
        # value only. Put that name back on the left of the formula.
        if (
            lhs_name
            and isinstance(formula, str)
            and "=" not in formula
        ):
            formula = f"{lhs_name} = {formula}"
        if formula is None:
            raise ValueError(
                'geom_function() needs a formula, for example '
                'geom_function("y = 2x + 2")'
            )
        self.formula = parse_formula(formula, bound)
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.n = None if n is None else int(n)
        self.linewidth = None if linewidth is None else float(linewidth)
        self.wireframe = bool(wireframe)
        self.params = bound


class geom_col(_Geom):
    """Bars with heights from ``y`` (ggplot2 ``geom_col``).

    Requires ``aes(x=, y=)``. ``x`` may be categorical or numeric.

    Parameters
    ----------
    width:
        Bar width as a fraction of the x resolution (ggplot2 default
        ``0.9``). Data width is ``resolution(x) * width``. Override freely.
    """

    kind = "col"

    def __init__(self, mapping=None, *, width=0.9, **kw):
        super().__init__(mapping, **kw)
        self.width = float(width)
        self.data_override = None  # optional layer-local frame (stats)


class geom_bar(_Geom):
    """Count bars for a discrete ``x`` (ggplot2 ``geom_bar`` / ``stat_count``).

    Only ``aes(x=)`` is required; counts become ``y``. Expanded to
    ``geom_col`` at build time. Counted ``x`` values are drawn on a
    **discrete** scale (like ``factor(x)`` in ggplot2). 2D only.

    Parameters
    ----------
    width:
        Bar width as a fraction of category spacing (ggplot2 default
        ``0.9``). Use ``1.0`` for flush bars, smaller for more gap.
    """

    kind = "bar"

    def __init__(self, mapping=None, *, width=0.9, **kw):
        super().__init__(mapping, **kw)
        self.width = float(width)


class geom_histogram(_Geom):
    """Histogram of a continuous ``x`` (ggplot2 ``geom_histogram`` / ``stat_bin``).

    Only ``aes(x=)`` is required. Bins are computed in Python and drawn as
    ``geom_col`` with **full bin width** so adjacent bars touch (no gaps).
    2D only.

    Parameters
    ----------
    bins:
        Explicit number of bins. When omitted (default), binning is chosen
        from the data via ``method`` (Freedman–Diaconis by default). Ignored
        when ``binwidth`` is set. Pass ``bins=30`` to force a fixed count
        (ggplot2's historical default).
    binwidth:
        Absolute bin width in data units. When set, overrides ``bins`` and
        ``method``.
    method:
        Automatic rule used only when both ``bins`` and ``binwidth`` are
        omitted: ``"fd"`` (Freedman–Diaconis, default), ``"scott"``,
        ``"sturges"``, ``"auto"`` (numpy's multi-rule choice), or other
        names accepted by ``numpy.histogram_bin_edges``.
    boundary:
        Optional bin boundary (ggplot2 ``boundary``). Aligns edges so that
        one edge falls on this value (modulo ``binwidth``).
    closed:
        ``"right"`` (default) or ``"left"`` — which side of each bin is
        closed (matches numpy / ggplot2 closed intervals).
    """

    kind = "histogram"

    def __init__(
        self,
        mapping=None,
        *,
        bins: int | None = None,
        binwidth: float | None = None,
        method: str = "fd",
        boundary: float | None = None,
        closed: str = "right",
        **kw,
    ):
        super().__init__(mapping, **kw)
        if bins is not None and int(bins) < 1:
            raise ValueError("bins must be positive")
        if binwidth is not None and float(binwidth) <= 0:
            raise ValueError("binwidth must be positive")
        if closed not in {"right", "left"}:
            raise ValueError("closed must be 'right' or 'left'")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a non-empty string")
        self.bins = None if bins is None else int(bins)
        self.binwidth = None if binwidth is None else float(binwidth)
        self.method = method.strip().lower()
        self.boundary = None if boundary is None else float(boundary)
        self.closed = closed
        # Histograms use absolute bin width (bars touch); no relative width.


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

