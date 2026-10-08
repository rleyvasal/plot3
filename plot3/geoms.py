"""Grammar objects: aes, geoms, labs, colour scales, theme helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from plot3.themes import _CONT_PALETTES, _THEMES


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
    """Aesthetic mapping: aes(x=, y=, z=, colour=/color=, fill=, size=, group=,
    ymin=, ymax=).

    ``fill`` colours the inside of filled shapes (bars, boxes, violins,
    ribbons, densities, surfaces), as in ggplot2. Points, lines, and error
    bars use ``colour`` and ignore ``fill``, so ``geom_col(aes(fill=g))``
    with ``geom_errorbar()`` gives coloured bars and black error bars.

    ``size`` maps a numeric column onto point area (radius follows the square
    root), so a bubble chart reads population as area. A constant
    ``geom_point(size=)`` is still one size for the whole layer.

    ``xmin``/``xmax``/``xend``/``yend`` place rectangles and segments, and
    ``sample`` is the column a Q-Q plot compares with a distribution.
    ``ymin`` and ``ymax`` are the ends of an error bar or ribbon. ``label``
    is the text of ``geom_text``. ``shape`` picks point symbols and
    ``linetype`` dash patterns per group.

    ``group`` is the identity of an object. Lines use it to split series.
    ``transition_time`` uses it to match the same object across frames
    (a country, for example).

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
        size=None,
        group=None,
        ymin=None,
        ymax=None,
        label=None,
        shape=None,
        linetype=None,
        xmin=None,
        xmax=None,
        xend=None,
        yend=None,
        sample=None,
    ):
        super().__init__()
        colour_value = color if color is not None else colour
        for k, v in (("x", x), ("y", y), ("z", z),
                     ("color", colour_value),
                     ("fill", fill),
                     ("size", size),
                     ("group", group),
                     ("ymin", ymin),
                     ("ymax", ymax),
                     ("label", label),
                     ("shape", shape),
                     ("linetype", linetype),
                     ("xmin", xmin),
                     ("xmax", xmax),
                     ("xend", xend),
                     ("yend", yend),
                     ("sample", sample)):
            if v is not None:
                self[k] = _as_column_name(v)


class _Geom:
    kind = ""
    sort_x = False

    def __init__(self, mapping: aes | None = None, *, color=None, colour=None,
                 alpha=None, data=None, **params):
        self.mapping = mapping or aes()
        self.const_color = color if color is not None else colour
        self.alpha = alpha
        # A layer's own rows (ggplot2's geom_rect(data = periods, ...)).
        self.layer_data = data
        self.params = params


class geom_point(_Geom):
    """Scatter points.

    In **2D**, a constant ``size`` is pixels. In **3D** (when ``aes(z=...)``
    is set), a constant ``size`` is scene units with distance attenuation
    (unit-cube space after encoding). Prefer :class:`geom_point3d` for
    explicit 3D intent. ``aes(size=)`` maps a column to area instead.

    When ``size`` is omitted in 3D, a density-aware default is chosen
    (pcviz-like fine points on dense clouds).
    """

    kind = "point"

    def __init__(self, mapping=None, *, size=None, shape=None, **kw):
        super().__init__(mapping, **kw)
        self.size = size
        self.shape = None if shape is None else shape_name(shape)


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
        ``"auto"`` (default) keeps relative axis spans, except that a tall z
        is shortened to twice the wider horizontal side. ``"data"`` keeps
        true proportions always (lidar). ``"equal"`` forces a unit cube.
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
        aspect: str = "auto",
        size_mode: str = "scene",
        max_points: int | None = None,
    ):
        if aspect not in {"auto", "data", "equal"}:
            raise ValueError("aspect must be 'auto', 'data', or 'equal'")
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


class coord_polar:
    """Draw ``r = f(theta)`` with equal units on x and y.

    The angle runs from 0 to ``2π`` unless the layer sets ``tlim``.
    A cardioid is ``geom_function("r = 1 + cos(theta)") + coord_polar()``.
    """

    def to_spec(self) -> dict:
        return {"aspect": "equal", "ratio": 1.0}


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


class arrow:
    """An arrowhead for geom_segment, geom_path, geom_line, annotate("segment").

    ``angle`` in degrees, ``length`` in inches (ggplot2's 0.25), ``ends`` is
    "last", "first", or "both", and ``type`` is "open" or "closed" (filled).
    """

    def __init__(self, angle=30.0, length=0.25, ends="last", type="open"):  # noqa: A002
        if ends not in {"last", "first", "both"}:
            raise ValueError('arrow(ends=) is "last", "first", or "both"')
        if type not in {"open", "closed"}:
            raise ValueError('arrow(type=) is "open" or "closed"')
        self.angle = float(angle)
        self.length = float(length)
        self.ends = ends
        self.type = type

    def spec(self) -> dict:
        return {"angle": self.angle, "length": self.length * 96.0, "ends": self.ends, "type": self.type}


class geom_path(_Geom):
    kind = "line"
    sort_x = False  # ggplot2 geom_path: connect in data order

    def __init__(self, mapping=None, *, linewidth=None, width=None, linetype=None, arrow=None, **kw):
        super().__init__(mapping, **kw)
        self.arrow = arrow
        self.linewidth = linewidth if linewidth is not None else (width or 2.0)
        dash_pattern(linetype)
        self.linetype = linetype


class geom_line(geom_path):
    sort_x = True  # ggplot2 geom_line: connect in order of x


_MARK_NAMES = ("roots", "extrema", "intersections")


def _mark_tuple(mark) -> tuple[str, ...]:
    """``"roots"``, ``"roots, extrema"``, or a sequence of those names."""
    if mark is None:
        return ()
    if isinstance(mark, str):
        parts = [part.strip() for part in mark.split(",")]
        parts = [part for part in parts if part]
    elif isinstance(mark, (list, tuple)):
        parts = []
        for item in mark:
            if not isinstance(item, str):
                raise ValueError(
                    "mark must be 'roots', 'extrema', or 'intersections'"
                )
            text = item.strip()
            if text:
                parts.append(text)
    else:
        raise ValueError("mark must be 'roots', 'extrema', or 'intersections'")
    if not parts or any(part not in _MARK_NAMES for part in parts):
        raise ValueError("mark must be 'roots', 'extrema', or 'intersections'")
    ordered: list[str] = []
    for part in parts:
        if part not in ordered:
            ordered.append(part)
    return tuple(ordered)


def _assignment_text(name: str, value) -> str | None:
    """A formula string for this axis, or None when ``value`` is not text."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        raise ValueError(f"geom_function() {name}= is empty")
    if "=" not in text:
        text = f"{name} = {text}"
    return text


class geom_function(_Geom):
    """Draw a formula or callable as a curve or surface.

    The quoted string is the form that works in scripts and notebooks::

        ggplot() + geom_function("y = 2x + 2")
        ggplot() + geom_function("z = sin(x) cos(y)", xlim=(-4, 4), ylim=(-4, 4))
        ggplot() + geom_function("y = a x^2 + b", a=1, b=-2)
        ggplot() + geom_function("y = a x^2") + transition_time(a=(0, 3))

    A raw LaTeX string is accepted too. Without the ``r`` prefix Python
    eats the backslashes (``\\frac`` becomes a form feed) before plot3
    sees them::

        ggplot() + geom_function(r"y = \\frac{\\sin x}{x}")

    ``$...$`` or a backslash command selects LaTeX, and in that form
    ``xy`` means ``x`` times ``y``. Plain text is unchanged, and braces
    still group, so ``x^{2}`` works either way.

    In a notebook, the same expression can be written without quotes
    (``geom_function(y = 2*x + 2)``). A callable is full Python::

        ggplot() + geom_function(lambda x: np.where(x < 0, 0, x**2))

    The legend shows the formula typeset. A figure with one function puts
    that formula in the title instead. ``label`` replaces it. ``$...$`` in
    ``label`` or in ``labs()`` is math, and the rest of the string stays
    plain::

        geom_function("t = 2x^3 + 3y^3", label=r"Cubic: $t = 2x^3 + 3y^3$")

    ``xlim`` / ``ylim`` / ``zlim`` are axis limits. On a curve, ``xlim`` is
    the domain and ``ylim`` clips the view. On a surface, ``xlim`` and
    ``ylim`` are the domain and ``zlim`` clips the view. ``n`` is the sample
    count (default 501 on a curve, 80 per axis on a surface or implicit curve).
    An implicit curve then subdivides the cells it crosses, so the line stays
    smooth. A figure made only of implicit equations uses equal axis units,
    so a circle stays round. A formula surface uses equal aspect (a cube)
    unless you pass ``coord_3d``.

    A parametric curve assigns two or three of x, y, and z, either in one
    string or as keywords. ``tlim`` is the parameter interval (otherwise
    ``xlim``, otherwise −10 to 10)::

        geom_function("x = cos(t), y = sin(t)")
        geom_function(x="cos(t)", y="sin(t)", z="t")

    ``mark`` is ``"roots"``, ``"extrema"``, ``"intersections"``, or a
    combination. ``area(0, 2)``, ``tangent(at=1)``, and ``derivative()``
    attach to the curve that came before them. ``"y > x^2"`` shades the
    side where the inequality holds. ``where(x < 0, 0, x^2)`` and a LaTeX
    ``cases`` environment are piecewise.
    """

    kind = "function"

    def __init__(
        self,
        expr=None,
        mapping=None,
        *,
        x=None,
        y=None,
        z=None,
        f=None,
        xlim=None,
        ylim=None,
        zlim=None,
        tlim=None,
        n=None,
        linewidth=None,
        wireframe: bool = False,
        color=None,
        colour=None,
        alpha=None,
        label=None,
        mark=None,
        **params,
    ):
        from plot3.expr import parse_formula

        super().__init__(
            mapping, color=color, colour=colour, alpha=alpha, **params
        )
        bound = dict(params)
        pieces: list[str] = []
        for name, value in (("x", x), ("y", y), ("z", z), ("f", f)):
            piece = _assignment_text(name, value)
            if piece is not None:
                pieces.append(piece)
            elif value is not None and expr is not None:
                if isinstance(value, str):
                    raise ValueError(
                        "geom_function() takes one formula. "
                        'Use geom_function(x="cos(t)", y="sin(t)") '
                        "with no positional formula, or pass numbers such as a=1."
                    )
                bound[name] = value
            elif value is not None and not isinstance(value, str):
                bound[name] = value
        if expr is not None and pieces:
            raise ValueError(
                "geom_function() takes one formula. "
                'Use geom_function(x="cos(t)", y="sin(t)") '
                "with no positional formula, or pass numbers such as a=1."
            )
        if expr is not None:
            formula = expr
        elif len(pieces) >= 2:
            formula = ", ".join(pieces)
        elif len(pieces) == 1:
            formula = pieces[0]
        elif callable(f):
            formula = f
        else:
            raise ValueError(
                'geom_function() needs a formula, for example '
                'geom_function("y = 2x + 2")'
            )
        # Defer unbound coefficients (``a`` in ``y = a x^2``). The transition
        # is added with ``+`` afterwards, so it does not exist yet.
        self.formula = parse_formula(formula, bound, defer_missing=True)
        # Kept so a slider or transition that names a free symbol (t, mu)
        # can re-read it as a coefficient at build time.
        self._source = formula
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.tlim = tlim
        self.n = None if n is None else int(n)
        self.linewidth = None if linewidth is None else float(linewidth)
        self.wireframe = bool(wireframe)
        self.label = None if label is None else str(label)
        self.marks = _mark_tuple(mark)
        self.params = bound


class area:
    """Shade ``y = f(x)`` from ``lo`` to ``hi`` and label the integral.

    Add it after the curve. Limits swap when ``hi < lo``. ``baseline``
    is the lower edge (default 0)::

        ggplot() + geom_function("y = x^2") + area(0, 2)

    Under a density the label is a probability, and a limit may be
    infinite (it stops at the edge of the curve)::

        geom_function("y = dnorm(x)") + area(-inf, -1.96) + area(1.96, inf)
    """

    def __init__(self, lo, hi, *, baseline=0):
        try:
            left = float(lo)
            right = float(hi)
            base = float(baseline)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "area() needs numeric limits, for example area(0, 2)"
            ) from exc
        if any(math.isnan(value) for value in (left, right)) or not math.isfinite(base):
            raise ValueError(
                "area() needs numeric limits, for example area(0, 2)"
            )
        if left == right:
            raise ValueError("area() needs two different limits")
        if right < left:
            left, right = right, left
        self.lo = left
        self.hi = right
        self.baseline = base


class tangent:
    """Tangent line to the preceding ``geom_function`` curve at ``x = at``."""

    def __init__(self, at):
        try:
            value = float(at)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "tangent() needs a number, for example tangent(at=1)"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                "tangent() needs a number, for example tangent(at=1)"
            )
        self.at = value


class derivative:
    """Draw ``f'`` of the preceding ``geom_function`` curve."""

    def __init__(self):
        return None


def _field_piece(name: str, value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"geom_vector_field() {name}= is empty")
        if "=" not in text:
            text = f"{name} = {text}"
        return text
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"geom_vector_field() {name}= must be a formula or a number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"geom_vector_field() {name}= must be a formula or a number"
        )
    return f"{name} = {number}"


class geom_vector_field(_Geom):
    """Arrows or streamlines for a planar field.

    ``dx`` and ``dy`` are formulas in ``x`` and ``y``. The default window
    is −2 to 2 with an 11 by 11 grid, so a rotation field stays readable.
    ``stream=True`` draws unit-speed streamlines instead of arrows::

        ggplot() + geom_vector_field("dx = -y, dy = x")
    """

    kind = "vector"

    def __init__(
        self,
        expr=None,
        mapping=None,
        *,
        dx=None,
        dy=None,
        dz=None,
        xlim=(-2, 2),
        ylim=(-2, 2),
        zlim=None,
        n=11,
        stream: bool = False,
        linewidth=None,
        color=None,
        colour=None,
        alpha=None,
        label=None,
        **params,
    ):
        from plot3.expr import parse_formula

        super().__init__(mapping, color=color, colour=colour, alpha=alpha)
        if expr is not None and any(value is not None for value in (dx, dy, dz)):
            raise ValueError(
                'geom_vector_field() takes one formula, for example '
                'geom_vector_field("dx = -y, dy = x")'
            )
        if expr is not None:
            formula = expr
        else:
            parts = [
                piece
                for piece in (
                    _field_piece("dx", dx),
                    _field_piece("dy", dy),
                    _field_piece("dz", dz),
                )
                if piece is not None
            ]
            if len(parts) < 2:
                raise ValueError(
                    'geom_vector_field() needs dx and dy, for example '
                    '"dx = -y, dy = x"'
                )
            formula = ", ".join(parts)
        self.formula = parse_formula(
            formula, dict(params), defer_missing=True, role="field"
        )
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.n = 11 if n is None else int(n)
        self.stream = bool(stream)
        self.linewidth = None if linewidth is None else float(linewidth)
        self.label = None if label is None else str(label)
        self.params = dict(params)


class geom_col(_Geom):
    """Bars with heights from ``y`` (ggplot2 ``geom_col``).

    Requires ``aes(x=, y=)``. ``x`` may be categorical or numeric.

    Parameters
    ----------
    width:
        Bar width as a fraction of the x resolution (ggplot2 default
        ``0.9``). Data width is ``resolution(x) * width``. Override freely.
    position:
        With ``aes(colour=)`` groups: ``"stack"`` (default), ``"dodge"``
        (side by side), ``"fill"`` (stacked to proportions), or
        ``"identity"`` (overlapping). ``position_dodge(width=)`` also works.
    """

    kind = "col"

    def __init__(self, mapping=None, *, width=0.9, position="stack", **kw):
        super().__init__(mapping, **kw)
        self.width = float(width)
        self.position = position
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
    position:
        With ``aes(colour=)`` groups the counts split by group:
        ``"stack"`` (default), ``"dodge"``, ``"fill"``, or ``"identity"``.
    """

    kind = "bar"

    def __init__(self, mapping=None, *, width=0.9, position="stack", **kw):
        super().__init__(mapping, **kw)
        self.width = float(width)
        self.position = position


class position_dodge:
    """Side by side within each x. ``width`` is the slot, as a fraction of x spacing."""

    kind = "dodge"

    def __init__(self, width=None):
        self.width = None if width is None else float(width)


class position_stack:
    """Stacked, first group on top (ggplot2 order)."""

    kind = "stack"
    width = None


class position_fill:
    """Stacked and scaled so each x sums to 1."""

    kind = "fill"
    width = None


class geom_jitter(_Geom):
    """Points nudged at random so overplotted values separate (ggplot2 ``geom_jitter``).

    ``width`` and ``height`` default to 40% of the spacing between x and y
    values, as in ggplot2. Pass ``height=0`` to keep y exact. ``seed`` makes
    the jitter repeatable, so a saved figure does not change between runs.
    """

    kind = "jitter"

    def __init__(self, mapping=None, *, width=None, height=None, seed=0, size=None, **kw):
        super().__init__(mapping, **kw)
        self.width = width
        self.height = height
        self.seed = seed
        self.size = size


class geom_errorbar(_Geom):
    """Vertical error bars from ``ymin`` to ``ymax`` with caps.

    Requires ``aes(x=, ymin=, ymax=)``. ``width`` is the cap width as a
    fraction of x spacing. ``position="dodge"`` lines up with dodged bars.
    """

    kind = "errorbar"

    def __init__(self, mapping=None, *, width=0.5, linewidth=1.0, position="identity", **kw):
        super().__init__(mapping, **kw)
        self.width = float(width)
        self.linewidth = float(linewidth)
        self.position = position


class geom_linerange(_Geom):
    """A vertical line from ``ymin`` to ``ymax``. Requires ``aes(x=, ymin=, ymax=)``."""

    kind = "linerange"

    def __init__(self, mapping=None, *, linewidth=1.0, position="identity", **kw):
        super().__init__(mapping, **kw)
        self.linewidth = float(linewidth)
        self.position = position


class geom_pointrange(_Geom):
    """A point at ``y`` on a line from ``ymin`` to ``ymax``.

    Requires ``aes(x=, y=, ymin=, ymax=)``.
    """

    kind = "pointrange"

    def __init__(self, mapping=None, *, size=None, linewidth=1.0, position="identity", **kw):
        super().__init__(mapping, **kw)
        self.size = size
        self.linewidth = float(linewidth)
        self.position = position


class geom_ribbon(_Geom):
    """A band between ``ymin`` and ``ymax`` along x (confidence bands, ranges).

    Requires ``aes(x=, ymin=, ymax=)``; ``aes(colour=)`` draws one band per group.
    """

    kind = "ribbon"


class geom_smooth(_Geom):
    """A fitted trend with its confidence band (ggplot2 ``geom_smooth``).

    ``method="loess"`` (default, local quadratic with ``span``) or ``"lm"``
    (a straight line). ``se=True`` shades the ``level`` confidence band,
    using Student's t like R. ``aes(colour=)`` fits each group.
    """

    kind = "smooth"

    def __init__(
        self, mapping=None, *, method="loess", se=True, level=0.95,
        span=0.75, n=80, linewidth=2.0, **kw,
    ):
        super().__init__(mapping, **kw)
        self.method = method
        self.se = bool(se)
        self.level = float(level)
        self.span = float(span)
        self.n = int(n)
        self.linewidth = float(linewidth)


class stat_summary(_Geom):
    """Summarise ``y`` at each x, then draw it (ggplot2 ``stat_summary``).

    ``fun_data`` is ``"mean_se"`` (default), ``"mean_cl_normal"`` (t
    interval), ``"mean_sdl"`` (mean ± 2 SD), ``"median_hilow"`` (median and
    the middle 95%), or a function returning ``(y, ymin, ymax)``.
    ``fun_args`` passes options, e.g. ``{"mult": 1}``. ``geom`` is
    ``"pointrange"`` (default), ``"errorbar"``, ``"linerange"``, ``"col"``,
    or ``"point"``.
    """

    kind = "summary"

    def __init__(
        self, mapping=None, *, fun_data="mean_se", fun_args=None, geom="pointrange",
        width=None, linewidth=1.0, size=None, position=None, **kw,
    ):
        super().__init__(mapping, **kw)
        self.fun_data = fun_data
        self.fun_args = dict(fun_args or {})
        self.geom = geom
        self.width = (0.9 if geom in {"col", "bar"} else 0.5) if width is None else float(width)
        self.linewidth = float(linewidth)
        self.size = size
        self.position = position


# ggplot2's default shape palette, in order: what aes(shape=) assigns.
SHAPE_ORDER = ["circle", "triangle", "square", "diamond", "plus", "cross"]
_SHAPE_NUMBERS = {
    0: "square", 1: "circle", 2: "triangle", 3: "plus", 4: "cross", 5: "diamond",
    15: "square", 16: "circle", 17: "triangle", 18: "diamond", 19: "circle",
    20: "circle", 21: "circle", 22: "square", 23: "diamond", 24: "triangle",
}
LINETYPE_ORDER = ["solid", "dashed", "dotted", "dotdash", "longdash", "twodash"]


def shape_name(shape) -> str:
    """``"triangle"`` or R's numbers (17 = filled triangle) to a shape name."""
    if isinstance(shape, (int, float)) and not isinstance(shape, bool):
        if int(shape) in _SHAPE_NUMBERS:
            return _SHAPE_NUMBERS[int(shape)]
    name = str(shape).strip().lower()
    if name in SHAPE_ORDER:
        return name
    raise ValueError(f"shape {shape!r} is not one of {SHAPE_ORDER} or an R shape number")


# ggplot2 line types as dash patterns, in multiples of the line width.
LINETYPES = {
    "solid": None,
    "dashed": (4.0, 4.0),
    "dotted": (1.0, 3.0),
    "dotdash": (1.0, 3.0, 4.0, 3.0),
    "longdash": (8.0, 4.0),
    "twodash": (2.0, 2.0, 6.0, 2.0),
}
_LINETYPE_NUMBERS = ["blank", "solid", "dashed", "dotted", "dotdash", "longdash", "twodash"]


def dash_pattern(linetype) -> tuple[float, ...] | None:
    """``"dashed"`` -> (4, 4). Also R's numbers (2 = dashed) and hex strings ("44")."""
    if linetype is None:
        return None
    if isinstance(linetype, (int, float)) and not isinstance(linetype, bool):
        index = int(linetype)
        if 0 <= index < len(_LINETYPE_NUMBERS):
            linetype = _LINETYPE_NUMBERS[index]
    name = str(linetype).strip().lower()
    if name in LINETYPES:
        return LINETYPES[name]
    if name and len(name) % 2 == 0 and all(c in "0123456789abcdef" for c in name):
        return tuple(float(int(c, 16)) for c in name)
    raise ValueError(
        f"linetype {linetype!r} is not one of {sorted(LINETYPES)} "
        "or a hex pattern such as '44'"
    )


class geom_hline(_Geom):
    """Horizontal reference line(s) across the panel: ``geom_hline(yintercept=0)``.

    ``yintercept`` may be a list. ``linetype`` is ``"solid"``, ``"dashed"``,
    ``"dotted"``, ``"dotdash"``, ``"longdash"``, or ``"twodash"``.
    """

    kind = "hline"

    def __init__(self, mapping=None, *, yintercept, linetype="solid", linewidth=1.0, **kw):
        super().__init__(mapping, **kw)
        self.values = _as_list(yintercept, "yintercept")
        self.linetype = linetype
        dash_pattern(linetype)
        self.linewidth = float(linewidth)


class geom_vline(_Geom):
    """Vertical reference line(s): ``geom_vline(xintercept=[1, 2], linetype="dashed")``."""

    kind = "vline"

    def __init__(self, mapping=None, *, xintercept, linetype="solid", linewidth=1.0, **kw):
        super().__init__(mapping, **kw)
        self.values = _as_list(xintercept, "xintercept")
        self.linetype = linetype
        dash_pattern(linetype)
        self.linewidth = float(linewidth)


class geom_abline(_Geom):
    """The line ``y = intercept + slope * x`` across the panel (default ``y = x``)."""

    kind = "abline"

    def __init__(self, mapping=None, *, slope=1.0, intercept=0.0, linetype="solid", linewidth=1.0, **kw):
        super().__init__(mapping, **kw)
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.linetype = linetype
        dash_pattern(linetype)
        self.linewidth = float(linewidth)


def _as_list(value, name: str) -> list:
    if value is None:
        raise ValueError(f"{name}= is required")
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    return [value]


class geom_text(_Geom):
    """Text at each row: ``geom_text(aes(label=name))`` (ggplot2 ``geom_text``).

    ``size`` is in millimetres like ggplot2 (default 3.88, about 11 pt).
    ``hjust``/``vjust`` are 0 (left/bottom) to 1 (right/top). ``nudge_x`` and
    ``nudge_y`` shift labels off their points. ``check_overlap=True`` skips a
    label that would overlap one already drawn. ``fontface`` is ``"plain"``,
    ``"bold"``, ``"italic"``, or ``"bold.italic"``.
    """

    kind = "text"
    _box = False

    def __init__(
        self, mapping=None, *, size=3.88, hjust=0.5, vjust=0.5, nudge_x=0.0,
        nudge_y=0.0, check_overlap=False, fontface="plain", **kw,
    ):
        super().__init__(mapping, **kw)
        self.size = float(size)
        self.hjust = float(hjust)
        self.vjust = float(vjust)
        self.nudge_x = float(nudge_x)
        self.nudge_y = float(nudge_y)
        self.check_overlap = bool(check_overlap)
        if fontface not in {"plain", "bold", "italic", "bold.italic"}:
            raise ValueError("fontface is 'plain', 'bold', 'italic', or 'bold.italic'")
        self.fontface = fontface


class geom_label(geom_text):
    """Like ``geom_text`` with a box behind each label (ggplot2 ``geom_label``)."""

    kind = "text"
    _box = True


def annotate(geom: str, *, x=None, y=None, xmin=None, xmax=None, ymin=None,
             ymax=None, xend=None, yend=None, label=None, **params):
    """One-off marks in data coordinates (ggplot2 ``annotate``).

    * ``annotate("text", x=2, y=5, label="peak")`` (also ``"label"``)
    * ``annotate("rect", xmin=1, xmax=2, ymin=0, ymax=10, alpha=0.2)``
    * ``annotate("segment", x=1, y=1, xend=2, yend=3)``
    * ``annotate("point", x=1, y=1, size=8)``

    Values may be lists for several marks. Text, segments, and points
    default to the ink colour; a rectangle to translucent grey.
    """
    def seq(value):
        if value is None:
            return None
        return list(value) if isinstance(value, (list, tuple, np.ndarray, pd.Series)) else [value]

    def frame(**cols):
        given = {k: seq(v) for k, v in cols.items()}
        size = max(len(v) for v in given.values())
        return pd.DataFrame({k: v * size if len(v) == 1 else v for k, v in given.items()})

    colour = params.pop("colour", params.pop("color", None))
    fill = params.pop("fill", None)
    kind = str(geom).lower()
    if kind in {"text", "label"}:
        if x is None or y is None or label is None:
            raise ValueError(f'annotate("{kind}") needs x=, y=, and label=')
        cls = geom_label if kind == "label" else geom_text
        out = cls(aes(x="x", y="y", label="label"), colour=colour, **params)
        out.data_override = frame(x=x, y=y, label=label)
    elif kind == "rect":
        if None in (xmin, xmax, ymin, ymax):
            raise ValueError('annotate("rect") needs xmin=, xmax=, ymin=, ymax=')
        box = frame(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        xs, ys, groups = [], [], []
        for row in box.itertuples(index=False):
            groups.append([len(xs), 4])
            xs += [row.xmin, row.xmin, row.xmax, row.xmax]
            ys += [row.ymin, row.ymax, row.ymax, row.ymin]
        out = _Geom(aes(x="x", y="y"), colour=fill or colour or "#7f7f7f",
                    alpha=params.pop("alpha", 0.2))
        out.kind = "poly"
        out.data_override = pd.DataFrame({"x": xs, "y": ys})
        out._groups = groups
        out.linewidth = 0.0
    elif kind == "segment":
        if None in (x, y, xend, yend):
            raise ValueError('annotate("segment") needs x=, y=, xend=, yend=')
        seg = frame(x=x, y=y, xend=xend, yend=yend)
        xs, ys, groups = [], [], []
        for row in seg.itertuples(index=False):
            groups.append([len(xs), 2])
            xs += [row.x, row.xend]
            ys += [row.y, row.yend]
        out = geom_path(aes(x="x", y="y"), colour=colour, **params)
        out.data_override = pd.DataFrame({"x": xs, "y": ys})
        out._groups = groups
        out._ink_default = colour is None
    elif kind == "point":
        if x is None or y is None:
            raise ValueError('annotate("point") needs x= and y=')
        out = geom_point(aes(x="x", y="y"), colour=colour, **params)
        out.data_override = frame(x=x, y=y)
        out._ink_default = colour is None
    else:
        raise ValueError('annotate() draws "text", "label", "rect", "segment", or "point"')
    out._replace_mapping = True
    out._annotation = True
    return out


class geom_tile(_Geom):
    """Heatmap cells: a rectangle at each (x, y), coloured by ``fill``.

    x and y may be categories or numbers; ``width``/``height`` default to
    the spacing of the values. ``geom_raster`` is the same.
    """

    kind = "tile"

    def __init__(self, mapping=None, *, width=None, height=None, **kw):
        super().__init__(mapping, **kw)
        self.width = width
        self.height = height


geom_raster = geom_tile


class geom_area(_Geom):
    """Filled area under ``y``; groups (``fill``) stack, first level on top.

    ``position="stack"`` (default), ``"fill"`` (shares of 1), or ``"identity"``.
    """

    kind = "area_stat"

    def __init__(self, mapping=None, *, position="stack", **kw):
        super().__init__(mapping, **kw)
        self.position = position


class geom_step(_Geom):
    """A staircase line: ``direction="hv"`` (default), ``"vh"``, or ``"mid"``."""

    kind = "step"

    def __init__(self, mapping=None, *, direction="hv", linewidth=2.0, linetype=None, **kw):
        super().__init__(mapping, **kw)
        self.direction = direction
        self.linewidth = float(linewidth)
        dash_pattern(linetype)
        self.linetype = linetype


class geom_segment(_Geom):
    """A line from (x, y) to (xend, yend) for every row."""

    kind = "segment"

    def __init__(self, mapping=None, *, linewidth=1.0, linetype=None, arrow=None, **kw):
        super().__init__(mapping, **kw)
        self.arrow = arrow
        self.linewidth = float(linewidth)
        dash_pattern(linetype)
        self.linetype = linetype


class geom_rect(_Geom):
    """A rectangle from xmin..xmax and ymin..ymax for every row (shaded periods)."""

    kind = "rect"


class geom_qq(_Geom):
    """Q-Q plot: sorted ``aes(sample=)`` against normal quantiles."""

    kind = "qq"

    def __init__(self, mapping=None, *, size=None, **kw):
        super().__init__(mapping, **kw)
        self.size = size


class geom_qq_line(_Geom):
    """The reference line of a Q-Q plot, through the quartiles (as in R)."""

    kind = "qq_line"

    def __init__(self, mapping=None, *, linewidth=1.5, linetype=None, **kw):
        super().__init__(mapping, **kw)
        self.linewidth = float(linewidth)
        dash_pattern(linetype)
        self.linetype = linetype


stat_qq = geom_qq
stat_qq_line = geom_qq_line


class stat_ecdf(_Geom):
    """Empirical cumulative distribution of ``x`` as a step line, per group."""

    kind = "ecdf"

    def __init__(self, mapping=None, *, pad=True, linewidth=2.0, **kw):
        super().__init__(mapping, **kw)
        self.pad = bool(pad)
        self.linewidth = float(linewidth)


class coord_flip:
    """Swap x and y: horizontal bars and boxplots, long category names on y."""

    kind = "flip"


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
        position="stack",
        **kw,
    ):
        super().__init__(mapping, **kw)
        # With aes(fill=g): "stack" (default), "dodge", "fill", "identity".
        self.position = position
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


class facet_grid:
    """Panels in a grid: one row per level of ``rows``, one column per level
    of ``cols`` (ggplot2 ``facet_grid``).

    ``facet_grid(rows="sex", cols="day")`` or the formula ``"sex ~ day"``
    (``". ~ day"`` for columns only). Column labels sit above the top row and
    row labels to the right, as in ggplot2. ``scales="fixed"`` (default)
    shares axes across panels; ``"free"`` lets each panel fit its data.
    """

    def __init__(self, facets: str | None = None, *, rows=None, cols=None, scales: str = "fixed"):
        if facets is not None:
            if not isinstance(facets, str) or "~" not in facets:
                raise ValueError('facet_grid() takes "rows ~ cols", or rows= and cols=')
            left, right = (part.strip() for part in facets.split("~", 1))
            rows = rows if rows is not None else (left if left not in {"", "."} else None)
            cols = cols if cols is not None else (right if right not in {"", "."} else None)
        rows = _as_column_name(rows) if rows is not None else None
        cols = _as_column_name(cols) if cols is not None else None
        if rows is None and cols is None:
            raise ValueError("facet_grid() needs rows=, cols=, or both")
        if scales not in {"fixed", "free"}:
            raise ValueError("scales must be 'fixed' or 'free'")
        self.rows = rows
        self.cols = cols
        self.scales = scales


class labs(dict):
    """Titles. ``colour`` (or ``fill``) names the colour legend.

    ``subtitle`` sits under the title, ``caption`` at the bottom right (a
    data source, say), and ``tag`` at the top left ("A", "B" for the panels
    of a figure).
    """

    def __init__(self, title=None, x=None, y=None, z=None, color=None,
                 colour=None, size=None, fill=None, subtitle=None,
                 caption=None, tag=None):
        super().__init__()
        legend = color if color is not None else colour if colour is not None else fill
        for k, v in (("title", title), ("x", x), ("y", y), ("z", z),
                     ("color", legend),
                     ("size", size),
                     ("subtitle", subtitle),
                     ("caption", caption),
                     ("tag", tag)):
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


class scale_x_log10:
    """Base-10 logarithmic scale for x.

    Positions are encoded in log10 space, so a decade is a constant distance
    and a tween along the axis moves in log space. Non-positive values are
    omitted. Tick labels stay in the original units (1, 10, 100, …).
    """

    axis = "x"


class scale_y_log10:
    """Base-10 logarithmic scale for y. See :class:`scale_x_log10`."""

    axis = "y"


def _as_range(value, name: str, who: str = "transition_time") -> tuple[float, float]:
    """A finite ``(lo, hi)`` pair. Inverted pairs are swapped."""
    pair = None
    if isinstance(value, (tuple, list, np.ndarray)) and not isinstance(
        value, (str, bytes)
    ):
        try:
            if len(value) == 2:
                pair = value
        except TypeError:
            pair = None
    if pair is not None:
        try:
            lo = float(pair[0])
            hi = float(pair[1])
        except (TypeError, ValueError):
            lo = hi = float("nan")
        else:
            if math.isfinite(lo) and math.isfinite(hi):
                if hi < lo:
                    lo, hi = hi, lo
                return (lo, hi)
    raise TypeError(
        f"{who}() range {name} must be a pair of numbers, "
        f"for example {name}=(0, 3)"
    )


class transition_time:
    """Animate points over a column, or a formula over parameter ranges.

    Column form (Gapminder). Each ``aes(group=)`` value is one object.
    Rows are the keyframes of that object. The viewer stores every
    keyframe and interpolates in the browser, in scale space, with the
    axes held fixed on the range of every frame. The clock runs linearly
    from the first time value to the last, so a ten-year gap takes ten
    times as long as a one-year gap.

    Range form. ``transition_time(a=(0, 3))`` sweeps every unbound
    coefficient of a ``geom_function`` together across ``frames`` steps
    (default 60). ``{frame_time}`` shows the parameters (``a = 1.50``).

        (
            ggplot(df, aes(x="gdp", y="life", size="pop", colour="continent",
                           group="country"))
            + geom_point()
            + scale_x_log10()
            + transition_time("year")
            + labs(title="{frame_time}")
        )

        ggplot() + geom_function("y = a x^2") + transition_time(a=(0, 3))
    """

    kind = "time"

    def __init__(self, column=None, *, frames: int = 60, **ranges):
        if (column is None) == (not ranges):
            raise TypeError(
                "transition_time() takes a column (transition_time('year')) "
                "or parameter ranges (transition_time(a=(0, 3)))"
            )
        if column is None:
            self.column = None
        else:
            name = _as_column_name(column)
            if not isinstance(name, str) or not name:
                raise TypeError("transition_time() needs a column name")
            self.column = name
        self.ranges = {k: _as_range(v, k) for k, v in ranges.items()}
        nframes = int(frames)
        if self.ranges and nframes < 2:
            raise ValueError("transition_time() needs at least 2 frames")
        self.frames = nframes


class slider:
    """Drag formula coefficients, one control per parameter.

    Unlike :class:`transition_time`, each range moves on its own and nothing
    plays by itself. The grid is sampled in Python (``steps`` values along
    each range, last keyword varying fastest) and the viewer blends curves
    and surfaces between the neighboring cells. An implicit contour snaps
    to the nearest cell, because its vertex count changes.

        ggplot() + geom_function("y = a sin(k x)") + slider(a=(0, 3), k=(1, 5))

    ``steps`` defaults to 25 per parameter. A surface on the default grid
    is too large for several parameters at that count; pass a smaller
    ``steps`` or ``n``. Do not combine with ``transition_time()``.
    """

    kind = "slider"

    def __init__(self, *, steps: int = 25, **ranges):
        if not ranges:
            raise TypeError(
                "slider() needs a parameter range, for example slider(a=(0, 3))"
            )
        try:
            nsteps = int(steps)
        except (TypeError, ValueError):
            raise TypeError(
                "slider() steps= must be an integer, for example steps=25"
            ) from None
        if nsteps < 2:
            raise ValueError("slider() needs at least 2 steps")
        self.steps = nsteps
        self.ranges = {k: _as_range(v, k, "slider") for k, v in ranges.items()}


class transition_states:
    """Animate a point layer over a discrete column.

    Frames follow the order the states first appear in the data (so
    ``before`` then ``after`` stays in that order). Objects ease from one
    state to the next. ``aes(group=)`` matches the same object across states.
    ``{frame_time}`` shows the current state label.
    """

    kind = "states"

    def __init__(self, column):
        name = _as_column_name(column)
        if not isinstance(name, str) or not name:
            raise TypeError("transition_states() needs a column name")
        self.column = name


class _Theme:
    """A named theme plus the font used by ``ggsave``.

    ``base_size`` is in points, as in ggplot2. ``base_family`` is a CSS
    font-family list. Both apply when the figure is saved to PNG, SVG, or
    PDF. The interactive viewer keeps its own type.
    """

    def __init__(
        self,
        name: str,
        *,
        base_size: float | None = None,
        base_family: str | None = None,
    ):
        if name not in _THEMES:
            known = ", ".join(sorted(_THEMES))
            raise ValueError(f"unknown theme {name!r}. Known themes: {known}")
        self.name = name
        self.base_size = _theme_points(base_size)
        self.base_family = None if base_family in (None, "") else str(base_family)


def _theme_points(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("base_size must be a font size in points, for example base_size=11")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "base_size must be a font size in points, for example base_size=11"
        ) from exc
    if not math.isfinite(number) or number < 1 or number > 96:
        raise ValueError("base_size must be between 1 and 96 points")
    return number


def _theme(name: str, base_size=None, base_family=None) -> _Theme:
    return _Theme(name, base_size=base_size, base_family=base_family)


def theme_dark(base_size=None, base_family=None) -> _Theme:
    return _theme("dark", base_size, base_family)


def theme_light(base_size=None, base_family=None) -> _Theme:
    return _theme("light", base_size, base_family)


def theme_bw(base_size=None, base_family=None) -> _Theme:
    """White page, grey grid, and a dark panel border."""
    return _theme("bw", base_size, base_family)


def theme_classic(base_size=None, base_family=None) -> _Theme:
    """White page, no grid, and black axis lines."""
    return _theme("classic", base_size, base_family)


def theme_minimal(base_size=None, base_family=None) -> _Theme:
    """White page, light grid, and no panel box."""
    return _theme("minimal", base_size, base_family)


class _ThemePatch:
    """Theme settings that do not change the colour theme."""

    def __init__(self, legend_position=None, **options):
        self.legend_position = legend_position
        self.options = options


def theme(
    *,
    legend_position=None,
    legend_title=None,
    panel_grid=None,
    axis_text_x_angle=None,
    plot_title_hjust=None,
    base_size=None,
    base_family=None,
) -> _ThemePatch:
    """Change parts of the theme; later ``theme()`` calls add to earlier ones.

    ``legend_position``: ``"right"``, ``"bottom"``, ``"none"``, or a pair
    ``(x, y)`` in 0–1 panel coordinates (inside the panel).
    ``legend_title``: ``False`` hides the legend title.
    ``panel_grid``: ``False`` removes the grid lines.
    ``axis_text_x_angle``: ``45`` or ``90`` turns long x labels.
    ``plot_title_hjust``: ``0`` left (default), ``0.5`` centred, ``1`` right.
    ``base_size`` (points) and ``base_family`` set the type for saved files.
    """
    options = {}
    if legend_title is not None:
        options["legend_title"] = bool(legend_title)
    if panel_grid is not None:
        options["panel_grid"] = bool(panel_grid)
    if axis_text_x_angle is not None:
        angle = float(axis_text_x_angle)
        if not 0.0 <= angle <= 90.0:
            raise ValueError("axis_text_x_angle is from 0 to 90 degrees")
        options["axis_text_x_angle"] = angle
    if plot_title_hjust is not None:
        hjust = float(plot_title_hjust)
        if not 0.0 <= hjust <= 1.0:
            raise ValueError("plot_title_hjust is from 0 (left) to 1 (right)")
        options["plot_title_hjust"] = hjust
    if base_size is not None:
        options["base_size"] = float(base_size)
    if base_family is not None:
        options["base_family"] = str(base_family)
    return _ThemePatch(_check_legend_position(legend_position), **options)


def _check_legend_position(value):
    if value is None:
        return None
    if isinstance(value, str):
        name = value.strip().lower()
        if name not in {"right", "bottom", "none"}:
            raise ValueError(
                "legend_position must be 'right', 'bottom', 'none', "
                "or a pair (x, y) from 0 to 1"
            )
        return name
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            x = float(value[0])
            y = float(value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "legend_position (x, y) uses numbers from 0 to 1"
            ) from exc
        if not (
            math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        ):
            raise ValueError("legend_position (x, y) uses numbers from 0 to 1")
        return (x, y)
    raise ValueError(
        "legend_position must be 'right', 'bottom', 'none', or a pair (x, y)"
    )

