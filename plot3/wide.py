"""Wide-form input: several measurement columns in one layer.

``geom_boxplot(columns=["a", "b", "c"])`` draws one box per column without
reshaping the data first. Before statistics run, the layer's rows are
stacked into long form, as ``tidyr::pivot_longer`` would: a ``name`` column
holds the column labels (in the order given) and a ``value`` column holds
the measurements. The layer is then mapped by what the geom draws:

* boxplot, violin: x is the column name, y its values, one fill per column.
* histogram, density, freqpoly, ecdf: x is the values, one fill or colour
  per column (overlaid, stacked, or in panels as the geom's position says).
* geom_qq: the values are the sample.
* everything else (line, point, col, area, ...): the plot's x is shared,
  y is the values, one colour or fill per column. ``geom_col`` stacks the
  columns as components of a total; ``position="dodge"`` groups them.

``columns`` is a list, a dict ``{column: label}`` to rename them, or a
pattern such as ``"measurement_*"``. ``names_to`` and ``values_to`` name the
two new columns (and so the legend and value-axis titles).

The same stacking can be written in ``aes()``, where you say which
aesthetic gets the values and which the column names, as for any mapping::

    ggplot(df, aes(x=column_names(), y=columns("measurement_*"), fill=column_names()))
        + geom_boxplot() + geom_jitter()

In the plot's ``aes()`` the data are stacked once, before facets split
them, so every layer sees the long rows and ``facet_wrap("name")`` works.
Each column is its own group unless ``group`` is mapped.
"""

from __future__ import annotations

import copy
import fnmatch

import pandas as pd

_DISTRIBUTION_BY_NAME = {"boxplot", "violin"}
_ONE_VARIABLE = {"histogram", "density", "freqpoly", "ecdf"}
_SAMPLE = {"qq", "qq_line"}
_COLOUR_AES = ("color", "fill")


class _WideMarker:
    """A placeholder in aes() that stacking replaces with a column name."""


class columns(_WideMarker):
    """The values of several columns, stacked: ``aes(y=columns("m_*"))``.

    Takes column names (``columns("a", "b")``), a list, a ``{column:
    label}`` dict, or a pattern. ``names_to`` / ``values_to`` name the two
    stacked columns (default ``name`` and ``value``).
    """

    def __init__(self, *cols, names_to: str = "name", values_to: str = "value"):
        if len(cols) == 1 and isinstance(cols[0], (str, list, tuple, dict)):
            self.spec = cols[0]
        else:
            self.spec = [str(c) for c in cols]
        self.names_to = str(names_to)
        self.values_to = str(values_to)

    def __repr__(self):
        return f"columns({self.spec!r})"


class column_names(_WideMarker):
    """Which stacked column a row came from: ``aes(fill=column_names())``."""

    def __repr__(self):
        return "column_names()"


def _markers(mapping) -> tuple[columns | None, list[str], list[str]]:
    """The ``columns()`` in a mapping, the aesthetics holding it, and those
    holding ``column_names()``."""
    found, value_aes, name_aes = None, [], []
    for aes_name, value in dict(mapping or {}).items():
        if isinstance(value, columns):
            if found is not None and (value.spec, value.names_to, value.values_to) != (
                found.spec, found.names_to, found.values_to
            ):
                raise ValueError("one aes() can stack one set of columns()")
            found = found or value
            value_aes.append(aes_name)
        elif isinstance(value, column_names):
            name_aes.append(aes_name)
    if name_aes and found is None:
        raise ValueError("column_names() needs columns(...) in the same aes()")
    return found, value_aes, name_aes


def _resolved_mapping(mapping, marker: columns, value_aes, name_aes, *, add_group: bool) -> dict:
    out = dict(mapping or {})
    for aes_name in value_aes:
        out[aes_name] = marker.values_to
    for aes_name in name_aes:
        out[aes_name] = marker.names_to
    if add_group and "group" not in out:
        out["group"] = marker.names_to
    return out


def _selected(spec, available: list[str]) -> dict[str, str]:
    """``{column: label}`` in the order given."""
    if isinstance(spec, str):
        if any(ch in spec for ch in "*?["):
            found = [c for c in available if fnmatch.fnmatchcase(c, spec)]
            if not found:
                raise ValueError(f"columns={spec!r} matches no column of the data")
            return {c: c for c in found}
        spec = [spec]
    if isinstance(spec, dict):
        chosen = {str(k): str(v) for k, v in spec.items()}
    else:
        chosen = {str(c): str(c) for c in spec}
    if not chosen:
        raise ValueError("columns= needs at least one column")
    if len(set(chosen.values())) != len(chosen):
        raise ValueError("columns= labels must differ from one another")
    return chosen


def _as_pandas(data) -> pd.DataFrame:
    from plot3.table import _eager_polars, _polars_to_pandas, detect_backend

    backend = detect_backend(data)
    if backend == "pandas":
        return data
    if backend in {"polars", "tidy"}:
        return _polars_to_pandas(_eager_polars(data))
    raise ValueError("columns= needs a pandas or polars data frame")


def _melt(data, chosen: dict[str, str], names_to: str, values_to: str) -> pd.DataFrame:
    from plot3.table import ColumnNotFound, get_columns

    frame = _as_pandas(data)
    missing = [c for c in chosen if c not in frame.columns]
    if missing:
        raise ColumnNotFound(missing, data)
    for column in chosen:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(f"columns=: {column!r} is not numeric")
    for new in (names_to, values_to):
        if new in frame.columns and new not in chosen:
            raise ValueError(
                f"the data already has a column {new!r}; pass names_to= / values_to= "
                "to name the stacked columns"
            )
    keep = [c for c in get_columns(frame) if c not in chosen]
    pieces = []
    for column, label in chosen.items():
        piece = frame[keep].copy()
        piece[names_to] = label
        piece[values_to] = frame[column].to_numpy()
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    # Columns keep the order they were given in, not alphabetical.
    out[names_to] = pd.Categorical(out[names_to], categories=list(chosen.values()))
    return out


def _mapping_for(geom, base_mapping, names_to: str, values_to: str) -> dict:
    from plot3.build import _FILL_KINDS

    kind = getattr(geom, "kind", "")
    own = dict(getattr(geom, "mapping", None) or {})
    merged = {**dict(base_mapping or {}), **own}
    taken = [a for a in ("x", "y", "sample") if a in own]

    def place(aes_name):
        if aes_name in own:
            raise ValueError(
                f"columns= supplies aes({aes_name}=) on this layer; remove it from the mapping"
            )
        own[aes_name] = values_to

    if kind in _DISTRIBUTION_BY_NAME:
        if taken:
            raise ValueError(f"columns= supplies x and y on this layer; remove aes({taken[0]}=)")
        own["x"], own["y"] = names_to, values_to
    elif kind in _ONE_VARIABLE:
        place("x")
    elif kind in _SAMPLE:
        place("sample")
    else:
        if "x" not in merged:
            raise ValueError(
                "columns= puts the values on y; map the shared x, "
                'for example aes(x="observation")'
            )
        place("y")
    channel = "fill" if kind in _FILL_KINDS else "color"
    has_colour = any(a in merged for a in _COLOUR_AES) or getattr(geom, "const_color", None) is not None
    if not has_colour:
        own[channel] = names_to
    if kind not in _DISTRIBUTION_BY_NAME and "group" not in merged:
        own["group"] = names_to
    return own


def _as_aes(mapping: dict):
    from plot3.geoms import aes

    out = aes()
    out.update(mapping)
    return out


def melt_wide_mapping(g):
    """The figure with ``aes(y=columns(...))`` in its own mapping stacked:
    its data hold the long rows, before facets split them."""
    from plot3.table import get_columns

    marker, value_aes, name_aes = _markers(g.mapping)
    if marker is None:
        return g
    if g.data is None:
        raise ValueError("columns() needs data: ggplot(df, aes(...))")
    chosen = _selected(marker.spec, [str(c) for c in get_columns(g.data)])
    out = copy.copy(g)
    out.data = _melt(g.data, chosen, marker.names_to, marker.values_to)
    out.mapping = _as_aes(_resolved_mapping(g.mapping, marker, value_aes, name_aes, add_group=True))
    return out


def melt_wide_layers(g):
    """The figure with each ``columns=`` layer, and each layer whose own
    ``aes()`` holds ``columns()``, stacked into long form."""
    from plot3.geoms import aes
    from plot3.table import get_columns

    if not any(
        getattr(layer, "columns", None) is not None or _markers(getattr(layer, "mapping", None))[0]
        for layer in g.layers
    ):
        return g
    out = copy.copy(g)
    labs = dict(getattr(g, "labs", None) or {})

    def title(axis, text):
        # labs() and scale names still win, as for any mapped column.
        pscale = getattr(g, f"{axis}scale", None)
        if labs.get(axis) is None and getattr(pscale, "name", None) is None:
            labs[axis] = text

    layers = []
    for layer in g.layers:
        spec = getattr(layer, "columns", None)
        marker, value_aes, name_aes = _markers(getattr(layer, "mapping", None))
        if marker is not None:
            if spec is not None:
                raise ValueError("give columns= or aes(...=columns()), not both")
            data = layer.layer_data if layer.layer_data is not None else g.data
            if data is None:
                raise ValueError("columns() needs data: ggplot(df) or the layer's data=")
            moved = copy.copy(layer)
            moved.layer_data = _melt(
                data, _selected(marker.spec, [str(c) for c in get_columns(data)]),
                marker.names_to, marker.values_to,
            )
            moved.mapping = _as_aes(
                _resolved_mapping(layer.mapping, marker, value_aes, name_aes, add_group=True)
            )
            layers.append(moved)
            continue
        if spec is None:
            layers.append(layer)
            continue
        data = layer.layer_data if layer.layer_data is not None else g.data
        if data is None:
            raise ValueError("columns= needs data: ggplot(df) or the layer's data=")
        names_to, values_to = layer.names_to, layer.values_to
        chosen = _selected(spec, [str(c) for c in get_columns(data)])
        moved = copy.copy(layer)
        moved.layer_data = _melt(data, chosen, names_to, values_to)
        moved.mapping = aes()
        moved.mapping.update(_mapping_for(layer, g.mapping, names_to, values_to))
        moved.columns = None
        layers.append(moved)
        for axis in ("x", "y"):
            if moved.mapping.get(axis) in (names_to, values_to):
                title(axis, moved.mapping[axis])
        if names_to in (moved.mapping.get("color"), moved.mapping.get("fill")):
            # The legend is titled like a mapped column; labs(fill=) still wins.
            if labs.get("color") is None and labs.get("fill") is None:
                labs["color"] = names_to
    out.layers = layers
    out.labs = labs
    return out
