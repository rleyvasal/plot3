"""Multi-backend table adapters: pandas, polars, tidy3.

Design rules
------------
* Keep the original table type as long as possible (do not normalize at input).
* Stats / transforms run in the table backend (``count_by``, ``group_keys``, …).
* Rendering only needs plain arrays: call ``materialize_columns`` /
  ``to_arrays`` at the build boundary for the selected columns.

Backends
--------
* ``"pandas"``  — ``pandas.DataFrame``
* ``"polars"``  — ``polars.DataFrame`` or ``polars.LazyFrame``
* ``"tidy"``    — tidy3 ``TidyFrame`` (resolved to polars lazily)
* plain mappings / records fall back to a pandas DataFrame
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

Backend = Literal["pandas", "polars", "tidy"]

_POLARS_TYPES: tuple[type, ...] | None = None
_POLARS_READY = False


def _polars_types() -> tuple[type, ...] | None:
    """Lazy import of polars frame types (optional dependency)."""
    global _POLARS_TYPES, _POLARS_READY
    if _POLARS_READY:
        return _POLARS_TYPES
    _POLARS_READY = True
    try:
        import polars as pl

        _POLARS_TYPES = (pl.DataFrame, pl.LazyFrame)
    except ImportError:
        _POLARS_TYPES = None
    return _POLARS_TYPES


def _is_polars(data: Any) -> bool:
    types = _polars_types()
    return types is not None and isinstance(data, types)


def _is_tidy(data: Any) -> bool:
    """Duck-type tidy3.TidyFrame without importing tidy3."""
    if data is None:
        return False
    cls = type(data)
    mod = getattr(cls, "__module__", "") or ""
    name = cls.__name__
    if name == "TidyFrame" and ("tidy3" in mod or mod == "tidy3"):
        return True
    # Fallback: tidy frames expose backend + to_polars / collect
    if name == "TidyFrame" and hasattr(data, "to_polars") and hasattr(data, "columns"):
        return True
    return False


def detect_backend(data: Any) -> Backend:
    """Return the table backend name for *data*."""
    if isinstance(data, pd.DataFrame):
        return "pandas"
    if _is_polars(data):
        return "polars"
    if _is_tidy(data):
        return "tidy"
    raise TypeError(
        f"unsupported table type {type(data).__name__!r}; "
        "expected pandas DataFrame, polars DataFrame/LazyFrame, "
        "tidy3 TidyFrame, or a mapping/records constructor for pandas"
    )


def as_table(data: Any) -> Any:
    """Keep known table types; wrap plain constructors as pandas.

    pandas / polars / tidy3 frames are returned unchanged. Mappings, lists of
    records, and similar constructors become a ``pandas.DataFrame``.
    """
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data
    if _is_polars(data) or _is_tidy(data):
        return data
    # Already a known backend — detect_backend would succeed after wrap
    try:
        detect_backend(data)
        return data
    except TypeError:
        pass
    return pd.DataFrame(data)


def get_columns(data: Any) -> list[str]:
    """Column names as ``list[str]``."""
    backend = detect_backend(data)
    if backend == "pandas":
        return [str(c) for c in data.columns]
    if backend == "polars":
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            return list(data.collect_schema().names())
        return list(data.columns)
    # tidy
    cols = getattr(data, "columns", None)
    if cols is not None:
        return list(cols)
    return list(data.to_polars().columns)


def n_rows(data: Any) -> int:
    """Number of rows (may materialize a count for lazy frames)."""
    backend = detect_backend(data)
    if backend == "pandas":
        return int(len(data))
    if backend == "polars":
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            return int(data.select(pl.len()).collect().item())
        return int(data.height)
    # tidy
    return int(len(data))


def has_column(data: Any, name: str) -> bool:
    return name in get_columns(data)


def resolve_polars(data: Any):
    """Materialize tidy / LazyFrame to an eager polars DataFrame."""
    import polars as pl

    backend = detect_backend(data)
    if backend == "tidy":
        to_polars = getattr(data, "to_polars", None)
        if callable(to_polars):
            return to_polars()
        collect = getattr(data, "collect", None)
        if callable(collect):
            out = collect(as_="polars")
            if isinstance(out, pl.DataFrame):
                return out
        raise TypeError("tidy frame could not be resolved to polars")
    if backend == "polars":
        if isinstance(data, pl.LazyFrame):
            return data.collect()
        return data
    raise TypeError("resolve_polars() expects a polars or tidy table")


def _polars_to_pandas(frame) -> pd.DataFrame:
    """Convert a polars DataFrame without requiring pyarrow."""
    try:
        return frame.to_pandas()
    except ModuleNotFoundError:
        # pyarrow missing — build column-by-column from Python/NumPy values
        data = {}
        for name in frame.columns:
            series = frame.get_column(name)
            try:
                data[name] = series.to_numpy(allow_copy=True)
            except Exception:
                data[name] = series.to_list()
        return pd.DataFrame(data)


def as_pandas(data: Any) -> pd.DataFrame:
    """Escape hatch: full conversion to pandas (for non-migrated stats)."""
    backend = detect_backend(data)
    if backend == "pandas":
        return data
    if backend == "polars":
        return _polars_to_pandas(resolve_polars(data))
    # tidy
    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        try:
            out = to_pandas()
            if isinstance(out, pd.DataFrame):
                return out
        except ModuleNotFoundError:
            pass
    return _polars_to_pandas(resolve_polars(data))


def select_cols(data: Any, cols: list[str]):
    """Project columns; return same backend type when possible."""
    backend = detect_backend(data)
    missing = [c for c in cols if c not in get_columns(data)]
    if missing:
        raise KeyError(f"column(s) not in DataFrame: {missing}")
    if backend == "pandas":
        return data.loc[:, list(cols)]
    if backend == "polars":
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            return data.select(cols)
        return data.select(cols)
    # tidy — project via polars materialization of selected columns
    to_polars = getattr(data, "to_polars", None)
    if callable(to_polars):
        try:
            return to_polars(columns=cols)
        except TypeError:
            return resolve_polars(data).select(cols)
    return resolve_polars(data).select(cols)


def subsample_rows(data: Any, step: int):
    """Deterministic row stride (for ``coord_3d(max_points=…)``)."""
    step = max(1, int(step))
    if step == 1:
        return data
    backend = detect_backend(data)
    if backend == "pandas":
        return data.iloc[::step].copy()
    if backend == "polars":
        import polars as pl

        frame = resolve_polars(data)
        return frame.with_row_index("_plot3_i").filter(
            (pl.col("_plot3_i") % step) == 0
        ).drop("_plot3_i")
    # tidy → polars subsample (keeps plot path on polars)
    frame = resolve_polars(data)
    import polars as pl

    return frame.with_row_index("_plot3_i").filter(
        (pl.col("_plot3_i") % step) == 0
    ).drop("_plot3_i")


def count_by(data: Any, x: str):
    """Count rows by discrete *x*; return a table with columns ``[x, "y"]``.

    * pandas → ``groupby(...).size()``
    * polars → ``group_by(...).len()``
    * tidy   → collect to polars, then the polars path
    """
    if not has_column(data, x):
        raise KeyError(f"column(s) not in DataFrame: {[x]}")
    backend = detect_backend(data)
    if backend == "pandas":
        return (
            data.groupby(x, dropna=False, observed=True, sort=False)
            .size()
            .rename("y")
            .reset_index()
        )
    import polars as pl

    if backend == "tidy":
        frame = resolve_polars(data)
    elif isinstance(data, pl.LazyFrame):
        frame = data.collect()
    else:
        frame = data

    return (
        frame.group_by(x, maintain_order=True)
        .len()
        .rename({"len": "y"})
    )


def materialize_columns(data: Any, cols: list[str]) -> pd.DataFrame:
    """Select *cols*, drop rows with any NA, return a pandas frame.

    This is the **render boundary**: only the columns needed for encoding
    are converted. The original plot data stays in its native backend.
    """
    cols = list(dict.fromkeys(cols))
    missing = [c for c in cols if c not in get_columns(data)]
    if missing:
        raise KeyError(f"column(s) not in DataFrame: {missing}")
    backend = detect_backend(data)
    if backend == "pandas":
        return data.loc[:, cols].dropna()
    import polars as pl

    if backend == "tidy":
        frame = resolve_polars(data).select(cols)
    elif isinstance(data, pl.LazyFrame):
        frame = data.select(cols).collect()
    else:
        frame = data.select(cols)
    # drop_nulls on any of the selected columns
    frame = frame.drop_nulls(subset=cols)
    return _polars_to_pandas(frame)


def to_arrays(data: Any, cols: list[str]) -> dict[str, np.ndarray]:
    """Extract selected columns as NumPy arrays (after dropna on those cols)."""
    frame = materialize_columns(data, cols)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s.to_numpy(dtype=np.float64, copy=False)
        else:
            out[c] = s.to_numpy(copy=False)
    return out


def unique_levels(data: Any, col: str) -> list[Any]:
    """Discrete levels in order of appearance (for facets / violin)."""
    if not has_column(data, col):
        raise KeyError(f"column(s) not in DataFrame: {[col]}")
    backend = detect_backend(data)
    if backend == "pandas":
        s = data[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            levels = [c for c in s.cat.categories if (s == c).any()]
            if s.isna().any():
                levels = list(levels) + [pd.NA]
            return list(levels)
        return list(dict.fromkeys(s.tolist()))
    import polars as pl

    if backend == "tidy":
        frame = resolve_polars(data)
    elif isinstance(data, pl.LazyFrame):
        frame = data.collect()
    else:
        frame = data

    series = frame.get_column(col)
    # preserve appearance order
    seen: list[Any] = []
    for v in series.to_list():
        if v not in seen:
            # polars null
            if v is None:
                if None not in seen:
                    seen.append(None)
            else:
                seen.append(v)
    return seen


def filter_equal(data: Any, col: str, value: Any):
    """Rows where ``col == value`` (``value is None`` → nulls). Same backend."""
    backend = detect_backend(data)
    if backend == "pandas":
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
            return data.loc[data[col].isna()].copy()
        return data.loc[data[col] == value].copy()
    import polars as pl

    if backend == "tidy":
        frame = resolve_polars(data)
    elif isinstance(data, pl.LazyFrame):
        frame = data.collect()
    else:
        frame = data
    if value is None:
        return frame.filter(pl.col(col).is_null())
    return frame.filter(pl.col(col) == value)


def _eager_polars(data: Any):
    """Eager polars DataFrame from polars or tidy input."""
    import polars as pl

    backend = detect_backend(data)
    if backend == "tidy":
        return resolve_polars(data)
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data


def numeric_array(data: Any, col: str, *, dropna: bool = True) -> np.ndarray:
    """Extract *col* as ``float64`` (non-numeric → NaN).

    This is the preferred path for stats that operate on plain arrays
    (histogram, KDE, isosurface samples) without converting the whole table.
    """
    if not has_column(data, col):
        raise KeyError(f"column(s) not in DataFrame: {[col]}")
    backend = detect_backend(data)
    if backend == "pandas":
        arr = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        import polars as pl

        frame = _eager_polars(data)
        series = frame.get_column(col)
        # Cast loosely to float; strings / mixed become null.
        try:
            series = series.cast(pl.Float64, strict=False)
        except Exception:
            vals = []
            for v in series.to_list():
                try:
                    vals.append(float(v) if v is not None else np.nan)
                except (TypeError, ValueError):
                    vals.append(np.nan)
            arr = np.asarray(vals, dtype=np.float64)
        else:
            arr = series.to_numpy()
            if arr.dtype != np.float64:
                arr = np.asarray(arr, dtype=np.float64)
            # polars may yield object array for some null encodings
            if arr.dtype == object:
                arr = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(
                    dtype=np.float64
                )
    if dropna:
        return arr[np.isfinite(arr)]
    return arr


def group_pieces(
    data: Any, group_cols: list[str]
) -> list[tuple[tuple[Any, ...], Any]]:
    """Split *data* by *group_cols* in order of first appearance.

    Returns ``[(key_tuple, subgroup), ...]``. Subgroups keep the pandas
    backend when the source is pandas; polars/tidy sources yield polars
    ``DataFrame`` pieces (tidy is resolved first).
    """
    if not group_cols:
        return [((), data)]
    missing = [c for c in group_cols if not has_column(data, c)]
    if missing:
        raise KeyError(f"column(s) not in DataFrame: {missing}")
    backend = detect_backend(data)
    if backend == "pandas":
        out: list[tuple[tuple[Any, ...], Any]] = []
        for key, piece in data.groupby(
            group_cols, dropna=False, observed=True, sort=False
        ):
            key_tuple = key if isinstance(key, tuple) else (key,)
            out.append((tuple(key_tuple), piece))
        return out

    frame = _eager_polars(data)
    out = []
    for key, piece in frame.group_by(group_cols, maintain_order=True):
        if not isinstance(key, tuple):
            key_tuple = (key,)
        else:
            key_tuple = tuple(key)
        out.append((key_tuple, piece))
    return out


def category_labels(data: Any, col: str) -> list[str]:
    """Stable category labels for a discrete column (violin x, etc.).

    Prefer categorical dtype ordering when present; otherwise first-appearance
    order of stringified values.
    """
    if not has_column(data, col):
        raise KeyError(f"column(s) not in DataFrame: {[col]}")
    backend = detect_backend(data)
    if backend == "pandas":
        s = data[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            return [str(c) for c in s.cat.categories]
        return list(dict.fromkeys(s.astype(str).tolist()))

    import polars as pl

    frame = _eager_polars(data)
    series = frame.get_column(col)
    dtype = series.dtype
    if dtype == pl.Categorical or dtype == pl.Enum:
        cats = series.cat.get_categories().to_list()
        return [str(c) for c in cats]
    # appearance order
    seen: list[str] = []
    for v in series.to_list():
        label = "None" if v is None else str(v)
        if label not in seen:
            seen.append(label)
    return seen


def require_columns(data: Any, cols: list[str]) -> None:
    """Raise ``KeyError`` if any of *cols* is missing."""
    missing = [c for c in cols if not has_column(data, c)]
    if missing:
        raise KeyError(f"column(s) not in DataFrame: {missing}")
