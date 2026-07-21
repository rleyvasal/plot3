"""Shared assertions for plot3 functional and parity tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from plot3.build import build_spec


def assert_html_figure(html: str, *, min_chars: int = 400) -> None:
    assert isinstance(html, str)
    assert len(html) >= min_chars
    assert "<iframe" in html or "three" in html.lower() or "WebGL" in html


def assert_spec_3d(spec: dict[str, Any], *, expect: bool = True) -> None:
    assert "is3d" in spec
    assert spec["is3d"] is expect
    if expect:
        assert "z" in spec["scales"]
        assert spec.get("coord") is not None


def assert_layer_kind(spec: dict[str, Any], kind: str, *, index: int = 0) -> None:
    assert spec["layers"][index]["kind"] == kind
    assert spec["layers"][index]["n"] >= 0


def build_figure_spec(fig) -> dict[str, Any]:
    spec, _payloads = build_spec(fig)
    return spec


def nearly_equal(
    a: float | list[float] | np.ndarray,
    b: float | list[float] | np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    return bool(
        np.allclose(
            np.asarray(a, dtype=float),
            np.asarray(b, dtype=float),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    )


def tukey_boxplot_stats(values: np.ndarray, coef: float = 1.5) -> dict[str, float | list[float]]:
    """Python reference for Tukey five-number + outliers (ggplot2 default fences)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    v = np.sort(v)
    if v.size == 0:
        raise ValueError("empty sample")
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_fence = q1 - coef * iqr
    hi_fence = q3 + coef * iqr
    inside = v[(v >= lo_fence) & (v <= hi_fence)]
    ymin = float(inside.min()) if inside.size else float(q1)
    ymax = float(inside.max()) if inside.size else float(q3)
    outliers = v[(v < lo_fence) | (v > hi_fence)].tolist()
    return {
        "ymin": ymin,
        "lower": float(q1),
        "middle": float(med),
        "upper": float(q3),
        "ymax": ymax,
        "outliers": outliers,
    }


def assert_frame_close(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    cols: list[str] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    cols = cols or list(left.columns)
    assert list(left.columns) == list(right.columns) or set(cols) <= set(left.columns)
    for col in cols:
        a, b = left[col], right[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            assert nearly_equal(a.to_numpy(), b.to_numpy(), rtol=rtol, atol=atol), col
        else:
            assert a.astype(str).tolist() == b.astype(str).tolist(), col
