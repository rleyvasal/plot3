"""Shared fixtures for the plot3 local test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
LOCAL = TESTS / "local"


@pytest.fixture
def root_dir() -> Path:
    return ROOT


@pytest.fixture
def local_dir() -> Path:
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / "output").mkdir(parents=True, exist_ok=True)
    return LOCAL


@pytest.fixture
def cars() -> pd.DataFrame:
    """Small mtcars-like frame for grammar smoke tests."""
    return pd.DataFrame(
        {
            "mpg": [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4],
            "cyl": [6, 6, 4, 6, 8, 6, 8, 4],
            "disp": [160, 160, 108, 258, 360, 225, 360, 146.7],
            "hp": [110, 110, 93, 110, 175, 105, 245, 62],
            "wt": [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19],
            "gear": [4, 4, 4, 3, 3, 3, 3, 4],
        }
    )


@pytest.fixture
def cloud() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "z": rng.normal(size=n),
            "intensity": rng.uniform(0, 1, n),
        }
    )


@pytest.fixture
def grid() -> pd.DataFrame:
    xs = np.linspace(-2, 2, 12)
    ys = np.linspace(-1.5, 1.5, 10)
    xx, yy = np.meshgrid(xs, ys)
    return pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "height": (np.sin(xx) * np.cos(yy)).ravel(),
        }
    )


@pytest.fixture
def box_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "g": ["a"] * 40 + ["b"] * 40,
            "y": list(rng.normal(0, 1, 40))
            + list(rng.normal(2, 1, 39))
            + [12.0],
        }
    )
