"""ggplot figure object, display, and ggsave."""

from __future__ import annotations

import copy
import html as _htmlesc

import pandas as pd

from plot3.geoms import (
    _Geom,
    aes,
    facet_wrap,
    labs,
    scale_colour_continuous,
    _Theme,
)
from plot3.themes import _THEMES


class ggplot:
    """A plot3 figure, optionally deferred until data arrives via ``>>``."""

    def __init__(
        self,
        data=None,
        mapping: aes | None = None,
        *,
        height="480px",
        quantize=True,
        compress=True,
        hide=None,
    ):
        # ``ggplot(aes(...))`` is the R-shaped, pipeable form.  ``aes`` is a
        # dict subclass, so detect it before treating arbitrary mappings as
        # dataframe constructor input.
        if isinstance(data, aes) and mapping is None:
            data, mapping = None, data
        self.data = self._as_pandas(data) if data is not None else None
        self.mapping = mapping or aes()
        self.layers: list[_Geom] = []
        self.labs: dict = {}
        self.theme_name = "dark"
        self.cscale: scale_colour_continuous | None = None
        self.facet: facet_wrap | None = None
        self.height = height if isinstance(height, str) else f"{int(height)}px"
        self.quantize = bool(quantize)
        self.compress = bool(compress)
        self.hide = hide  # None -> module default (autohide())
    @staticmethod
    def _as_pandas(data) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        to_pandas = getattr(data, "to_pandas", None)
        if callable(to_pandas):
            data = to_pandas()
            if isinstance(data, pd.DataFrame):
                return data
        return pd.DataFrame(data)

    def __rrshift__(self, data):
        """Bind data to a deferred ``ggplot(aes(...))`` template."""
        if self.data is not None:
            raise TypeError("cannot pipe data into a ggplot that already has data")
        g = copy.copy(self)
        g.layers = list(self.layers)
        g.labs = dict(self.labs)
        g.facet = self.facet
        g.data = self._as_pandas(data)
        return g

    def __add__(self, other):
        g = copy.copy(self)
        g.layers = list(self.layers)
        g.labs = dict(self.labs)
        g.facet = self.facet
        if isinstance(other, _Geom):
            g.layers.append(other)
        elif isinstance(other, labs):
            g.labs.update(other)
        elif isinstance(other, _Theme):
            g.theme_name = other.name
        elif isinstance(other, scale_colour_continuous):
            g.cscale = other
        elif isinstance(other, facet_wrap):
            g.facet = other
        elif isinstance(other, aes):
            m = aes()
            m.update(self.mapping)
            m.update(other)
            g.mapping = m
        else:
            raise TypeError(f"cannot add {type(other).__name__!r} to ggplot")
        return g

    def _repr_html_(self) -> str:
        html = self._iframe()
        # SolveIt: big viewer HTML must not enter LLM context.
        if self.hide if self.hide is not None else AUTOHIDE:
            try:
                from plot3.jupyter import hide_caller_from_ai

                hide_caller_from_ai()
            except Exception:
                pass
        return html

    def html(self) -> str:
        """The full standalone document (what the iframe srcdoc carries)."""
        from plot3.build import build_doc

        return build_doc(self)

    def save(self, path: str) -> str:
        doc = self.html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        print(f"plot3: saved {path} ({len(doc) // 1024} KB)")
        return path

    def _iframe(self) -> str:
        doc = self.html()
        title = self.labs.get("title", "plot3 figure")
        return (
            f'<iframe srcdoc="{_htmlesc.escape(doc, quote=True)}" '
            f'style="width:100%;height:{self.height};border:0;'
            f'border-radius:6px;background:{_THEMES[self.theme_name]["surface"]}" '
            f'title="{_htmlesc.escape(str(title))}"></iframe>'
        )


AUTOHIDE = True


def autohide(on: bool = True) -> None:
    """Default hide-from-AI behavior for displayed figures (SolveIt red eye)."""
    global AUTOHIDE
    AUTOHIDE = bool(on)


def ggsave(filename, plot: ggplot | None = None, **_kw) -> str:
    """ggsave("fig.html", p) — ggplot2-style save (HTML only)."""
    if isinstance(filename, ggplot) and isinstance(plot, str):
        filename, plot = plot, filename  # tolerate swapped args
    if plot is None:
        raise ValueError("ggsave(filename, plot) needs the plot")
    return plot.save(filename)
