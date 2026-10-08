"""Several plots in one figure (patchwork-style): ``p1 | p2``, ``p1 / p2``.

    (p1 | p2) / p3 + plot_annotation(title="Results", tag_levels="A")

``|`` puts plots side by side, ``/`` stacks them, and the two nest.
``plot_layout(widths=[2, 1])`` sets relative sizes. ``plot_annotation``
adds a figure title, subtitle, and caption, and tags the panels A, B, C
(or ``"a"``, ``"1"``, ``"I"``). A plot's own ``labs(tag=)`` wins.

Every plot keeps its own scales and legend. Static export draws them all
with one type size, taken from the whole figure, so panels match.
"""

from __future__ import annotations

import copy
import html as _htmlesc
from pathlib import Path
from typing import Any


class plot_layout:
    """``widths`` / ``heights``: relative sizes of the side-by-side or stacked
    parts. ``height`` is the whole figure's height in the notebook viewer
    (pixels, or a CSS length such as ``"60vh"``)."""

    def __init__(self, *, widths=None, heights=None, height=None):
        self.widths = None if widths is None else [float(w) for w in widths]
        self.heights = None if heights is None else [float(h) for h in heights]
        self.height = None if height is None else (height if isinstance(height, str) else f"{int(height)}px")
        for sizes in (self.widths, self.heights):
            if sizes is not None and any(v <= 0 for v in sizes):
                raise ValueError("plot_layout() sizes must be positive")


class plot_annotation:
    """Title, subtitle, and caption for the whole figure, and panel tags.

    ``tag_levels`` is ``"A"`` (A, B, C), ``"a"``, ``"1"``, or ``"I"``
    (roman). ``tag_prefix`` / ``tag_suffix`` wrap each tag, e.g. ``"("``
    and ``")"``.
    """

    def __init__(self, *, title=None, subtitle=None, caption=None,
                 tag_levels=None, tag_prefix="", tag_suffix=""):
        if tag_levels not in {None, "A", "a", "1", "I", "i"}:
            raise ValueError('tag_levels is "A", "a", "1", "I", or "i"')
        self.title = title
        self.subtitle = subtitle
        self.caption = caption
        self.tag_levels = tag_levels
        self.tag_prefix = str(tag_prefix)
        self.tag_suffix = str(tag_suffix)


def _roman(n: int) -> str:
    parts = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, symbol in parts:
        while n >= value:
            out += symbol
            n -= value
    return out


def _tag(index: int, levels: str) -> str:
    if levels == "A":
        return chr(ord("A") + index % 26)
    if levels == "a":
        return chr(ord("a") + index % 26)
    if levels == "1":
        return str(index + 1)
    if levels == "I":
        return _roman(index + 1)
    return _roman(index + 1).lower()


class Composition:
    """A row (``|``) or column (``/``) of plots and nested compositions."""

    def __init__(self, direction: str, items: list, sizes=None, annotation=None):
        self.direction = direction  # "row" | "col"
        self.items = list(items)
        self.sizes = sizes
        self.annotation = annotation
        # None: tall enough for its rows of plots (see figure_height).
        self.height = None

    def rows(self) -> int:
        """How many plots stand above one another."""
        counts = [item.rows() if isinstance(item, Composition) else 1 for item in self.items]
        return sum(counts) if self.direction == "col" else max(counts)

    def figure_height(self) -> str:
        """The viewer's height: as set, or 400 px per row of plots (480 px at
        least, a single plot's height), plus room for a figure title."""
        if self.height:
            return self.height
        extra = 40 if getattr(self.annotation, "title", None) else 0
        return f"{max(480, 400 * self.rows()) + extra}px"

    @property
    def theme_family(self):
        return getattr(self.leaves()[0], "theme_family", None)

    @property
    def theme_base_size(self):
        return getattr(self.leaves()[0], "theme_base_size", None)

    # ── grammar ──────────────────────────────────────────────────────────
    def _join(self, other, direction):
        _check_plot(other)
        if self.direction == direction and self.sizes is None and self.annotation is None:
            return Composition(direction, self.items + [other])
        return Composition(direction, [self, other])

    def __or__(self, other):
        return self._join(other, "row")

    def __truediv__(self, other):
        return self._join(other, "col")

    def __ror__(self, other):
        return Composition("row", [other, self])

    def __rtruediv__(self, other):
        return Composition("col", [other, self])

    def __add__(self, other):
        out = copy.copy(self)
        if isinstance(other, plot_layout):
            sizes = other.widths if self.direction == "row" else other.heights
            if sizes is not None and len(sizes) != len(self.items):
                raise ValueError(
                    f"plot_layout() got {len(sizes)} sizes for {len(self.items)} parts"
                )
            out.sizes = sizes if sizes is not None else self.sizes
            if other.height is not None:
                out.height = other.height
            return out
        if isinstance(other, plot_annotation):
            out.annotation = other
            return out
        raise TypeError(
            "add plot_layout() or plot_annotation() to a figure of several plots"
        )

    # ── layout ───────────────────────────────────────────────────────────
    def leaves(self) -> list:
        out = []
        for item in self.items:
            out.extend(item.leaves() if isinstance(item, Composition) else [item])
        return out

    def tagged(self) -> "Composition":
        """A copy whose plots carry their tags (the figure's tag_levels)."""
        levels = getattr(self.annotation, "tag_levels", None)
        if not levels:
            return self
        prefix = self.annotation.tag_prefix
        suffix = self.annotation.tag_suffix
        counter = iter(range(10_000))

        def walk(node):
            if isinstance(node, Composition):
                clone = copy.copy(node)
                clone.items = [walk(item) for item in node.items]
                return clone
            index = next(counter)
            if node.labs.get("tag"):
                return node
            plot = copy.copy(node)
            plot.labs = dict(node.labs, tag=f"{prefix}{_tag(index, levels)}{suffix}")
            return plot

        return walk(self)

    def rects(self, x: float, y: float, w: float, h: float, gap: float = 10.0) -> list:
        """``[(plot, x, y, w, h)]`` for every plot inside this rectangle."""
        n = len(self.items)
        weights = self.sizes or [1.0] * n
        total = float(sum(weights))
        out = []
        along = w if self.direction == "row" else h
        usable = along - gap * (n - 1)
        at = x if self.direction == "row" else y
        for item, weight in zip(self.items, weights):
            size = usable * weight / total
            if self.direction == "row":
                box = (at, y, size, h)
            else:
                box = (x, at, w, size)
            at += size + gap
            if isinstance(item, Composition):
                out.extend(item.rects(*box, gap=gap))
            else:
                out.append((item, *box))
        return out

    # ── output ───────────────────────────────────────────────────────────
    def html(self) -> str:
        """A standalone page: the plots in nested flex boxes."""
        fig = self.tagged()
        esc = _htmlesc.escape
        theme = _first_theme(fig)

        def node_html(node) -> str:
            if isinstance(node, Composition):
                flow = "row" if node.direction == "row" else "column"
                weights = node.sizes or [1.0] * len(node.items)
                parts = "".join(
                    f"<div class='part' style='flex:{weight} 1 0'>{node_html(item)}</div>"
                    for item, weight in zip(node.items, weights)
                )
                return f"<div class='box' style='flex-direction:{flow}'>{parts}</div>"
            doc = node.html()
            return f"<iframe srcdoc=\"{esc(doc, quote=True)}\" title=\"panel\"></iframe>"

        note = fig.annotation
        title = esc(str(getattr(note, "title", "") or ""))
        subtitle = esc(str(getattr(note, "subtitle", "") or ""))
        caption = esc(str(getattr(note, "caption", "") or ""))
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;height:100%;background:{theme["surface"]};color:{theme["ink"]};
  font:12px system-ui,-apple-system,"Segoe UI",sans-serif}}
#wrap{{box-sizing:border-box;height:100%;padding:8px;display:flex;flex-direction:column;gap:4px}}
#ctitle{{font-size:15px;font-weight:600;margin:0 4px}}
#csub{{font-size:12px;color:{theme["ink2"]};margin:0 4px 4px}}
#ccap{{font-size:11px;color:{theme["muted"]};text-align:right;margin:0 4px}}
.box{{display:flex;gap:10px;flex:1;min-height:0;min-width:0;height:100%}}
.part{{display:flex;min-height:0;min-width:0}}
iframe{{flex:1;width:100%;height:100%;border:0;background:{theme["surface"]}}}
</style></head><body><div id="wrap">
{f"<div id='ctitle'>{title}</div>" if title else ""}{f"<div id='csub'>{subtitle}</div>" if subtitle else ""}
{node_html(fig)}
{f"<div id='ccap'>{caption}</div>" if caption else ""}
</div></body></html>"""

    def save(self, path, *, width=None, height=None, units="px", dpi=None,
             family=None, fontsize=None, notes=False) -> str:
        suffix = Path(str(path)).suffix.lower()
        if suffix in {".png", ".svg", ".pdf"}:
            from plot3.static import save_static

            return save_static(
                self, path, width=width, height=height, units=units, dpi=dpi,
                family=family, fontsize=fontsize, notes=notes,
            )
        doc = self.html()
        Path(str(path)).write_text(doc, encoding="utf-8")
        print(f"plot3: saved {path} ({len(doc) // 1024} KB)")
        return str(path)

    def _iframe(self) -> str:
        theme = _first_theme(self)
        return (
            f'<div class="plot3-fig"><iframe srcdoc="{_htmlesc.escape(self.html(), quote=True)}" '
            'sandbox="allow-scripts allow-same-origin allow-pointer-lock allow-downloads" '
            'allow="fullscreen; clipboard-write" '
            f'style="width:100%;height:{self.figure_height()};border:0;border-radius:6px;'
            f'background:{theme["surface"]}" title="plot3 figure"></iframe></div>'
        )

    def _repr_html_(self) -> str:
        return self._iframe()

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display

        display(HTML(self._iframe()))


def _check_plot(item: Any) -> None:
    from plot3.ggplot import ggplot

    if not isinstance(item, (ggplot, Composition)):
        raise TypeError("| and / combine plots: ggplot(...) | ggplot(...)")


def _first_theme(fig) -> dict:
    from plot3.themes import _THEMES

    leaf = fig.leaves()[0]
    return _THEMES[leaf.theme_name]
