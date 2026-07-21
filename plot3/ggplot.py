"""ggplot figure object, display, and ggsave."""

from __future__ import annotations

import copy
import html as _htmlesc
import os
import sys
import webbrowser
from pathlib import Path

import pandas as pd

from plot3.geoms import (
    _Geom,
    aes,
    coord_3d,
    facet_wrap,
    labs,
    scale_colour_continuous,
    stat_density_3d,
    _Theme,
)
from plot3.themes import _THEMES


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _in_solveit() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("dialoghelper") is not None
    except Exception:
        return False


def _in_vscode_notebook() -> bool:
    """True when the kernel was launched from VS Code / Cursor.

    The kernel process often does **not** inherit VSCODE_* env vars, so also
    walk parent process names (macOS/Linux).
    """
    keys = (
        "VSCODE_PID",
        "VSCODE_CWD",
        "VSCODE_NLS_CONFIG",
        "VSCODE_ESM_ENTRYPOINT",
        "VSCODE_HANDLES_UNCAUGHT_ERRORS",
        "CURSOR_TRACE_ID",
    )
    if any(k in os.environ for k in keys):
        return True
    # Connection file / argv hints used by the Jupyter extension
    joined = " ".join(sys.argv).lower()
    if "vscode" in joined or "cursor" in joined:
        return True
    try:
        import subprocess

        pid = os.getpid()
        for _ in range(6):
            out = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "ppid=,comm="],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if not out:
                break
            parts = out.split(None, 1)
            if len(parts) < 2:
                break
            ppid_s, comm = parts[0], parts[1].lower()
            if any(
                tag in comm
                for tag in (
                    "visual studio code",
                    "code helper",
                    "cursor",
                    "electron",
                    "code ",
                )
            ) or comm in {"code", "cursor"}:
                return True
            try:
                pid = int(ppid_s)
            except ValueError:
                break
            if pid <= 1:
                break
    except Exception:
        pass
    return False


def _prefer_external_browser() -> bool:
    """VS Code notebook webviews block CDN ES modules inside iframes.

    SolveIt / full browsers render the srcdoc iframe fine. Force either mode
    with PLOT3_DISPLAY=browser|iframe.
    """
    mode = (os.environ.get("PLOT3_DISPLAY") or "").strip().lower()
    if mode in ("browser", "external", "file"):
        return True
    if mode in ("iframe", "inline", "notebook"):
        return False
    # SolveIt first: dialoghelper present ⇒ always iframe (even if a VS Code
    # env var leaked into the kernel process).
    if _in_solveit():
        return False
    return _in_vscode_notebook()


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
        self.coord: coord_3d | None = None
        self.stat_density_3d: stat_density_3d | None = None
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
        g.coord = self.coord
        g.stat_density_3d = self.stat_density_3d
        g.data = self._as_pandas(data)
        return g

    def __add__(self, other):
        g = copy.copy(self)
        g.layers = list(self.layers)
        g.labs = dict(self.labs)
        g.facet = self.facet
        g.coord = self.coord
        g.stat_density_3d = self.stat_density_3d
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
        elif isinstance(other, coord_3d):
            g.coord = other
        elif isinstance(other, stat_density_3d):
            g.stat_density_3d = other
        elif isinstance(other, aes):
            m = aes()
            m.update(self.mapping)
            m.update(other)
            g.mapping = m
        else:
            raise TypeError(f"cannot add {type(other).__name__!r} to ggplot")
        return g

    def _maybe_hide_from_ai(self) -> None:
        # SolveIt: big viewer HTML must not enter LLM context.
        if self.hide if self.hide is not None else AUTOHIDE:
            try:
                from plot3.jupyter import hide_caller_from_ai

                hide_caller_from_ai()
            except Exception:
                pass

    def _repr_html_(self) -> str:
        # Used by hosts that only understand HTML reprs (and by tests).
        # Prefer ``display(fig)`` / ``fig.show()`` so VS Code can open a browser.
        self._maybe_hide_from_ai()
        return self._iframe()

    def _ipython_display_(self) -> None:
        """IPython entry point — browser in VS Code, iframe in SolveIt."""
        self._maybe_hide_from_ai()
        self.show(browser=_prefer_external_browser())

    def html(self) -> str:
        """The full standalone document (what the iframe srcdoc carries)."""
        from plot3.build import build_doc

        return build_doc(self)

    def save(self, path: str | Path) -> str:
        path = str(path)
        doc = self.html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        print(f"plot3: saved {path} ({len(doc) // 1024} KB)")
        return path

    def _write_preview(self, path: str | Path | None = None) -> Path:
        if path is None:
            out_dir = Path.cwd() / ".plot3_preview"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "latest.html"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html(), encoding="utf-8")
        return path.resolve()

    def show(self, *, browser: bool | None = None, path: str | Path | None = None):
        """Display the figure.

        Parameters
        ----------
        browser:
            ``True`` write a standalone HTML file and open it in the system
            browser (reliable in VS Code). ``False`` embed an iframe in the
            notebook output (SolveIt / classic Jupyter). ``None`` auto-detect.
        path:
            Optional HTML path when using the browser path (default
            ``./.plot3_preview/latest.html``).
        """
        if browser is None:
            browser = _prefer_external_browser()

        if browser:
            out = self._write_preview(path)
            uri = out.as_uri()
            open_browser = _env_flag("PLOT3_NO_BROWSER") is not True
            try:
                from IPython.display import HTML, display

                display(
                    HTML(
                        "<div style='font:13px system-ui,sans-serif;padding:8px 10px;"
                        "border-radius:8px;background:#1e293b;color:#e2e8f0'>"
                        "<b>plot3</b>: opened in your system browser (VS Code notebook "
                        "webviews block the WebGL/CDN viewer inline). File: "
                        f"<code style='color:#93c5fd'>{out}</code>"
                        "</div>"
                    )
                )
            except Exception:
                print(f"plot3: open in browser → {out}")
            if open_browser:
                webbrowser.open(uri)
            return out

        try:
            from IPython.display import HTML, display

            display(HTML(self._iframe()))
            # Restricted hosts (VS Code) often show a blank panel: keep a file
            # fallback so the figure is never "lost" when auto-detect misses.
            if not _in_solveit():
                out = self._write_preview(path)
                display(
                    HTML(
                        "<div style='font:12px system-ui,sans-serif;margin-top:6px;"
                        "color:#94a3b8'>If the panel above is blank, run "
                        "<code style='color:#93c5fd'>fig.show(browser=True)</code> "
                        f"or open <code style='color:#93c5fd'>{out}</code></div>"
                    )
                )
                return out
        except Exception:
            out = self._write_preview(path)
            webbrowser.open(out.as_uri())
            return out
        return None

    def _iframe(self) -> str:
        doc = self.html()
        title = self.labs.get("title", "plot3 figure")
        # sandbox must allow scripts or the three.js viewer never starts.
        return (
            f'<iframe srcdoc="{_htmlesc.escape(doc, quote=True)}" '
            f'sandbox="allow-scripts allow-same-origin allow-pointer-lock" '
            f'allow="fullscreen" '
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
