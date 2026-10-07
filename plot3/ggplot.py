"""ggplot figure object, display, and ggsave."""

from __future__ import annotations

import copy
import html as _htmlesc
import os
import sys
import webbrowser
from pathlib import Path

from plot3.geoms import (
    _Geom,
    aes,
    coord_3d,
    coord_equal,
    facet_wrap,
    labs,
    scale_colour_continuous,
    scale_x_log10,
    scale_y_log10,
    stat_density_3d,
    slider,
    transition_states,
    transition_time,
    _Theme,
)
from plot3.table import as_table, detect_backend
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


def _display_mode() -> str:
    return (os.environ.get("PLOT3_DISPLAY") or "").strip().lower()


def _skip_blank_hint() -> bool:
    """No fallback hint in SolveIt, or when the user already chose an iframe."""
    if _in_solveit():
        return True
    return _display_mode() in ("iframe", "inline", "notebook")


def _prefer_external_browser() -> bool:
    """VS Code notebook webviews block CDN ES modules inside iframes.

    SolveIt / full browsers render the srcdoc iframe fine. Force either mode
    with PLOT3_DISPLAY=browser|iframe.
    """
    mode = _display_mode()
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

    # Prefer ggplot.__rrshift__ over NumPy bitshift when piping an ndarray:
    # ``points >> ggplot(aes(x=0, y=1)) + geom_point()``.
    __array_priority__ = 10000

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
        self.data = self._as_table(data) if data is not None else None
        self.backend = self._detect_backend(self.data) if self.data is not None else None
        self.mapping = mapping or aes()
        self.layers: list[_Geom] = []
        self.labs: dict = {}
        self.theme_name = "dark"
        self.cscale: scale_colour_continuous | None = None
        self.facet: facet_wrap | None = None
        self.coord: coord_3d | coord_equal | None = None
        self.stat_density_3d: stat_density_3d | None = None
        self.scale_x: scale_x_log10 | None = None
        self.scale_y: scale_y_log10 | None = None
        self.transition: transition_time | transition_states | None = None
        self.slider: slider | None = None
        self.height = height if isinstance(height, str) else f"{int(height)}px"
        self.quantize = bool(quantize)
        self.compress = bool(compress)
        self.hide = hide  # None -> module default (autohide())
        # Frozen PlotPayload (Phase C): display without source data.
        self._payload: dict | None = None

    @staticmethod
    def _as_table(data):
        """Keep pandas / polars / tidy3 frames; wrap plain constructors as pandas."""
        return as_table(data)

    @staticmethod
    def _detect_backend(data) -> str:
        return detect_backend(data)

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        *,
        height: str | int = "480px",
        hide: bool | None = None,
    ) -> "ggplot":
        """Rebuild a displayable figure from a PlotPayload (no DataFrame).

        The returned object can call :meth:`html`, :meth:`show`, and
        :meth:`save` using only the payload. Grammar operators (``+``, ``>>``)
        invalidate the payload and require live data again.
        """
        from plot3.payload import validate_payload

        payload = validate_payload(payload)
        g = cls(data=None, height=height, hide=hide)
        g._payload = payload
        # Best-effort labels for iframe title; full labs live in the payload.
        labs_map = payload["spec"].get("labs") or {}
        g.labs = {
            k: v
            for k, v in labs_map.items()
            if v not in (None, "") and k in ("title", "x", "y", "z", "color")
        }
        return g

    def freeze(self) -> "ggplot":
        """Encode stats into a PlotPayload and drop the source table.

        After :meth:`freeze`, display methods use only the payload (suitable
        for shipping across a local/remote boundary). Adding layers or piping
        new data clears the freeze.
        """
        if getattr(self, "facet", None) is not None:
            raise ValueError(
                "freeze() does not support facet_wrap(); use html() for faceted figures"
            )
        if self._payload is None:
            if self.data is None and not self.layers:
                raise ValueError("freeze() needs data and at least one layer")
            self._payload = self.to_payload()
        self.data = None
        self.backend = None
        return self

    def __rrshift__(self, data):
        """Bind data to a deferred ``ggplot(aes(...))`` template."""
        if self.data is not None:
            raise TypeError("cannot pipe data into a ggplot that already has data")
        if self._payload is not None:
            raise TypeError(
                "cannot pipe data into a payload-backed ggplot; "
                "build a new ggplot(...) template instead"
            )
        g = copy.copy(self)
        g.layers = list(self.layers)
        g.labs = dict(self.labs)
        g.facet = self.facet
        g.coord = self.coord
        g.stat_density_3d = self.stat_density_3d
        g._payload = None
        g.data = self._as_table(data)
        g.backend = self._detect_backend(g.data)
        return g

    def __add__(self, other):
        g = copy.copy(self)
        g.layers = list(self.layers)
        g.labs = dict(self.labs)
        g.facet = self.facet
        g.coord = self.coord
        g.stat_density_3d = self.stat_density_3d
        # Grammar changes invalidate a frozen payload.
        g._payload = None
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
        elif isinstance(other, (coord_3d, coord_equal)):
            g.coord = other
        elif isinstance(other, stat_density_3d):
            g.stat_density_3d = other
        elif isinstance(other, scale_x_log10):
            g.scale_x = other
        elif isinstance(other, scale_y_log10):
            g.scale_y = other
        elif isinstance(other, slider):
            if self.transition is not None:
                raise ValueError(
                    "slider() cannot be combined with transition_time() "
                    "or transition_states()"
                )
            g.slider = other
        elif isinstance(other, (transition_time, transition_states)):
            if self.slider is not None:
                raise ValueError(
                    "slider() cannot be combined with transition_time() "
                    "or transition_states()"
                )
            g.transition = other
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

    def _repr_mimebundle_(self, include=None, exclude=None):
        """HTML plus optional PlotPayload mime for CRAFT host handoff.

        Always includes ``text/html`` (iframe). When the figure can produce a
        single-panel PlotPayload, also includes
        ``application/vnd.plot3.v1+json`` so a host can render without
        re-encoding. Faceted figures only provide HTML.
        """
        from plot3.remote import MIME_PLOT3

        self._maybe_hide_from_ai()
        bundle: dict = {"text/html": self._iframe()}
        try:
            if self.facet is None and (
                self._payload is not None or self.data is not None or self.layers
            ):
                bundle[MIME_PLOT3] = self.to_payload()
        except Exception:
            # Facet / incomplete figure: HTML only.
            pass
        if include is not None:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude is not None:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle

    def _ipython_display_(self) -> None:
        """IPython entry point — browser in VS Code, iframe in SolveIt.

        On a CRAFT remote kernel (``PLOT3_REMOTE=1`` etc.), publish a mimebundle
        that includes the PlotPayload so a smart host can take the compact
        form; HTML remains the universal fallback.
        """
        self._maybe_hide_from_ai()
        try:
            from plot3.remote import MIME_PLOT3, is_remote_kernel

            if is_remote_kernel():
                from IPython.display import publish_display_data

                data = self._repr_mimebundle_()
                # Prefer publish so both HTML and payload are available.
                publish_display_data(data)
                return
        except Exception:
            pass
        self.show(browser=_prefer_external_browser())

    def to_payload(self) -> dict:
        """Serialize this figure to a PlotPayload (stats + encoded blobs).

        The payload is JSON-friendly (spec dict + base64 blobs) and does not
        retain the source DataFrame. Suitable for shipping from a remote/GPU
        kernel to a local viewer via :func:`plot3.payload.render_payload`.

        Returns a cached payload when the figure was built with
        :meth:`from_payload` or :meth:`freeze`. Faceted figures are not
        supported here; use :meth:`html` / ``build_doc``.
        """
        if self._payload is not None:
            return self._payload
        from plot3.payload import build_payload

        return build_payload(self)

    def html(self) -> str:
        """The full standalone document (what the iframe srcdoc carries)."""
        if self._payload is not None:
            from plot3.payload import render_payload

            return render_payload(self._payload, log=True)
        from plot3.build import build_doc

        return build_doc(self)

    def save(self, path: str | Path) -> str:
        path = str(path)
        doc = self.html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        print(f"plot3: saved {path} ({len(doc) // 1024} KB)")
        return path

    def _write_preview(
        self, path: str | Path | None = None, doc: str | None = None
    ) -> Path:
        if path is None:
            out_dir = Path.cwd() / ".plot3_preview"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "latest.html"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        if doc is None:
            doc = self.html()
        path.write_text(doc, encoding="utf-8")
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
        doc = self.html()

        if browser:
            out = self._write_preview(path, doc)
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

            display(HTML(self._iframe(doc)))
            # Restricted hosts (VS Code) often show a blank panel: keep a file
            # fallback so the figure is never lost when auto-detect misses.
            # The hint itself is once per session, and skipped when the user
            # already asked for an iframe.
            if not _skip_blank_hint():
                out = self._write_preview(path, doc)
                global _BLANK_HINT_SHOWN
                if not _BLANK_HINT_SHOWN:
                    _BLANK_HINT_SHOWN = True
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
            out = self._write_preview(path, doc)
            webbrowser.open(out.as_uri())
            return out
        return None

    def _iframe(self, doc: str | None = None) -> str:
        if doc is None:
            doc = self.html()
        if self._payload is not None:
            spec = self._payload.get("spec") or {}
            title = (spec.get("labs") or {}).get("title") or "plot3 figure"
            surface = (spec.get("theme") or {}).get("surface") or _THEMES[
                self.theme_name
            ]["surface"]
        else:
            title = self.labs.get("title", "plot3 figure")
            surface = _THEMES[self.theme_name]["surface"]
        # sandbox must allow scripts or the three.js viewer never starts.
        # A wrapping div keeps the string from starting with ``<iframe``,
        # which is what makes IPython suggest display.IFrame.
        return (
            f'<div class="plot3-fig"><iframe srcdoc="{_htmlesc.escape(doc, quote=True)}" '
            f'sandbox="allow-scripts allow-same-origin allow-pointer-lock" '
            f'allow="fullscreen" '
            f'style="width:100%;height:{self.height};border:0;'
            f'border-radius:6px;background:{surface}" '
            f'title="{_htmlesc.escape(str(title))}"></iframe></div>'
        )


AUTOHIDE = True
_BLANK_HINT_SHOWN = False


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
