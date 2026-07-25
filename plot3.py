"""Standalone SolveIt / Jupyter loader — one command for local and CRAFT/GPU.

Usage::

    %run /app/data/gpudevd/plot3/plot3.py
    %run /path/to/plot3/plot3.py
    %run /path/to/plot3/load.py          # same loader

What this does (always):

1. Puts the package root on ``sys.path`` (no pip install required)
2. Fresh-imports plot3 so a ``git pull`` takes effect
3. Registers ``%plot3``, injects ``ggplot`` / ``aes`` / geoms into user_ns
4. Enables R-style bare names / backticks in ``aes`` / ``facet_wrap``

What this does **only when CRAFT is present** (auto-detected):

5. Registers remote seed hooks so ``%gpu`` cells get plot3 without a second command
6. Exposes ``seed_plot3_remote(force=True)`` for kernel restarts

You do **not** need different commands for local vs GPU. Load once; CRAFT is
optional and is detected from the environment (``remote_run_``, ``_exec_mgr``,
or ``gpudev_craft``).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

print("plot3: loader starting…", flush=True)

# %run must execute this file as a script, not import it as package "plot3".
if __name__ == "plot3":  # pragma: no cover
    raise ImportError(
        "plot3.py was imported as module 'plot3' (sys.path shadowing). "
        "Load it with %run /path/to/plot3/plot3.py — the package lives in plot3/."
    )

_ROOT = Path(__file__).resolve().parent
_pkg_dir = str(_ROOT)
while _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)
sys.path.insert(0, _pkg_dir)
print(f"plot3: root={_ROOT}", flush=True)

try:
    from IPython import get_ipython
except Exception:  # pragma: no cover
    get_ipython = None

# Drop stale modules so re-%run / git pull is picked up without kernel restart.
for _m in [m for m in list(sys.modules) if m == "plot3" or m.startswith("plot3.")]:
    del sys.modules[_m]

try:
    import plot3
    from plot3.jupyter import register_plot3
except Exception:
    print("plot3: FAILED to import package:", flush=True)
    traceback.print_exc()
    raise

print(
    f"plot3: imported v{getattr(plot3, '__version__', '?')} "
    f"from {Path(plot3.__file__).resolve()}",
    flush=True,
)

_PUBLIC = {
    name: getattr(plot3, name)
    for name in getattr(plot3, "__all__", [])
    if name != "load_ipython_extension" and hasattr(plot3, name)
}
_PUBLIC["plot3"] = plot3

# ── CRAFT / GPU detection (optional) ─────────────────────────────────────────

_SEED_STATE = {"stamp": None, "kc_id": None, "ok": False}


def _craft_status(ip=None) -> str:
    """Return ``connected`` | ``present`` | ``absent``.

    * connected — remote_run_ + exec manager ready (can seed now)
    * present   — CRAFT machinery importable or partially in ns (seed on %gpu)
    * absent    — pure local / no CRAFT
    """
    if ip is None:
        try:
            ip = get_ipython() if get_ipython else None
        except Exception:
            ip = None
    ns = (getattr(ip, "user_ns", None) or {}) if ip is not None else {}
    rr = ns.get("remote_run_")
    mgr = ns.get("_exec_mgr")
    if callable(rr) and mgr is not None:
        return "connected"
    if "remote_run_" in ns or "register_local_magic" in ns:
        return "present"
    try:
        import gpudev_craft  # noqa: F401

        return "present"
    except Exception:
        pass
    # CRAFT.py often leaves these markers after %run even before %gpu
    for key in ("_craft_cfg", "CRAFT", "remote_run", "gpu_mode"):
        if key in ns:
            return "present"
    return "absent"


def seed_remote(*, force: bool = False, quiet: bool = False) -> bool:
    """Ship plot3 source to the CRAFT remote kernel (no-op without CRAFT)."""
    try:
        from plot3 import craft
    except ImportError:
        if not quiet:
            print(
                "plot3: remote seed unavailable (no craft module in this build)",
                flush=True,
            )
        return False

    ip = get_ipython() if get_ipython else None
    if ip is None:
        return False
    ns = ip.user_ns or {}
    rr = ns.get("remote_run_")
    mgr = ns.get("_exec_mgr")
    if not callable(rr) or mgr is None:
        if not quiet:
            print(
                "plot3: CRAFT not connected yet — local only "
                "(will seed automatically on first %gpu cell)",
                flush=True,
            )
        return False

    payload, stamp = craft.build_payload()
    kc_id = id(getattr(mgr, "remote_kc", None))
    if (
        not force
        and _SEED_STATE["stamp"] == stamp
        and _SEED_STATE["kc_id"] == kc_id
    ):
        return _SEED_STATE["ok"]

    ok, msg = craft.seed(rr, payload=payload, stamp=stamp)
    _SEED_STATE.update(stamp=stamp, kc_id=kc_id, ok=ok)
    if ok:
        if not quiet:
            print(f"plot3: {msg}", flush=True)
    else:
        print(
            "plot3: remote seed FAILED — remote ggplot cells won't work.\n"
            + msg
            + "\nRetry with seed_plot3_remote(force=True). "
            "Local %plot3 / host figures still work.",
            flush=True,
        )
    return ok


def _maybe_seed_on_cell(_info=None):
    """Before each cell: if CRAFT is in %gpu Python mode, seed plot3."""
    try:
        import gpudev_craft.core as _core

        router = getattr(_core, "ROUTER", None)
        py_be = getattr(_core, "PY_BACKEND", None)
        if router is None or py_be is None or router.backend is not py_be:
            return
    except Exception:
        return
    seed_remote(quiet=True)


ip = get_ipython() if get_ipython else None
_craft = _craft_status(ip)
print(f"plot3: environment = {_craft} (local setup always runs)", flush=True)

if ip is None:
    print(
        "plot3: WARNING — get_ipython() is None; run with %run inside "
        "IPython / SolveIt / Jupyter for magics.\n"
        "  Plain import still works: from plot3 import ggplot, aes, geom_point",
        flush=True,
    )
elif getattr(ip, "user_ns", None) is None:
    print("plot3: WARNING — no user_ns on IPython shell", flush=True)
else:
    register_plot3(quiet=False, r_style=True)
    ip.user_ns.update(_PUBLIC)
    ip.user_ns["seed_plot3_remote"] = seed_remote
    print(f"plot3: injected {len(_PUBLIC)} names into user_ns", flush=True)

    # Host-local magic under %gpu (viewer stays on the dialog machine)
    try:
        reg = ip.user_ns.get("register_local_magic")
        if callable(reg):
            reg("%plot3")
    except Exception:
        pass

    if _craft != "absent":
        try:
            prev = ip.user_ns.get("_plot3_seed_cb")
            if prev is not None:
                try:
                    ip.events.unregister("pre_run_cell", prev)
                except Exception:
                    pass
            ip.events.register("pre_run_cell", _maybe_seed_on_cell)
            ip.user_ns["_plot3_seed_cb"] = _maybe_seed_on_cell
        except Exception:
            pass
        seed_remote(quiet=False)
    else:
        print(
            "plot3: no CRAFT detected — local only (fine for VS Code / SolveIt without %gpu)",
            flush=True,
        )

print(
    f"plot3 {plot3.__version__} ready "
    f"from {Path(plot3.__file__).resolve().parent}",
    flush=True,
)
print(
    "  ggplot(df, aes(x=wt, y=mpg)) + geom_point()   # bare names in Jupyter\n"
    "  aes(x=`First Name`, y=mpg)                    # backticks for spaces\n"
    "  %plot3 df x=a y=b [z=c] [color=d]\n"
    "  GPU: same %run; seeds remote when CRAFT is connected "
    "(seed_plot3_remote(force=True) after kernel restart)",
    flush=True,
)
