"""IPython/SolveIt integration: hide-from-AI, %plot3, R-style aes, extension load."""

from __future__ import annotations

import json
import shlex
from typing import Any

import numpy as np
import pandas as pd

try:
    from IPython import get_ipython
except Exception:  # pragma: no cover
    get_ipython = None

from plot3.io import ssh_bytes
from plot3.masking import (
    BT_NAME,
    Plot3MaskTransformer,
    is_backtick_transformer,
    is_mask_transformer,
    is_tidy3_backtick_transformer,
    plot3_backtick_transform,
)

# R-style bare-name / backtick masking for aes() / facet_wrap().
_R_STYLE_ON = True


def _bt_fallback(name: str) -> str:
    """If a backtick sentinel escapes AST rewrite, treat it as a column name."""
    return str(name)


def enable_r_style(ipython: Any | None = None) -> bool:
    """Enable bare-name + backtick masking for ``aes`` / ``facet_wrap``.

    Jupyter / SolveIt only. After this::

        aes(x=wt, y=mpg, colour=cyl)
        aes(x=`First Name`, y=`Age (%)`)
        facet_wrap(cyl)

    become string column names, matching ggplot2 / tidy3 conventions.
    """
    global _R_STYLE_ON
    _R_STYLE_ON = True
    if ipython is None:
        try:
            ipython = get_ipython() if get_ipython is not None else None
        except Exception:
            ipython = None
    if ipython is None:
        return False

    ns = getattr(ipython, "user_ns", None)
    if ns is not None:
        # Escape hatch if AST missed a sentinel (still a valid column string).
        ns[BT_NAME] = _bt_fallback
        # When tidy3 is not loaded, leave tidy3 sentinel unset; plot3 AST
        # still rewrites __tidy3_bt__ if tidy3's preparser ran first.

    # Source preparser: only install plot3's if tidy3 did not already provide one.
    for attr in ("input_transformers_cleanup", "input_transformers_post"):
        transformers = getattr(ipython, attr, None)
        if not isinstance(transformers, list):
            continue
        has_tidy3_bt = any(is_tidy3_backtick_transformer(t) for t in transformers)
        transformers[:] = [
            t for t in transformers if not is_backtick_transformer(t)
        ]
        if not has_tidy3_bt:
            transformers.insert(0, plot3_backtick_transform)

    # AST pass for bare names inside aes / facet_wrap.
    ast_transformers = getattr(ipython, "ast_transformers", None)
    if isinstance(ast_transformers, list):
        ast_transformers[:] = [
            t for t in ast_transformers if not is_mask_transformer(t)
        ]
        ast_transformers.append(Plot3MaskTransformer())
    return True


def disable_r_style(ipython: Any | None = None) -> None:
    """Disable bare-name / backtick masking for plot3 aesthetics."""
    global _R_STYLE_ON
    _R_STYLE_ON = False
    if ipython is None:
        try:
            ipython = get_ipython() if get_ipython is not None else None
        except Exception:
            ipython = None
    if ipython is None:
        return
    for attr in ("input_transformers_cleanup", "input_transformers_post"):
        transformers = getattr(ipython, attr, None)
        if isinstance(transformers, list):
            transformers[:] = [
                t for t in transformers if not is_backtick_transformer(t)
            ]
    ast_transformers = getattr(ipython, "ast_transformers", None)
    if isinstance(ast_transformers, list):
        ast_transformers[:] = [
            t for t in ast_transformers if not is_mask_transformer(t)
        ]


def r_style_enabled() -> bool:
    return _R_STYLE_ON

def find_caller_msg_id():
    import inspect

    frame = inspect.currentframe()
    try:
        f = frame.f_back if frame is not None else None
        while f is not None:
            for ns in (f.f_locals, f.f_globals):
                mid = ns.get("__msg_id") if isinstance(ns, dict) else None
                if mid:
                    return str(mid)
            f = f.f_back
    finally:
        del frame
    try:
        ip = get_ipython()
        for ns_name in ("user_ns", "user_global_ns"):
            ns = getattr(ip, ns_name, None) or {}
            mid = ns.get("__msg_id") if isinstance(ns, dict) else None
            if mid:
                return str(mid)
    except Exception:
        pass
    try:
        from safepyrun import find_var  # type: ignore

        mid = find_var("__msg_id")
        if mid:
            return str(mid)
    except Exception:
        pass
    return None


def hide_caller_from_ai(mid=None):
    """Best-effort ``skipped=1`` on the calling cell; no-op outside SolveIt."""
    try:
        from dialoghelper.core import update_msg
    except Exception:
        return

    async def _run():
        import inspect

        m = mid or find_caller_msg_id()
        if not m:
            try:
                from dialoghelper.core import read_msg

                msg = read_msg(n=0, relative=True)
                if inspect.iscoroutine(msg):
                    msg = await msg
                m = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
            except Exception:
                m = None
        if not m:
            try:
                from dialoghelper.core import find_msgs

                msgs = find_msgs(msg_type="code", re_pattern=r"%plot3",
                                 include_output=False, include_meta=True,
                                 include_skipped=True, use_regex=True)
                if inspect.iscoroutine(msgs):
                    msgs = await msgs
                for msg in msgs or []:
                    m = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
            except Exception:
                m = None
        if not m:
            print("plot3: hide-from-ai failed — could not resolve msg id "
                  "(pass hide=0 to silence)")
            return
        m = str(m)
        err = None
        for cand in (m, m[1:] if m.startswith("_") else "_" + m):
            try:
                res = update_msg(id=cand, skipped=1)
                if inspect.iscoroutine(res):
                    await res
                return
            except Exception as e:
                err = e
        print(f"plot3: hide-from-ai failed — update_msg({m}): {err}")

    try:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            asyncio.run(_run())
            return
        try:
            import nest_asyncio

            nest_asyncio.apply()
            loop.run_until_complete(_run())
        except Exception:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(lambda: asyncio.run(_run())).result()
    except Exception as e:
        print(f"plot3: hide-from-ai failed — {e}")


# ═════════════════════════════════════════════════════════════════════════════
# %plot3 magic — remote (CRAFT) or local DataFrame snapshot -> figure
# ═════════════════════════════════════════════════════════════════════════════


def remote_df(expr: str, cols: dict, max_points: int) -> pd.DataFrame:
    """Snapshot mapped columns of a remote DataFrame over CRAFT's SSH pipe."""
    import io
    import uuid

    ip = get_ipython()
    rr = (ip.user_ns or {}).get("remote_run_") if ip is not None else None
    if not callable(rr):
        raise RuntimeError("remote_run_ missing — load CRAFT and run %gpu")
    tmp = f"/tmp/plot3_{uuid.uuid4().hex}.npz"
    code = f"""
import numpy as _np, pandas as _pd, json as _json
_df = eval({expr!r})
_m = {int(max_points)}
if _m > 0 and len(_df) > _m:
    _df = _df.iloc[::(len(_df) + _m - 1) // _m]
_out, _meta = {{}}, {{}}
for _a, _c in {cols!r}.items():
    _s = _df[_c]
    if str(_s.dtype).startswith("datetime"):
        _out[_a] = _s.astype("datetime64[ns]").astype("int64").to_numpy()
        _meta[_a] = ["dt", _c]
    elif _s.dtype.kind in "ifub":
        _out[_a] = _s.to_numpy(_np.float32); _meta[_a] = ["num", _c]
    else:
        _codes, _cats = _pd.factorize(_s.astype(str))
        _out[_a] = _codes.astype(_np.int32)
        _meta[_a] = ["cat", _c, [str(x) for x in _cats]]
_np.savez({tmp!r}, **_out)
print(_json.dumps(_meta))
"""
    out = rr(code, max_chars=8000).strip()
    meta = json.loads(out.splitlines()[-1])
    try:
        raw = ssh_bytes("cat -- " + shlex.quote(tmp))
    finally:
        try:
            ssh_bytes("rm -f -- " + shlex.quote(tmp))
        except Exception:
            pass
    z = np.load(io.BytesIO(raw))
    data = {}
    for a, m in meta.items():
        v = z[a]
        col = m[1]
        if m[0] == "dt":
            data[col] = pd.to_datetime(v.astype("int64"), unit="ns")
        elif m[0] == "cat":
            data[col] = pd.Categorical.from_codes(
                np.clip(v, 0, len(m[2]) - 1), categories=m[2])
        else:
            data[col] = v
    return pd.DataFrame(data)


def run_plot3_from_magic(line: str = ""):
    parts = shlex.split(line or "")
    if not parts:
        raise ValueError(
            "usage: %plot3 <df_expr> x=col y=col [z=col] [color=col] "
            "[group=col] [kind=point|line|path|point+line] [size=F] "
            "[max_points=N] [theme=dark|light] [height=Npx] [hide=0|1]"
        )
    expr = parts[0]
    m: dict = {}
    kind, size, hide, theme = "point", None, True, "dark"
    max_points, height = 200_000, "480px"
    for tok in parts[1:]:
        k, _, v = tok.partition("=")
        if k in ("x", "y", "z", "color", "colour", "group"):
            m["color" if k == "colour" else k] = v
        elif k == "kind":
            kind = v
        elif k == "size":
            size = float(v)
        elif k == "max_points":
            max_points = int(v)
        elif k == "theme":
            theme = v
        elif k == "height":
            height = v if v.endswith("px") else f"{int(v)}px"
        elif k == "hide":
            hide = v.lower() in ("1", "true", "yes")
        else:
            raise ValueError(f"unknown option {tok!r}")
    if "x" not in m or "y" not in m:
        raise ValueError("%plot3 needs x= and y=")

    mid = find_caller_msg_id() if hide else None

    ip = get_ipython() if get_ipython is not None else None
    ns = (ip.user_ns or {}) if ip is not None else {}
    if callable(ns.get("remote_run_")):
        df = remote_df(expr, m, max_points)
    else:
        df = eval(expr, ns)  # local fallback (plain Jupyter)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if max_points and len(df) > max_points:
            df = df.iloc[:: (len(df) + max_points - 1) // max_points]

    from plot3.geoms import aes, geom_line, geom_path, geom_point, theme_light
    from plot3.ggplot import ggplot

    # hide=False: the magic manages the red eye itself (with its own msg id)
    fig = ggplot(df, aes(**m), height=height, hide=False)
    for part in kind.split("+"):
        part = part.strip()
        if part == "point":
            fig = fig + (geom_point(size=size) if size else geom_point())
        elif part == "line":
            fig = fig + geom_line()
        elif part == "path":
            fig = fig + geom_path()
        else:
            raise ValueError(f"unknown kind {part!r}")
    if theme != "dark":
        fig = fig + theme_light()

    # Prefer fig display path (opens system browser under VS Code; iframe in SolveIt).
    if hide:
        # show() also respects fig.hide / autohide; magic already resolved msg id.
        fig.hide = False
        fig.show()
        hide_caller_from_ai(mid)
    else:
        fig.show()
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Registration (addon contract: register everything via get_ipython)
# ═════════════════════════════════════════════════════════════════════════════


def register_plot3(*, quiet=True, r_style: bool = True) -> bool:
    if get_ipython is None:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    ok = False
    try:
        ip.magics_manager.register_function(
            run_plot3_from_magic, magic_kind="line", magic_name="plot3")
        ok = True
    except Exception as e:
        if not quiet:
            print(f"plot3: magic registration failed: {e}")
    # host-local under %gpu (CRAFT hook, when present)
    try:
        reg = (ip.user_ns or {}).get("register_local_magic")
        if callable(reg):
            reg("%plot3")
    except Exception:
        pass
    # public API into user_ns (never rely on %run leaking module globals)
    try:
        import plot3 as plot3_pkg

        ns = ip.user_ns
        for name in plot3_pkg.__all__:
            if name == "load_ipython_extension":
                continue
            ns[name] = getattr(plot3_pkg, name)
    except Exception:
        pass
    # R-style bare names / backticks in aes() (ggplot2 parity with tidy3)
    if r_style:
        try:
            enable_r_style(ip)
        except Exception as e:
            if not quiet:
                print(f"plot3: R-style masking not enabled: {e}")
    if ok and not quiet:
        print("plot3 ready")
        print("  ggplot(df, aes(x=wt, y=mpg[, colour=cyl])) + geom_point()")
        print("  aes(x=`First Name`, y=mpg)   # backticks for spaced names")
        print("  %plot3 df x=a y=b [z=c] [color=d]   read_bin(path)   ggsave()")
    return ok


def load_ipython_extension(ip=None):
    register_plot3(quiet=True)
