"""CRAFT remote seeding — ship the local plot3 source to the remote kernel.

Mirrors ``tidy3.craft``: tar the package, bootstrap on the remote via
``remote_run_``, install pandas if needed, register the plot3 IPython
extension. Idempotent via content stamp.
"""

from __future__ import annotations

import base64
import hashlib
import io
import tarfile
import time
from pathlib import Path
from typing import Callable

__all__ = ["build_payload", "bootstrap_code", "seed"]

_OK_PREFIX = "plot3 remote: OK"


def _pkg_files() -> list[tuple[str, bytes]]:
    pkg = Path(__file__).resolve().parent
    files: list[tuple[str, bytes]] = []
    for path in sorted(pkg.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".typed"} and path.name != "py.typed":
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(pkg.parent).as_posix()  # plot3/...
        files.append((rel, path.read_bytes()))
    return files


def build_payload() -> tuple[str, str]:
    """Return (base64 tar.gz of the package, content stamp)."""
    from plot3 import __version__

    files = _pkg_files()
    h = hashlib.sha256()
    for name, data in files:
        h.update(name.encode())
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
    stamp = f"{__version__}-{h.hexdigest()[:16]}"

    buf = io.BytesIO()
    now = int(time.time())
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = now
            tar.addfile(ti, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode("ascii"), stamp


_BOOTSTRAP = r'''
import base64 as _b64, io as _io, sys as _sys, tarfile as _tarfile
from pathlib import Path as _Path
_root = _Path.home() / ".plot3-src"
_stampf = _root / ".stamp"
_stamp = "%(stamp)s"
try:
    _fresh = _stampf.read_text().strip() == _stamp
except Exception:
    _fresh = False
if not _fresh:
    import shutil as _shutil
    _root.mkdir(parents=True, exist_ok=True)
    _shutil.rmtree(_root / "plot3", ignore_errors=True)
    _buf = _io.BytesIO(_b64.b64decode("%(payload)s"))
    with _tarfile.open(fileobj=_buf, mode="r:gz") as _tar:
        try:
            _tar.extractall(_root, filter="data")
        except TypeError:
            _tar.extractall(_root)
    _stampf.write_text(_stamp)
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
try:
    import pandas as _pd  # noqa: F401
except Exception:
    import subprocess as _sp
    print("plot3 remote: installing pandas (first time only)...", flush=True)
    _r = _sp.run(["uv", "pip", "install", "pandas"], capture_output=True, text=True)
    if _r.returncode != 0:
        _r = _sp.run([_sys.executable, "-m", "pip", "install", "pandas"],
                     capture_output=True, text=True)
    if _r.returncode != 0:
        print((_r.stdout or "")[-600:])
        print((_r.stderr or "")[-600:])
        raise RuntimeError("plot3 seed: pandas install failed")
    import importlib as _il
    _il.invalidate_caches()
if not _fresh and "plot3" in _sys.modules:
    for _k in [_m for _m in list(_sys.modules)
               if _m == "plot3" or _m.startswith("plot3.")]:
        del _sys.modules[_k]
import plot3 as _p3
from IPython import get_ipython as _gi
_ip = _gi()
if _ip is not None:
    from plot3.jupyter import register_plot3 as _reg
    _reg(quiet=True)
print("plot3 remote: OK v" + _p3.__version__ + " (" + _stamp + ")")
'''


def bootstrap_code(payload: str, stamp: str) -> str:
    return _BOOTSTRAP % {"payload": payload, "stamp": stamp}


def seed(
    remote_run: Callable[..., str],
    *,
    payload: str | None = None,
    stamp: str | None = None,
    max_chars: int = 12000,
) -> tuple[bool, str]:
    """Run the bootstrap via CRAFT's ``remote_run_``. Returns (ok, message)."""
    if payload is None or stamp is None:
        payload, stamp = build_payload()
    code = bootstrap_code(payload, stamp)
    try:
        out = remote_run(code, max_chars=max_chars) or ""
    except Exception as e:
        return False, f"remote bootstrap did not run: {e}"
    out = out.strip()
    for line in out.splitlines():
        if line.startswith(_OK_PREFIX):
            return True, line.strip()
    return False, (out[-1500:] if out else "no output from remote bootstrap")
