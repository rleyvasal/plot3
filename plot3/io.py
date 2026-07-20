"""Point-cloud IO and optional CRAFT SSH helpers."""

from __future__ import annotations

import shlex
import subprocess

import numpy as np
import pandas as pd

try:
    from IPython import get_ipython
except Exception:  # pragma: no cover
    get_ipython = None

def ssh_cfg():
    """CRAFT's SSH config from the IPython user namespace (optional)."""
    nss = [globals()]
    if get_ipython is not None:
        try:
            nss.append(get_ipython().user_ns)
        except Exception:
            pass
    for ns in nss:
        if isinstance(ns, dict) and ns.get("SSH_HOST"):
            return ns["SSH_HOST"], ns.get("SSH_OPTS", "")
    raise RuntimeError("SSH_HOST not found — load CRAFT and run %gpu first.")


def ssh_bytes(remote_cmd: str) -> bytes:
    host, opts = ssh_cfg()
    cmd = ["ssh", *shlex.split(opts or ""), host, remote_cmd]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ssh failed (rc={proc.returncode}): {err[:300]}")
    return proc.stdout or b""


def read_bin(path, *, stride=5, columns=None, remote=False, sub=1,
             max_points=500_000) -> pd.DataFrame:
    """Load an (N, stride) float32 point-cloud file as a DataFrame.

    columns: names for the stride columns (default x,y,z,intensity,c4,...).
    remote=True streams (and thins on) the CRAFT GPU host over SSH.
    """
    stride = max(3, int(stride))
    sub = max(1, int(sub or 1))
    if remote:
        thin = (
            "python3 -c " + shlex.quote(
                "import sys,numpy as np;"
                f"a=np.fromfile({str(path)!r},dtype=np.float32);"
                f"s={stride};n=a.size//s;a=a[:n*s].reshape(n,s);"
                f"m={int(max_points)};sb={sub};"
                "sb=max(sb,(n+m-1)//m) if m>0 else sb;"
                "sys.stdout.buffer.write("
                "np.ascontiguousarray(a[::sb],dtype=np.float32).tobytes())"
            )
        )
        raw = ssh_bytes(thin)
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        arr = np.fromfile(str(path), dtype=np.float32)
    n = arr.size // stride
    arr = arr[: n * stride].reshape(n, stride)
    if not remote:
        if max_points and n > max_points * sub:
            sub = max(sub, (n + max_points - 1) // max_points)
        if sub > 1:
            arr = np.ascontiguousarray(arr[::sub])
    if columns is None:
        base = ["x", "y", "z", "intensity"]
        columns = (base + [f"c{i}" for i in range(4, stride)])[:stride]
    return pd.DataFrame(arr, columns=list(columns)[:stride])

