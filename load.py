"""CRAFT / SolveIt entrypoint (replaces ``%run plot3.py``).

Usage::

    %run /path/to/plot3/load.py

Or via the gpudev addon (preferred under SolveIt)::

    %run /path/to/gpudev/addons/plot3.py

Prefer a normal install when possible::

    pip install -e /path/to/plot3
    %load_ext plot3
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow %run from a source checkout without prior pip install.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Drop stale modules so re-%run picks up git pulls.
for _m in [m for m in list(sys.modules) if m == "plot3" or m.startswith("plot3.")]:
    del sys.modules[_m]

from plot3.jupyter import register_plot3  # noqa: E402

register_plot3(quiet=False)
