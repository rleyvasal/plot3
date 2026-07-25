"""Alias entrypoint — same as ``plot3.py``.

::

    %run /path/to/plot3/load.py
    %run /path/to/plot3/plot3.py
"""

from pathlib import Path

_loader = Path(__file__).resolve().parent / "plot3.py"
exec(compile(_loader.read_text(encoding="utf-8"), str(_loader), "exec"), globals())
