#!/usr/bin/env bash
# Local full check: unit tests + smoke HTML artifacts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "== pytest =="
python -m pytest -q

echo "== smoke HTML =="
python tests/local/smoke_local.py

echo "== optional R oracle =="
if command -v Rscript >/dev/null 2>&1; then
  python -m pytest -q tests/test_r_oracle_parity.py
else
  echo "(skip: Rscript not found)"
fi

echo "all local checks finished."
