#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PYTHON_BIN="/Users/feizhanxia/miniforge3/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

"$PYTHON_BIN" scripts/run_strict_search.py --use_target_band --target_omega 1.5 --target_half_width 0.2
"$PYTHON_BIN" scripts/plot_strict_results.py

echo "Done."
