#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PYTHON_BIN="/Users/feizhanxia/miniforge3/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

WORKERS="${WORKERS:-50}"
N_SAMPLES="${N_SAMPLES:-1500}"
COARSE_SELECTIVITY_THRESHOLD="${COARSE_SELECTIVITY_THRESHOLD:-3.0}"
RNG_SEED="${RNG_SEED:-0}"
TARGET_OMEGA="${TARGET_OMEGA:-1.5}"
TARGET_HALF_WIDTH="${TARGET_HALF_WIDTH:-0.2}"

"$PYTHON_BIN" scripts/run_strict_search.py \
  --workers "$WORKERS" \
  --n_samples "$N_SAMPLES" \
  --coarse_selectivity_threshold "$COARSE_SELECTIVITY_THRESHOLD" \
  --rng_seed "$RNG_SEED" \
  --use_target_band \
  --target_omega "$TARGET_OMEGA" \
  --target_half_width "$TARGET_HALF_WIDTH"
"$PYTHON_BIN" scripts/analyze_strict_results.py
"$PYTHON_BIN" scripts/plot_strict_results.py

echo "Done."
