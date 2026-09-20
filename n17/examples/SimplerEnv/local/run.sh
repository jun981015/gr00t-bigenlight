#!/usr/bin/env bash
set -euo pipefail
SIMPLER_HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIMPLER_ROOT="$(cd -- "$SIMPLER_HERE/../../.." && pwd)"
source "$SIMPLER_ROOT/../activate_gr00t.sh"
export PYTHONPATH="$SIMPLER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MS2_REAL2SIM_ASSET_DIR="$SIMPLER_ROOT/external_dependencies/SimplerEnv/ManiSkill2_real2sim/data"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
cd "$SIMPLER_ROOT"
# manage.py uses only stdlib and dispatches to the correct isolated interpreter.
exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$SIMPLER_HERE/manage.py" "$@"
