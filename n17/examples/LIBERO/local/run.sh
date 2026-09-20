#!/usr/bin/env bash
set -euo pipefail
LIBERO_HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_ROOT="$(cd -- "$LIBERO_HERE/../../.." && pwd)"
source "$LIBERO_ROOT/../activate_gr00t.sh"
export PYTHONPATH="$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$VLA_STORAGE_ROOT/config/libero-n17"
export NUMBA_CACHE_DIR="$VLA_STORAGE_ROOT/cache/libero-numba"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR"
cd "$LIBERO_ROOT"
exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$LIBERO_HERE/manage.py" "$@"
