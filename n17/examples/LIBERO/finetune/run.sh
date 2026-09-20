#!/usr/bin/env bash
set -eo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)/activate_gr00t.sh"
cd "$GR00T_N17_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
# Small per-episode files otherwise exhaust the shared Xet token API quota.
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [ "${1:-}" = prepare-qvgm ]; then
    shift
    exec python -m examples.LIBERO.finetune.prepare_qvgm "$@"
fi
if [ "${1:-}" = data ]; then
    shift
    exec python -m examples.LIBERO.finetune.data "$@"
fi
exec python -m examples.LIBERO.finetune.launch "$@"
