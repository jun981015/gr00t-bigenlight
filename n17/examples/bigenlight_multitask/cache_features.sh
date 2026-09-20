#!/usr/bin/env bash
# Shares the allocated GPU deliberately, without touching existing learner PIDs.
set -eo pipefail
variant="${1:-both}"
if [[ $# -gt 0 ]]; then shift; fi
case "$variant" in
  both) variants=(all 50per-task) ;;
  all|50per-task) variants=("$variant") ;;
  *) echo 'Usage: cache_features.sh {all|50per-task|both} [--execute]' >&2; exit 2 ;;
esac
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --execute ) ]]; then exit 2; fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/activate_gr00t.sh"
export CUDA_VISIBLE_DEVICES="${CACHE_GPU:-0}"
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CACHE_CPU_THREADS:-2}"
if [[ "${1:-}" == --execute ]]; then
  test -f /.dockerenv || { echo 'Execute inside the allocated container.' >&2; exit 1; }
  mkdir -p "$VLA_STORAGE_ROOT/features"
  exec 9>"$VLA_STORAGE_ROOT/features/n17-extraction-gpu-${CACHE_GPU:-0}.lock"
  flock -n 9 || { echo 'Another feature extraction owns this GPU cache lock.' >&2; exit 1; }
fi
for item in "${variants[@]}"; do
  suffix=; if [[ "$item" == 50per-task ]]; then suffix=_50per_task; fi
  command=(python -m gr00t.rl.cache_features
    --model-path "$VLA_STORAGE_ROOT/outputs/bigenlight-n17-${item}-b32-10k-20260920T100453/checkpoint-10000"
    --dataset-path "$VLA_STORAGE_ROOT/datasets/bigenlight_multitask_gr00t${suffix}/n17"
    --output-dir "$VLA_STORAGE_ROOT/features/bigenlight-n17-${item}-bc10000-h16-v1"
    --horizon 16 --batch-size "${CACHE_BATCH_SIZE:-8}"
    --cpu-threads "${CACHE_CPU_THREADS:-2}" --sleep-ms "${CACHE_SLEEP_MS:-25}" --device cuda:0)
  printf '%q ' "${command[@]}"; printf '\n'
  if [[ "${1:-}" == --execute ]]; then "${command[@]}"; fi
done
