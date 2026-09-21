#!/usr/bin/env bash
# Resume the 50/task and all-data caches, never overwrite another model's cache.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/environments/activate_n15.sh"
cd "$GR00T_N15_ROOT"
export CUDA_VISIBLE_DEVICES="${CACHE_GPU:-0}" OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
test -f /.dockerenv
for variant in 50per-task all; do
  if [[ "$variant" == 50per-task ]]; then
    dataset="$VLA_STORAGE_ROOT/datasets/bigenlight_multitask_gr00t_50per_task/n15"
  else
    dataset="$VLA_STORAGE_ROOT/datasets/bigenlight_multitask_gr00t/n15"
  fi
  python -m experiments.bigenlight_multitask.cache_features \
    --model-path "$VLA_STORAGE_ROOT/outputs/bigenlight-n15-${variant}-b32-10k-20260920T162613/checkpoint-10000" \
    --dataset-path "$dataset" \
    --output-dir "$VLA_STORAGE_ROOT/features/bigenlight-n15-${variant}-bc10000-h16-v1" \
    --batch-size 32
done
