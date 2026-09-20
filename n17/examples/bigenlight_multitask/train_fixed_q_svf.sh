#!/usr/bin/env bash
# Dry run by default. Does NOT change or queue work behind current IQL jobs.
set -eo pipefail
variant="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$variant" in all|50per-task) ;; *) exit 2 ;; esac
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --execute ) ]]; then exit 2; fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/activate_gr00t.sh"
export CUDA_VISIBLE_DEVICES="${SVF_GPU:-0}" WANDB_MODE=online
suffix=; if [[ "$variant" == 50per-task ]]; then suffix=_50per_task; fi
q="${SVF_Q_CHECKPOINT:-$VLA_STORAGE_ROOT/outputs/bigenlight-n17-cached-iql-${variant}-step-cost-20260920/checkpoints/step-10000.pt}"
command=(python -m gr00t.rl.train --backend gr00t --algorithm svf
  --model-path "$VLA_STORAGE_ROOT/outputs/bigenlight-n17-${variant}-b32-10k-20260920T100453/checkpoint-10000"
  --dataset-path "$VLA_STORAGE_ROOT/datasets/bigenlight_multitask_gr00t${suffix}/n17"
  --fixed-iql-checkpoint "$q"
  --fixed-iql-cache "$VLA_STORAGE_ROOT/features/bigenlight-n17-${variant}-bc10000-h16-v1"
  --output-dir "${SVF_OUTPUT:-$VLA_STORAGE_ROOT/outputs/bigenlight-n17-fixed-q-svf-${variant}-$(date +%Y%m%dT%H%M%S)}"
  --annotation-format all-success-step-cost --horizon 16 --gamma 0.99
  --q-aggregation min --freeze-reference --batch-size "${SVF_BATCH_SIZE:-2}"
  --steps "${SVF_STEPS:-10000}" --device cuda:0 --loader-workers 4
  --episode-cache-count 16 --episode-cache-gib 2 --decoder-threads 2
  --wandb-project bigenlight-multitask-gr00t --wandb-log-every 50
  --save-every 5000 --first-save-step 100 --save-interval-seconds 1800
  --max-run-seconds 28800 --keep-latest-training-state)
if [[ -n "${SVF_RESUME:-}" ]]; then command+=(--resume "$SVF_RESUME"); fi
printf '%q ' "${command[@]}"; printf '\n'
if [[ "${1:-}" != --execute ]]; then exit 0; fi
test -f /.dockerenv
test -f "$q"
uuid=$(nvidia-smi -i "${SVF_GPU:-0}" --query-gpu=uuid --format=csv,noheader)
python "$PYTHONPATH/examples/carrot_in_pot/gpu_guard.py" --gpu-uuid "$uuid"
exec "${command[@]}"
