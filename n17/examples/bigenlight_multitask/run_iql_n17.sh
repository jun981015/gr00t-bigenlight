#!/usr/bin/env bash
# Dry-run by default. Critic only: never updates or overwrites BC checkpoints.
set -eo pipefail
variant="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$variant" in
  all) suffix= ;;
  50per-task) suffix=_50per_task ;;
  *) echo 'Usage: bash run_iql_n17.sh {all|50per-task} [--execute]' >&2; exit 2 ;;
esac
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --execute ) ]]; then
  echo 'Only --execute is accepted after the variant.' >&2; exit 2
fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/activate_gr00t.sh"
repo="$GR00T_N17_ROOT"
export PYTHONPATH="$repo"
export CUDA_VISIBLE_DEVICES="${IQL_GPU:-0}"
export WANDB_MODE=online
export NO_ALBUMENTATIONS_UPDATE=1
case "${IQL_REWARD:-terminal-success}" in
  terminal-success) annotation=all-success-terminal; reward_suffix= ;;
  step-cost) annotation=all-success-step-cost; reward_suffix=-step-cost ;;
  *) echo 'IQL_REWARD must be terminal-success or step-cost' >&2; exit 2 ;;
esac
model="${IQL_BC_MODEL:-$VLA_STORAGE_ROOT/outputs/bigenlight-n17-${variant}-b32-10k-20260920T100453/checkpoint-10000}"
output="${IQL_OUTPUT:-$VLA_STORAGE_ROOT/outputs/bigenlight-n17-iql-${variant}${reward_suffix}-$(date +%Y%m%dT%H%M%S)}"
steps="${IQL_STEPS:-10000}"
command=(python -m gr00t.rl.train
  --backend gr00t --algorithm iql --model-path "$model"
  --dataset-path "$VLA_STORAGE_ROOT/datasets/bigenlight_multitask_gr00t${suffix}/n17"
  --output-dir "$output" --embodiment-tag NEW_EMBODIMENT
  --annotation-format "$annotation" --device cuda:0
  --horizon 16 --gamma 0.99 --batch-size 32 --steps "$steps"
  --learning-rate 0.0003 --iql-expectile 0.7 --iql-target-tau 0.005
  --loader-workers 8 --loader-prefetch 2 --episode-cache-count 16
  --episode-cache-gib 4 --decoder-threads 2
  --wandb-project bigenlight-multitask-gr00t --wandb-log-every 50
  --save-every 5000 --first-save-step 100 --save-interval-seconds 1800
  --max-run-seconds 28800 --keep-latest-training-state)
if [[ -n "${IQL_RESUME:-}" ]]; then
  command+=(--resume "$IQL_RESUME")
fi
printf 'Frozen N1.7 BC; independent scalar twin-Q and V(s); GPU %s\n' "$CUDA_VISIBLE_DEVICES"
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${1:-}" != --execute ]]; then exit 0; fi
test -f /.dockerenv || { echo 'Execute inside the allocated GPU container.' >&2; exit 1; }
test -f "$model/config.json"
# A queued successor waits if the user-approved cache extractor is still using
# the GPU. Already running learners are unaffected; no process is killed here.
cache_lock="$VLA_STORAGE_ROOT/features/n17-extraction-gpu-${IQL_GPU:-0}.lock"
if [[ -f "$cache_lock" ]]; then
  flock --shared "$cache_lock" true
fi
uuid=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader)
python "$repo/examples/carrot_in_pot/gpu_guard.py" --gpu-uuid "$uuid"
cd "$repo"
exec "${command[@]}"
