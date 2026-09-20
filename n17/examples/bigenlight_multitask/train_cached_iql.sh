#!/usr/bin/env bash
# Preparation only unless --execute is explicit. Never changes an existing run.
set -eo pipefail
variant="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$variant" in all|50per-task) ;; *) exit 2 ;; esac
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --execute ) ]]; then exit 2; fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/activate_gr00t.sh"
export CUDA_VISIBLE_DEVICES="${IQL_GPU:-0}" WANDB_MODE=online
reward="${IQL_REWARD:-terminal-success}"
output="${IQL_OUTPUT:-$VLA_STORAGE_ROOT/outputs/bigenlight-n17-cached-iql-${variant}-${reward}-$(date +%Y%m%dT%H%M%S)}"
command=(python -m gr00t.rl.train_cached
  --cache "$VLA_STORAGE_ROOT/features/bigenlight-n17-${variant}-bc10000-h16-v1"
  --output-dir "$output" --reward "$reward" --steps 10000 --batch-size 32
  --device cuda:0 --wandb-project bigenlight-multitask-gr00t --log-every 50)
if [[ -n "${IQL_RESUME:-}" ]]; then command+=(--resume "$IQL_RESUME"); fi
printf '%q ' "${command[@]}"; printf '\n'
if [[ "${1:-}" != --execute ]]; then exit 0; fi
test -f /.dockerenv
uuid=$(nvidia-smi -i "${IQL_GPU:-0}" --query-gpu=uuid --format=csv,noheader)
python "$PYTHONPATH/examples/carrot_in_pot/gpu_guard.py" --gpu-uuid "$uuid"
exec "${command[@]}"
