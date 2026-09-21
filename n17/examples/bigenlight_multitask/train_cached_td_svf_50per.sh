#!/usr/bin/env bash
# Launch inside an allocated GPU container. No existing job is stopped.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$REPO_ROOT/environments/activate_n17.sh"
export WANDB_MODE=online WANDB_ENTITY=junhyeong
STORAGE_ROOT="${SVF_STORAGE_ROOT:-/raid/yoon/vla_finetune}"
RUN_OUTPUT="${SVF_OUTPUT_DIR:-$STORAGE_ROOT/outputs/bigenlight-n17-cached-svf-td-critic-50per-lora16-b32-flow4-10k-$(date +%Y%m%dT%H%M%S)}"
exec python -u -m gr00t.rl.train_cached_svf \
  --pooled-cache "$STORAGE_ROOT/features/bigenlight-n17-50per-task-bc10000-h16-v1" \
  --actor-cache "$STORAGE_ROOT/features/bigenlight-n17-50per-task-bc10000-h16-projected-v1" \
  --output-dir "$RUN_OUTPUT" \
  --env-q td --reward step-cost --gamma 0.99 \
  --steps 10000 --batch-size 32 --learning-rate 3e-4 \
  --dit-lora-rank 16 --dit-lora-alpha 32 \
  --flow-steps 4 --candidates 8 --q-aggregation min \
  --kappa "${SVF_KAPPA:-1.0}" --g "${SVF_G:-1.0}" \
  --save-every 5000 --log-every 50 \
  --wandb-project bigenlight-multitask-gr00t --device cuda:0
