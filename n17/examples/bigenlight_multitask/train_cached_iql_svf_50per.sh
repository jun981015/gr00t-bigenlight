#!/usr/bin/env bash
# General IQL Q (not DEAS), frozen; train actor LoRA and inner soft value.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$REPO_ROOT/environments/activate_n17.sh"
export WANDB_MODE=online WANDB_ENTITY=junhyeong
STORAGE_ROOT="${SVF_STORAGE_ROOT:-/raid/yoon/vla_finetune}"
RUN_OUTPUT="${SVF_OUTPUT_DIR:-$STORAGE_ROOT/outputs/bigenlight-n17-cached-svf-fixed-iql-50per-e07-q30k-lora16-b32-10k-$(date +%Y%m%dT%H%M%S)}"
exec python -u -m gr00t.rl.train_cached_svf \
  --pooled-cache "$STORAGE_ROOT/features/bigenlight-n17-50per-task-bc10000-h16-v1" \
  --actor-cache "$STORAGE_ROOT/features/bigenlight-n17-50per-task-bc10000-h16-projected-v1" \
  --output-dir "$RUN_OUTPUT" \
  --env-q fixed-iql \
  --iql-checkpoint "$STORAGE_ROOT/outputs/bigenlight-n17-cached-iql-50per-task-step-cost-kappa0.7-30k-20260921-50per-kappa-30k/checkpoints/step-30000.pt" \
  --reward step-cost --gamma 0.99 \
  --steps 10000 --batch-size 32 --learning-rate 3e-4 \
  --dit-lora-rank 16 --dit-lora-alpha 32 \
  --flow-steps 4 --candidates 8 --q-aggregation min \
  --kappa "${SVF_KAPPA:-1.0}" --g "${SVF_G:-1.0}" \
  --save-every 5000 --log-every 50 \
  --wandb-project bigenlight-multitask-gr00t --device cuda:0
