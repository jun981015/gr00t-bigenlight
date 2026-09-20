#!/usr/bin/env bash
# Dry-run by default. --execute starts training; --resume restores full state.
set -euo pipefail
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$RECIPE_DIR/activate.sh"
cd "$DEAS_REPO_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
BC_RUN_OUTPUT="${BC_RUN_OUTPUT:-$VLA_STORAGE_ROOT/outputs/deas-n15-bc24-1gpu-b32}"
BC_RUN_STEPS="${BC_RUN_STEPS:-30000}"
exec python -u "$RECIPE_DIR/manage.py" train bc24 \
  --num-gpus 1 --batch-size 32 --max-steps "$BC_RUN_STEPS" \
  --save-steps 10000 --logging-steps 50 --video-backend torchvision_av \
  --output "$BC_RUN_OUTPUT" "$@"
