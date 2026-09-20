#!/usr/bin/env bash
# Skip 24-task BC; initialize four-task filtered BC from the N1.5 base model.
# Dry-run by default; --execute starts, --resume restores this run's full state.
set -euo pipefail
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$RECIPE_DIR/activate.sh"
cd "$DEAS_REPO_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
BC_RUN_OUTPUT="${BC_RUN_OUTPUT:-$VLA_STORAGE_ROOT/outputs/deas-n15-filtered-bc-direct-1gpu-b32}"
BC_RUN_STEPS="${BC_RUN_STEPS:-30000}"
exec python -u "$RECIPE_DIR/manage.py" train filtered-bc \
  --base-model "$VLA_STORAGE_ROOT/models/GR00T-N1.5-3B" \
  --num-gpus 1 --batch-size 32 --max-steps "$BC_RUN_STEPS" \
  --save-steps 5000 --logging-steps 50 --video-backend torchvision_av \
  --output "$BC_RUN_OUTPUT" "$@"
