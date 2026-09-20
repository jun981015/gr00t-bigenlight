#!/usr/bin/env bash
# Explicit BC checkpoint required. Dry-run unless --execute is passed.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 /path/to/filtered-bc/checkpoint-N [--execute] [--resume]" >&2
  exit 2
fi
IQL_ACTOR_CHECKPOINT=$1
shift
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$RECIPE_DIR/activate.sh"
cd "$DEAS_REPO_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
IQL_RUN_OUTPUT="${IQL_RUN_OUTPUT:-$VLA_STORAGE_ROOT/outputs/deas-n15-iql-critic-1gpu-b16}"
IQL_RUN_STEPS="${IQL_RUN_STEPS:-30000}"
exec python -u "$RECIPE_DIR/manage.py" train critic --critic-algorithm iql \
  --base-model "$IQL_ACTOR_CHECKPOINT" --num-gpus 1 --batch-size 16 \
  --max-steps "$IQL_RUN_STEPS" --save-steps 5000 --logging-steps 50 \
  --video-backend torchvision_av --output "$IQL_RUN_OUTPUT" "$@"
