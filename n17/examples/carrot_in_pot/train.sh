#!/usr/bin/env bash
# Run inside the allocated GPU container. Does not stop any existing GPU job.
set -eo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/activate_gr00t.sh"
cd "$GR00T_N17_ROOT"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
if [ -z "${CARROT_KEEPALIVE_PIDS:-}" ]; then
    carrot_keepalive_pids=$(python examples/carrot_in_pot/ensure_keepalive.py)
    export CARROT_KEEPALIVE_PIDS="$carrot_keepalive_pids"
fi
if [ "${1:-}" = '--resume-dir' ]; then
    shift
    exec python examples/carrot_in_pot/resume_training.py "$@"
fi
carrot_root="$VLA_STORAGE_ROOT/datasets/carrot_in_pot_gr00t"
test -f "$carrot_root/VALIDATION.json" || { echo 'Run data preparation/validation first.' >&2; exit 1; }
python examples/carrot_in_pot/gpu_guard.py
carrot_output="$VLA_STORAGE_ROOT/outputs/carrot-bc-$(date +%Y%m%dT%H%M%S)-$$"
# Metrics/configuration go to W&B; model checkpoints stay on RAID.
# Leave WANDB_ENTITY unset to use the authenticated account's default entity.
export WANDB_PROJECT="${WANDB_PROJECT:-carrot-in-pot-gr00t}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$carrot_output"
export WANDB_CACHE_DIR="$VLA_STORAGE_ROOT/cache/wandb"
export WANDB_DATA_DIR="$VLA_STORAGE_ROOT/cache/wandb-data"
export WANDB_ARTIFACT_DIR="$VLA_STORAGE_ROOT/artifacts/wandb"
export WANDB_LOG_MODEL=false
export WANDB_WATCH=false
export WANDB_DISABLE_CODE=true
mkdir -p "$carrot_output" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"
printf 'Output: %s\nW&B project: %s; entity: %s; mode: %s\n' \
    "$carrot_output" "$WANDB_PROJECT" "${WANDB_ENTITY:-account default}" "$WANDB_MODE"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$VLA_STORAGE_ROOT/models/GR00T-N1.7-3B" \
    --dataset-path "$carrot_root/train" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/carrot_in_pot/carrot_config.py \
    --output-dir "$carrot_output" \
    --use-wandb --wandb-project "$WANDB_PROJECT" \
    --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
    --num-gpus 2 --global-batch-size 32 --max-steps 10000 \
    --learning-rate 1e-4 --dataloader-num-workers 2 \
    --shard-size 128 --episode-sampling-rate 1.0 --num-shards-per-epoch 128 \
    --shortest-image-edge 256 --crop-fraction 1.0 \
    --save-steps 2000 --save-total-limit 0 \
    --keep-latest-training-state --max-run-seconds 32400 \
    "$@" 2>&1 | tee "$carrot_output/train.log"
