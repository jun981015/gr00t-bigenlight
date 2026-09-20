#!/usr/bin/env bash
# Dry-run by default. Execute inside an allocated GPU container only.
set -eo pipefail
source "$HOME/vla_finetune/activate_gr00t.sh"
cd "$HOME/vla_finetune/Isaac-GR00T"
execute=false
if [[ "${1:-}" == --execute ]]; then execute=true; shift; fi
dataset="${BIGENLIGHT_DATASET_PATH:-$VLA_STORAGE_ROOT/datasets/bigenlight_multitask_gr00t/n17}"
output="${BIGENLIGHT_OUTPUT_PATH:-$VLA_STORAGE_ROOT/outputs/bigenlight-n17-bc-$(date +%Y%m%dT%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-bigenlight-multitask-gr00t}"
export WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
command=(python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1
    gr00t/experiment/launch_finetune.py
    --base-model-path "$VLA_STORAGE_ROOT/models/GR00T-N1.7-3B"
    --dataset-path "$dataset" --embodiment-tag NEW_EMBODIMENT
    --modality-config-path examples/bigenlight_multitask/config.py
    --output-dir "$output"
    --use-wandb --wandb-project "$WANDB_PROJECT"
    --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model
    --num-gpus 1 --global-batch-size 32 --gradient-accumulation-steps 1
    --max-steps 10000 --learning-rate 1e-4 --precision bf16-mixed --logging-steps 50
    --dataloader-num-workers 2 --shard-size 128 --episode-sampling-rate 1.0
    --num-shards-per-epoch 128 --shortest-image-edge 256 --crop-fraction 1.0
    --save-steps 5000 --save-total-limit 0 --keep-latest-training-state
    --first-save-step 100 --save-interval-seconds 1800
    --max-run-seconds 28800 "$@")
printf '%q ' "${command[@]}"; printf '\n'
if ! "$execute"; then echo 'DRY RUN: add --execute to train.'; exit 0; fi
test -f "$dataset/VALIDATION.json" || { echo 'Dataset validation missing' >&2; exit 1; }
if [[ -d "$output" && -n "$(ls -A "$output")" ]]; then
    echo "Output is not empty: $output. Use a new path; this launcher does not resume." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used --format=csv
# A new BC run must not inherit the N1.5 or previous dataset's W&B identity.
unset WANDB_RUN_ID WANDB_RESUME
export WANDB_DIR="$output"
export WANDB_CACHE_DIR="$VLA_STORAGE_ROOT/cache/wandb"
export WANDB_DATA_DIR="$VLA_STORAGE_ROOT/cache/wandb-data"
export WANDB_ARTIFACT_DIR="$VLA_STORAGE_ROOT/artifacts/wandb"
mkdir -p "$output"
"${command[@]}" 2>&1 | tee "$output/train.log"
