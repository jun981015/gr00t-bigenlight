#!/usr/bin/env bash
# Select exactly one of the two N1.7 BC runs. No execution without --execute.
set -eo pipefail
variant="${1:-}"
case "$variant" in
    50per-task) suffix=_50per_task; gpu=0 ;;
    all) suffix=; gpu=1 ;;
    *) echo 'Usage: bash run_n17.sh {50per-task|all} [--execute]' >&2; exit 2 ;;
esac
shift
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --execute ) ]]; then
    echo 'Only --execute is accepted after the variant; no work has started.' >&2
    exit 2
fi
source "$HOME/vla_finetune/activate_gr00t.sh"
repo="$HOME/vla_finetune/Isaac-GR00T"
export BIGENLIGHT_DATASET_PATH="$VLA_STORAGE_ROOT/datasets/bigenlight_multitask_gr00t${suffix}/n17"
export BIGENLIGHT_OUTPUT_PATH="$VLA_STORAGE_ROOT/outputs/bigenlight-n17-${variant}-b32-10k-$(date +%Y%m%dT%H%M%S)"
export WANDB_PROJECT=bigenlight-multitask-gr00t
export CUDA_VISIBLE_DEVICES="$gpu"
# Avoid a PYTHONPATH inherited from the DEAS/N1.5 environment.
export PYTHONPATH="$repo"
printf 'Prepared N1.7 %s: GPU %s; batch 32; 10000 optimizer steps\n' "$variant" "$gpu"
if [[ "${1:-}" == --execute ]]; then
    test -f /.dockerenv || { echo 'Execute only inside the allocated GPU container.' >&2; exit 1; }
    uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
    # Read-only check: refuses an occupied GPU, never stops or starts keepalives.
    python "$repo/examples/carrot_in_pot/gpu_guard.py" --gpu-uuid "$uuid"
fi
exec bash "$repo/examples/bigenlight_multitask/train.sh" "$@"
