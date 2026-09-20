#!/usr/bin/env bash
# source experiments/robocasa_deas/activate.sh
DEAS_RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEAS_REPO_DIR=$(cd -- "$DEAS_RECIPE_DIR/../.." && pwd)
source "$DEAS_REPO_DIR/../storage_env.sh"
export TMPDIR="$VLA_STORAGE_ROOT/tmp"
export WANDB_PROJECT=gr00t-deas-robocasa
export USE_TF=0
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$DEAS_REPO_DIR"
source "$VLA_STORAGE_ROOT/envs/deas-gr00t-n1.5/bin/activate"
