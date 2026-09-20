#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$RECIPE_DIR/activate.sh"
DATA_LOG_DIR="$VLA_STORAGE_ROOT/logs/deas-robocasa"
mkdir -p "$DATA_LOG_DIR"
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec >> "$DATA_LOG_DIR/download.log" 2>&1
date --iso-8601=seconds
exec flock -n "$DATA_LOG_DIR/download.lock" python -u "$RECIPE_DIR/manage.py" download offline4 model bc24
