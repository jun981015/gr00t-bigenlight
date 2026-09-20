#!/usr/bin/env bash
set -eo pipefail
source /home/yoon/vla_finetune/activate_gr00t.sh
cd /home/yoon/vla_finetune/Isaac-GR00T
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$VLA_STORAGE_ROOT/logs/robocasa-n17-svf"
exec >> "$VLA_STORAGE_ROOT/logs/robocasa-n17-svf/download.log" 2>&1
date --iso-8601=seconds
exec flock -n "$VLA_STORAGE_ROOT/logs/robocasa-n17-svf/download.lock" \
    python -u examples/robocasa_svf/prepare.py download-and-prepare offline4 bc24
