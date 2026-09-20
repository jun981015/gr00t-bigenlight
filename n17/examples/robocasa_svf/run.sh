#!/usr/bin/env bash
set -eo pipefail
source /home/yoon/vla_finetune/activate_gr00t.sh
cd /home/yoon/vla_finetune/Isaac-GR00T
exec python examples/robocasa_svf/launch.py "$@"
