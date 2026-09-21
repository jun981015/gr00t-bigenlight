#!/usr/bin/env bash
# New ten-head IQL only. Evaluate both saved milestones after the 10k run.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$REPO_ROOT/environments/activate_n17.sh"
export WANDB_MODE=online WANDB_ENTITY=junhyeong OMP_NUM_THREADS=2
STORAGE_ROOT=/raid/yoon/vla_finetune
RUN_OUTPUT="${1:?Pass a fresh output directory}"
TRAIN_CACHE="$STORAGE_ROOT/features/bigenlight-n17-50per-task-bc10000-h16-v1"
EVAL_CACHE="$STORAGE_ROOT/features/bigenlight-n17-50per-bc10000-holdout56-h16-v1"
test ! -e "$RUN_OUTPUT"
python -u -m gr00t.rl.train_action_iql \
  --cache "$TRAIN_CACHE" --output-dir "$RUN_OUTPUT" --assume-all-success \
  --reward step-cost --gamma 0.99 --num-q-heads 10 --hidden-dims 512 512 256 \
  --batch-size 64 --critic-lr 1e-4 --value-lr 1e-4 --expectile-tau 0.8 \
  --steps 10000 --save-every 5000 --log-every 50 \
  --wandb-project bigenlight-multitask-gr00t --device cuda:0
for step in 5000 10000; do
  python -u -m gr00t.rl.eval_action_iql \
    --checkpoint "$RUN_OUTPUT/checkpoints/model-step-$step.pt" \
    --train-cache "$TRAIN_CACHE" --eval-cache "$EVAL_CACHE" \
    --output-dir "$RUN_OUTPUT/validation-step-$step" --device cuda:0 \
    --wandb-project bigenlight-multitask-gr00t
done
python - "$RUN_OUTPUT" <<'PY'
import json
from pathlib import Path
import sys
from gr00t.rl.feature_cache import atomic_json

root = Path(sys.argv[1])
reports = {str(step): json.loads((root/f"validation-step-{step}/report.json").read_text())
           for step in (5000, 10000)}
assert reports['5000']['eval_cache_manifest_sha256'] == reports['10000']['eval_cache_manifest_sha256']
comparison = {step: {'transition_weighted': r['transition_weighted'],
                     'episode_weighted': r['episode_weighted'], 'per_task': r['per_task']}
              for step, r in reports.items()}
atomic_json(root/'validation-comparison.json', comparison)
print(json.dumps({'status': 'training_and_validation_completed', 'comparison': comparison}), flush=True)
PY
