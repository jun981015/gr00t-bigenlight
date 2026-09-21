#!/usr/bin/env bash
# Three independent runs; rerunning the same sweep ID resumes unfinished runs.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --execute ) ]]; then exit 2; fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/environments/activate_n17.sh"
export CUDA_VISIBLE_DEVICES="${IQL_GPU:-0}" WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-junhyeong}"
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$GR00T_N17_ROOT"
export IQL_SWEEP_ID="${IQL_SWEEP_ID:-$(date +%Y%m%dT%H%M%S)}"
export IQL_SWEEP_EXECUTE="${1:-}"
export IQL_VARIANT="${IQL_VARIANT:-all}"
export IQL_STEPS="${IQL_STEPS:-30000}"
[[ "$IQL_STEPS" =~ ^[1-9][0-9]*$ ]] || exit 2
case "$IQL_VARIANT" in all|50per-task) ;; *) exit 2 ;; esac
if [[ "$IQL_SWEEP_EXECUTE" == --execute ]]; then
  test -f /.dockerenv
  if [[ -n "${IQL_SWEEP_AFTER:-}" ]]; then
    python - <<'PY'
import json
import os
from pathlib import Path
import time

path = Path(os.environ['IQL_SWEEP_AFTER'])
print(f'Waiting for preceding sweep: {path}', flush=True)
deadline = time.monotonic() + 8 * 3600
while True:
    if not path.exists():
        if time.monotonic() >= deadline:
            raise SystemExit('Timed out waiting for preceding sweep to start')
        time.sleep(15)
        continue
    state = json.loads(path.read_text())
    if state['status'] == 'completed':
        break
    if state['status'] != 'running':
        raise SystemExit(f'Preceding sweep did not complete: {state["status"]}')
    if time.monotonic() >= deadline:
        raise SystemExit('Timed out waiting for preceding sweep')
    time.sleep(15)
PY
  fi
  uuid=$(nvidia-smi -i "${IQL_GPU:-0}" --query-gpu=uuid --format=csv,noheader)
  python examples/carrot_in_pot/gpu_guard.py --gpu-uuid "$uuid"
fi
python - <<'PY'
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from gr00t.rl.feature_cache import atomic_json

root = Path(os.environ['VLA_STORAGE_ROOT'])
sweep_id = os.environ['IQL_SWEEP_ID']
execute = os.environ['IQL_SWEEP_EXECUTE'] == '--execute'
variant = os.environ['IQL_VARIANT']
steps = int(os.environ['IQL_STEPS'])
step_label = f'{steps // 1000}k' if steps % 1000 == 0 else str(steps)
cache = root / f'features/bigenlight-n17-{variant}-bc10000-h16-v1'
status_path = root / 'outputs' / f'cached-iql-kappa-sweep-{sweep_id}.json'
outputs = {}
for kappa in ('0.7', '0.8', '0.9'):
    output = root / 'outputs' / f'bigenlight-n17-cached-iql-{variant}-step-cost-kappa{kappa}-{step_label}-{sweep_id}'
    outputs[kappa] = str(output)
    command = [sys.executable, '-m', 'gr00t.rl.train_cached',
               '--cache', str(cache), '--output-dir', str(output),
               '--reward', 'step-cost', '--steps', str(steps), '--batch-size', '32',
               '--expectile', kappa, '--learning-rate', '0.0003', '--gamma', '0.99',
               '--seed', '0', '--device', 'cuda:0', '--save-every', '5000',
               '--log-every', '50', '--wandb-project', 'bigenlight-multitask-gr00t']
    pointer = output / 'checkpoints/latest_resumable.json'
    if pointer.exists():
        state = json.loads(pointer.read_text())
        if state['step'] >= steps:
            print(f'Already complete: {output}', flush=True)
            continue
        command += ['--resume', str(pointer.parent / state['file'])]
    print(shlex.join(command), flush=True)
    if not execute:
        continue
    atomic_json(status_path, {'status': 'running', 'kappa': kappa, 'outputs': outputs})
    try:
        subprocess.run(command, check=True)
        state = json.loads(pointer.read_text())
        if state['step'] != steps:
            raise RuntimeError(f'Run stopped at {state["step"]}; rerun with the same IQL_SWEEP_ID')
    except BaseException as exc:
        atomic_json(status_path, {'status': 'stopped', 'kappa': kappa,
                                 'outputs': outputs, 'error': str(exc)})
        raise
if execute:
    atomic_json(status_path, {'status': 'completed', 'outputs': outputs, 'steps_each': steps})
PY
