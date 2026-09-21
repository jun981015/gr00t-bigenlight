#!/usr/bin/env bash
# Extract only unseen episodes, then evaluate all three 30k critics without updates.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/environments/activate_n17.sh"
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_ENTITY=junhyeong
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1
cd "$GR00T_N17_ROOT"
python - <<'PY'
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from gr00t.rl.feature_cache import atomic_json

root = Path('/raid/yoon/vla_finetune')
status_path = root / 'outputs/iql-50per-30k-holdout-20260921.json'
with status_path.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dependencies = [root/'outputs'/f'cached-iql-kappa-sweep-20260921-{name}.json'
                    for name in ['50per-kappa-30k', '50per-kappa-10k', 'all-kappa-10k']]
    phase = 'waiting_for_training'
    atomic_json(status_path, {'status': phase})
    try:
        deadline = time.monotonic() + 8*3600
        for path in dependencies:
            while True:
                state = json.loads(path.read_text()) if path.exists() else None
                if state and state['status'] == 'completed': break
                if state and state['status'] != 'running':
                    raise RuntimeError(f'Preceding job stopped: {path}')
                if time.monotonic() >= deadline: raise TimeoutError('Preceding jobs not completed')
                time.sleep(15)
        subprocess.run([sys.executable, 'examples/carrot_in_pot/gpu_guard.py'], check=True)
        train_cache = root/'features/bigenlight-n17-50per-task-bc10000-h16-v1'
        manifest = json.loads((train_cache/'manifest.json').read_text())
        data = root/'datasets/bigenlight_multitask_gr00t_holdout_50per_task/n17'
        cache = root/'features/bigenlight-n17-50per-bc10000-holdout56-h16-v1'
        phase = 'extracting_holdout_features'
        atomic_json(status_path, {'status': phase, 'cache':str(cache)})
        subprocess.run([sys.executable, '-m', 'gr00t.rl.cache_features',
                        '--model-path', manifest['identity']['model_path'],
                        '--dataset-path', str(data), '--output-dir', str(cache),
                        '--batch-size', '32', '--cpu-threads', '4', '--sleep-ms', '0',
                        '--horizon', '16', '--device', 'cuda:0'], check=True)
        reports = []
        for kappa in ['0.7', '0.8', '0.9']:
            run = root/'outputs'/f'bigenlight-n17-cached-iql-50per-task-step-cost-kappa{kappa}-30k-20260921-50per-kappa-30k'
            output = root/'outputs'/f'bigenlight-n17-iql-50per-holdout56-kappa{kappa}-30k-20260921'
            phase = f'evaluating_kappa_{kappa}'
            atomic_json(status_path, {'status':phase, 'reports':reports})
            if not (output/'report.json').exists() or not (output/'wandb.json').exists():
                subprocess.run([sys.executable, '-m', 'gr00t.rl.eval_cached_iql',
                                '--checkpoint', str(run/'checkpoints/model-step-30000.pt'),
                                '--train-cache', str(train_cache), '--eval-cache', str(cache),
                                '--output-dir', str(output), '--device', 'cuda:0',
                                '--wandb-project', 'bigenlight-multitask-gr00t'], check=True)
            report = json.loads((output/'report.json').read_text())
            reports.append({'kappa':kappa, 'path':str(output/'report.json'),
                            'metrics':report['episode_weighted'],
                            'wandb':json.loads((output/'wandb.json').read_text())['url']})
        atomic_json(status_path, {'status':'completed', 'reports':reports})
    except BaseException as exc:
        atomic_json(status_path, {'status':'failed', 'phase':phase, 'error':str(exc)})
        raise
PY
