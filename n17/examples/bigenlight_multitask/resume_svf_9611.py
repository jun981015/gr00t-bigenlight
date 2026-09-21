"""Finish the interrupted batch-32 run without overwriting later metrics."""

import json
import os
from pathlib import Path
import sys

root = Path('/raid/yoon/vla_finetune/outputs')
source = root / 'bigenlight-n17-svf-joint-deas-all-lora16-b32-10k-20260921'
settings = json.loads((source / 'run.json').read_text())['args']
settings.update(
    resume=str(source / 'checkpoints/step-9611.pt'),
    output_dir=str(root / 'bigenlight-n17-svf-joint-deas-all-lora16-b32-10k-resume9611-20260921'),
    batch_size=32,
    steps=10000,
    allow_batch_size_change=True,
)
command = [sys.executable, '-m', 'gr00t.rl.train']
for key, value in settings.items():
    if value is None or value is False:
        continue
    command.append('--' + key.replace('_', '-'))
    if value is not True:
        command.extend(str(v) for v in value) if isinstance(value, list) else command.append(str(value))
os.environ.update(WANDB_MODE='online', WANDB_ENTITY='junhyeong')
os.execv(sys.executable, command)
