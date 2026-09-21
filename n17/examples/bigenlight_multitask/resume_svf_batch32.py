"""Explicit batch-32 continuation, preserving joint SVF optimizer and step."""

import json
import os
from pathlib import Path
import sys


root = Path("/home/yoon/raid/vla_finetune/outputs")
previous = root / "bigenlight-n17-svf-joint-deas-all-lora16-10k-20260921-reenabled"
output = root / "bigenlight-n17-svf-joint-deas-all-lora16-b32-10k-20260921"
settings = json.loads((previous / "run.json").read_text())["args"]
pointer = output / "checkpoints/latest_resumable.json"
if not pointer.exists():
    pointer = previous / "checkpoints/latest_resumable.json"
checkpoint = pointer.parent / json.loads(pointer.read_text())["file"]
settings.update(
    batch_size=32,
    steps=10000,
    output_dir=str(output),
    resume=str(checkpoint),
    allow_batch_size_change=True,
)
command = [sys.executable, "-m", "gr00t.rl.train"]
for key, value in settings.items():
    if value is None or value is False:
        continue
    command.append("--" + key.replace("_", "-"))
    if value is not True:
        command.extend(str(v) for v in value) if isinstance(value, list) else command.append(
            str(value)
        )
os.environ.update(WANDB_MODE="online", WANDB_ENTITY="junhyeong")
os.execv(sys.executable, command)
