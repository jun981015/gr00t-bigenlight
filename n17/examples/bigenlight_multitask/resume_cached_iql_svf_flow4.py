"""Explicit 10->4 SDE-step continuation, preserving optimizer and update count."""

import json
import os
from pathlib import Path
import sys

root = Path("/raid/yoon/vla_finetune/outputs")
source = root / "bigenlight-n17-cached-svf-fixed-iql-50per-e07-q30k-lora16-b32-10k-20260921T043838"
settings = json.loads((source / "run.json").read_text())["args"]
pointer = json.loads((source / "checkpoints/latest_resumable.json").read_text())
settings.update(
    flow_steps=4,
    allow_flow_steps_change=True,
    resume=str(source / "checkpoints" / pointer["file"]),
    output_dir=str(
        root / "bigenlight-n17-cached-svf-fixed-iql-50per-e07-q30k-lora16-b32-flow4-10k-20260921"
    ),
)
command = [sys.executable, "-u", "-m", "gr00t.rl.train_cached_svf"]
for key, value in settings.items():
    if value is None or value is False:
        continue
    command.append("--" + key.replace("_", "-"))
    if value is not True:
        command.append(str(value))
os.environ.update(WANDB_MODE="online", WANDB_ENTITY="junhyeong")
os.execv(sys.executable, command)
