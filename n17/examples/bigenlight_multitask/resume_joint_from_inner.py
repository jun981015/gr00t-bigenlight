"""Continue the latest saved inner-only weights with actor+inner for 10k updates."""

import json
import os
from pathlib import Path
import sys

from gr00t.rl.feature_cache import atomic_json


def main():
    root = Path("/home/yoon/raid/vla_finetune/outputs")
    original = root / "bigenlight-n17-inner-only-deas-all-10k-20260921-from114"
    pointer = json.loads((original / "checkpoints/latest_resumable.json").read_text())
    source = original / "checkpoints" / pointer["file"]
    output = root / "bigenlight-n17-svf-joint-deas-all-lora16-10k-20260921-reenabled"
    settings = json.loads((original / "run.json").read_text())["args"]
    settings.update(
        inner_only=False,
        initialize_svf=str(source),
        output_dir=str(output),
        steps=10000,
        resume=None,
        save_every=5000,
    )
    resume = output / "checkpoints/latest_resumable.json"
    if resume.exists():
        settings["resume"] = str(resume.parent / json.loads(resume.read_text())["file"])
    status = root / "inner-only-sweep-20260921-from114.json"
    previous = json.loads(status.read_text())
    previous.update(
        status="cancelled",
        reason="User requested actor+inner 10k instead",
        replacement=str(output),
        saved_inner_step=pointer["step"],
    )
    atomic_json(status, previous)
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


if __name__ == "__main__":
    main()
