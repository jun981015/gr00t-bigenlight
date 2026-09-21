"""Run four encoder-IQL experiments after the currently allocated SVF finishes."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gr00t.rl.feature_cache import atomic_json


root = Path("/raid/yoon/vla_finetune/outputs")
tag = "20260921-encoder-iql"
status = root / f"{tag}-queue.json"
prior = root / "bigenlight-n17-cached-svf-fixed-iql-50per-e07-q30k-lora16-b32-flow4-10k-20260921"
os.environ.update(WANDB_MODE="online", WANDB_ENTITY="junhyeong")


def run():
    atomic_json(status, {"status": "waiting_for_svf", "dependency": str(prior)})
    while True:
        pointer = prior / "checkpoints/latest_resumable.json"
        if pointer.exists() and json.loads(pointer.read_text())["step"] >= 10000:
            break
        processes = subprocess.check_output(
            ["pgrep", "-af", "gr00t.rl.train_cached_svf"], text=True
        )
        if str(prior) not in processes:
            raise RuntimeError("SVF exited before step 10000; queue not started")
        time.sleep(30)
    # The finished checkpoint can precede process cleanup. Never overlap another GPU job.
    while subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        time.sleep(15)
    for subset in ["50per-task", "all"]:
        for steps in [10000, 30000]:
            output = (
                root / f"bigenlight-n17-iql-encoder-{subset}-expectile0.7-{steps // 1000}k-{tag}"
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                "gr00t.rl.train_cached",
                "--algorithm",
                "iql",
                "--iql-encoder",
                "deas-mlp",
                "--cache",
                f"/raid/yoon/vla_finetune/features/bigenlight-n17-{subset}-bc10000-h16-v1",
                "--output-dir",
                str(output),
                "--reward",
                "step-cost",
                "--gamma",
                ".99",
                "--expectile",
                ".7",
                "--learning-rate",
                "3e-4",
                "--batch-size",
                "32",
                "--steps",
                str(steps),
                "--save-every",
                "5000" if steps == 10000 else "10000",
                "--log-every",
                "50",
                "--device",
                "cuda:0",
                "--wandb-project",
                "bigenlight-multitask-gr00t",
            ]
            atomic_json(status, {"status": "running", "output": str(output), "command": command})
            print("START", str(output), flush=True)
            with Path(str(output) + ".log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            pointer = json.loads((output / "checkpoints/latest_resumable.json").read_text())
            if pointer["step"] != steps:
                raise RuntimeError("Training stopped before target; remaining queue aborted")
            print("COMPLETED", str(output), flush=True)
    atomic_json(status, {"status": "completed", "runs": 4})


with (root / f"{tag}-queue.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        run()
    except Exception as exc:
        atomic_json(status, {"status": "failed", "error": repr(exc)})
        raise
