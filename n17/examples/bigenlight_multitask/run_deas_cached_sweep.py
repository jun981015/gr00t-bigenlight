"""Eight independent DEAS cached runs, serially on the allocated GPU."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from gr00t.rl.feature_cache import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.sweep_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid sweep ID")
    root = Path("/raid/yoon/vla_finetune")
    status_path = root / "outputs" / f"deas-cached-sweep-{args.sweep_id}.json"
    if args.execute:
        if not Path("/.dockerenv").exists():
            raise RuntimeError("Run inside allocated GPU container")
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        uuid = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        subprocess.run(
            [sys.executable, "examples/carrot_in_pot/gpu_guard.py", "--gpu-uuid", uuid], check=True
        )
        lock = status_path.with_suffix(".lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = {
        **os.environ,
        "WANDB_MODE": "online",
        "WANDB_ENTITY": "junhyeong",
        "PYTHONUNBUFFERED": "1",
    }
    runs = []
    for steps in (10000, 30000):
        for variant in ("50per-task", "all"):
            for lr in ("1e-4", "3e-4"):
                output = (
                    root
                    / "outputs"
                    / f"bigenlight-n17-deas-{variant}-lr{lr}-{steps // 1000}k-{args.sweep_id}"
                )
                runs.append({"output": str(output), "steps": steps, "variant": variant, "lr": lr})
    for index, run in enumerate(runs):
        output = Path(run["output"])
        command = [
            sys.executable,
            "-m",
            "gr00t.rl.train_cached",
            "--algorithm",
            "deas",
            "--cache",
            str(root / "features" / f"bigenlight-n17-{run['variant']}-bc10000-h16-v1"),
            "--output-dir",
            str(output),
            "--reward",
            "step-cost",
            "--batch-size",
            "32",
            "--steps",
            str(run["steps"]),
            "--save-every",
            "5000" if run["steps"] == 10000 else "10000",
            "--discount1",
            "0.9",
            "--discount2",
            "0.99",
            "--expectile",
            "0.7",
            "--learning-rate",
            run["lr"],
            "--seed",
            "0",
            "--device",
            "cuda:0",
            "--wandb-project",
            "bigenlight-multitask-gr00t",
            "--log-every",
            "50",
        ]
        pointer = output / "checkpoints/latest_resumable.json"
        if pointer.exists():
            recovery = json.loads(pointer.read_text())
            if recovery["step"] == run["steps"]:
                continue
            command += ["--resume", str(pointer.parent / recovery["file"])]
        print(shlex.join(command), flush=True)
        if not args.execute:
            continue
        atomic_json(status_path, {"status": "running", "index": index, "runs": runs})
        try:
            with Path(str(output) + ".log").open("a") as log:
                subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            recovery = json.loads(pointer.read_text())
            if recovery["step"] != run["steps"]:
                raise RuntimeError(f"Interrupted at step {recovery['step']}; resume same sweep ID")
        except BaseException as error:
            atomic_json(
                status_path,
                {"status": "stopped", "index": index, "runs": runs, "error": str(error)},
            )
            raise
    if args.execute:
        atomic_json(status_path, {"status": "completed", "runs": runs})


if __name__ == "__main__":
    main()
