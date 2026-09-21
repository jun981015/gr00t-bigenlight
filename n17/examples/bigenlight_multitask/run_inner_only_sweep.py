"""Fork two inner-only runs from one stopped joint SVF snapshot."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from gr00t.rl.feature_cache import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-step", type=int, required=True)
    parser.add_argument("--sweep-id", required=True)
    args = parser.parse_args()
    source = args.source_run / "checkpoints" / f"step-{args.source_step}.pt"
    if not source.exists():
        raise FileNotFoundError(source)
    original = json.loads((args.source_run / "run.json").read_text())["args"]
    status = args.source_run.parent / f"inner-only-sweep-{args.sweep_id}.json"
    lock = status.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runs = []
    for steps in (10000, 30000):
        output = (
            args.source_run.parent
            / f"bigenlight-n17-inner-only-deas-all-{steps // 1000}k-{args.sweep_id}"
        )
        runs.append({"steps": steps, "output": str(output)})
    for index, run in enumerate(runs):
        output = Path(run["output"])
        settings = {
            **original,
            "output_dir": str(output),
            "steps": run["steps"],
            "inner_only": True,
            "initialize_svf": str(source),
            "resume": None,
            "save_every": 5000 if run["steps"] == 10000 else 10000,
        }
        pointer = output / "checkpoints/latest_resumable.json"
        if pointer.exists():
            saved = json.loads(pointer.read_text())
            if saved["step"] == run["steps"]:
                continue
            settings["resume"] = str(pointer.parent / saved["file"])
        command = [sys.executable, "-m", "gr00t.rl.train"]
        for key, value in settings.items():
            if value is None or value is False:
                continue
            command.append("--" + key.replace("_", "-"))
            if value is not True:
                command.extend(str(v) for v in value) if isinstance(
                    value, list
                ) else command.append(str(value))
        atomic_json(
            status, {"status": "running", "index": index, "source": str(source), "runs": runs}
        )
        try:
            with Path(str(output) + ".log").open("a") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "WANDB_MODE": "online", "WANDB_ENTITY": "junhyeong"},
                )
            saved = json.loads(pointer.read_text())
            if saved["step"] != run["steps"]:
                raise RuntimeError(
                    f"Stopped at {saved['step']}; resume this sweep after allocation recovery"
                )
        except BaseException as error:
            atomic_json(
                status,
                {
                    "status": "stopped",
                    "index": index,
                    "source": str(source),
                    "runs": runs,
                    "error": str(error),
                },
            )
            raise
    atomic_json(status, {"status": "completed", "source": str(source), "runs": runs})


if __name__ == "__main__":
    main()
