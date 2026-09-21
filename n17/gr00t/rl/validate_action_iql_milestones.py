"""CPU validation worker for atomically published action-IQL model archives.

Does not update/interrupt training. One worker per run; exits after requested
milestones or a bounded timeout. Unchanged waiting state produces no log spam.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .feature_cache import atomic_json


def completed_report(directory, step, wandb_enabled):
    needed = [directory / "report.json", directory / "predictions.npz"]
    if wandb_enabled:
        needed.append(directory / "wandb.json")
    if not all(p.is_file() for p in needed):
        return False
    report = json.loads(needed[0].read_text())
    return report["step"] == step and report["q_aggregation"] == "mean"


def update_comparison(root, directories):
    reports = {}
    for directory in directories:
        report = json.loads((directory / "report.json").read_text())
        reports[str(report["step"])] = report
    hashes = {r["eval_cache_manifest_sha256"] for r in reports.values()}
    if len(hashes) != 1:
        raise ValueError("Cannot compare different holdout caches")
    atomic_json(
        root / "validation-comparison.json",
        {
            step: {k: report[k] for k in ("transition_weighted", "episode_weighted", "per_task")}
            for step, report in sorted(reports.items(), key=lambda item: int(item[0]))
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[100000, 150000, 200000, 250000, 300000]
    )
    parser.add_argument("--wandb-project", default="bigenlight-multitask-gr00t")
    parser.add_argument("--timeout-hours", type=float, default=12)
    args = parser.parse_args(argv)
    if args.timeout_hours <= 0 or not args.steps or min(args.steps) < 1:
        raise ValueError("Invalid milestones/timeout")
    root = args.run_dir.resolve(strict=True)
    metadata = json.loads((root / "run.json").read_text())
    if metadata["backend"] != "action-conditioned-iql-cache-v1":
        raise ValueError("Wrong training backend")
    train_cache = metadata["args"]["cache"]
    status_path = root / "validation-worker-status.json"
    state = {
        "status": "starting",
        "pid": os.getpid(),
        "steps": sorted(set(args.steps)),
        "completed": {},
    }
    with (root / "validation-worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + args.timeout_hours * 3600
        reports = [
            d
            for d in root.glob("validation-step-*")
            if (d / "report.json").is_file() and (d / "predictions.npz").is_file()
        ]
        try:
            for step in state["steps"]:
                archive = root / f"checkpoints/model-step-{step}.pt"
                state.update(status="waiting_for_checkpoint", current_step=step)
                atomic_json(status_path, state)
                # OfflineTrainer publishes archives with exclusive hard links:
                # existence means serialization/fsync finished, not a partial file.
                while not archive.is_file():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Checkpoint {step} not published before deadline")
                    time.sleep(10)
                output = root / f"validation-step-{step}"
                attempt = 0
                while output.exists() and not completed_report(
                    output, step, bool(args.wandb_project)
                ):
                    attempt += 1
                    output = root / f"validation-step-{step}-attempt-{attempt}"
                if not output.exists():
                    state.update(status="evaluating", report_dir=str(output))
                    atomic_json(status_path, state)
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "gr00t.rl.eval_action_iql",
                        "--checkpoint",
                        str(archive),
                        "--train-cache",
                        train_cache,
                        "--eval-cache",
                        str(args.eval_cache),
                        "--output-dir",
                        str(output),
                        "--device",
                        "cpu",
                    ]
                    if args.wandb_project:
                        command += ["--wandb-project", args.wandb_project]
                    with output.with_suffix(".log").open("a") as log:
                        subprocess.run(
                            command,
                            check=True,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=max(1, deadline - time.monotonic()),
                        )
                if not completed_report(output, step, bool(args.wandb_project)):
                    raise RuntimeError(f"Incomplete validation at step {step}")
                reports.append(output)
                update_comparison(root, reports)
                state["completed"][str(step)] = str(output / "report.json")
                atomic_json(status_path, state)
                print(f"Validation complete: step {step}, {output}", flush=True)
            state["status"] = "completed"
            atomic_json(status_path, state)
        except BaseException as exc:
            state.update(status="failed", error=str(exc))
            atomic_json(status_path, state)
            raise


if __name__ == "__main__":
    main()
