"""Wait for extraction, then train fresh cached IQL: all -> 50/task.

Container-local blocking flock, no periodic GPU/LLM monitoring. Incomplete caches
or interrupted training block the queue instead of silently using partial data.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

from after_iql import verify_completion
from gr00t.rl.feature_cache import FORMAT, atomic_json


def verify_cache(root):
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    complete = json.loads((root / "COMPLETE.json").read_text())
    if (
        complete.get("format") != FORMAT
        or complete.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
        or complete.get("episodes") != len(manifest["episodes"])
    ):
        raise ValueError(f"Incomplete or inconsistent cache: {root}")
    # The trainer validates every array before allocating its learner on the GPU.
    for episode in manifest["episodes"]:
        filename = episode["file"]
        if Path(filename).name != filename or not (root / filename).is_file():
            raise ValueError(f"Missing or unsafe episode file: {filename}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--all-output", type=Path, required=True)
    parser.add_argument("--subset-output", type=Path, required=True)
    parser.add_argument("--reward", choices=["terminal-success", "step-cost"], required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args(argv)
    outputs = {"all": args.all_output, "50per-task": args.subset_output}
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    with args.state_path.with_suffix(".lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state_path.exists() or any(p.exists() for p in outputs.values()):
            raise FileExistsError("Require fresh queue state and output directories")

        def status(phase, **extra):
            value = {
                "status": phase,
                "reward": args.reward,
                "outputs": {k: str(v) for k, v in outputs.items()},
                "steps_each": 10000,
                "batch_size": 32,
                "gpu": args.gpu,
                **extra,
            }
            atomic_json(args.state_path, value)
            print(json.dumps(value), flush=True)

        try:
            lock_path = args.storage_root / "features" / f"n17-extraction-gpu-{args.gpu}.lock"
            # Existing extractor owns EX across both datasets. Keep SH while
            # training to prevent another extractor from claiming the same GPU.
            with lock_path.open("r") as extraction_lock:
                status("waiting_for_feature_cache")
                fcntl.flock(extraction_lock, fcntl.LOCK_SH)
                for variant in outputs:
                    verify_cache(
                        args.storage_root / "features" / f"bigenlight-n17-{variant}-bc10000-h16-v1"
                    )
                environment = os.environ.copy()
                for key in ("IQL_RESUME", "IQL_BC_MODEL"):
                    environment.pop(key, None)
                environment.update(IQL_GPU=args.gpu, IQL_REWARD=args.reward, WANDB_MODE="online")
                for variant, output in outputs.items():
                    environment["IQL_OUTPUT"] = str(output)
                    status("training", variant=variant)
                    command = [
                        "bash",
                        str(Path(__file__).with_name("train_cached_iql.sh")),
                        variant,
                        "--execute",
                    ]
                    with Path(str(output) + ".log").open("x") as log:
                        subprocess.run(command, env=environment, stdout=log, stderr=log, check=True)
                    verify_completion(output, 10000)
                status("completed")
        except BaseException as exc:
            status("blocked", error=str(exc))
            raise


if __name__ == "__main__":
    main()
