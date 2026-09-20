"""Run the 50/task IQL experiment after one specific all-data learner completes.

No model calls while waiting. Container-local queue; it cannot survive termination
of the GPU allocation and does not restart or stop any existing training process.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import time


def verify_completion(previous, steps):
    import torch

    checkpoint = previous / "checkpoints" / f"step-{steps}.pt"
    # Trusted local output of our own trainer; loading never occurs while waiting.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["step"] != steps or state["algorithm"]["updates"] != steps:
        raise ValueError("Previous checkpoint has not reached the requested update count")
    if state["algorithm"]["algorithm"] != "iql-critic-only-v1":
        raise ValueError("Previous checkpoint is not critic-only IQL")
    manifest = json.loads((previous / "run.json").read_text())
    if state["metadata"] != manifest["metadata"]:
        raise ValueError("Previous checkpoint/run metadata mismatch")
    return checkpoint


def process_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def watch_pid(pid, previous):
    # pidfd tracks this exact process, avoiding PID reuse and periodic polling.
    identity = process_identity(pid)
    if identity is None:
        raise ProcessLookupError(pid)
    descriptor = os.pidfd_open(pid) if hasattr(os, "pidfd_open") else None
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        output = Path(command[command.index("--output-dir") + 1]).resolve()
        if "gr00t.rl.train" not in command or output != previous.resolve():
            raise ValueError("PID does not belong to the specified previous IQL run")
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise
    return descriptor, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--previous-pid", type=int, required=True)
    parser.add_argument("--next-output", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.next_output.exists():
        raise ValueError("Require positive steps and a fresh next output directory")
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    with args.state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state_path.exists():
            raise FileExistsError("Queue state already exists; refusing duplicate dispatch")

        def status(phase, **extra):
            value = {
                "status": phase,
                "previous_run": str(args.previous_run),
                "previous_pid": args.previous_pid,
                "next_output": str(args.next_output),
                "variant": "50per-task",
                "steps": args.steps,
                "gpu": args.gpu,
                **extra,
            }
            temporary = args.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, indent=2) + "\n")
            temporary.replace(args.state_path)
            print(json.dumps(value), flush=True)

        try:
            try:
                descriptor, identity = watch_pid(args.previous_pid, args.previous_run)
            except ProcessLookupError:
                descriptor, identity = None, None
            status("waiting")
            if descriptor is not None:
                try:
                    select.select([descriptor], [], [])
                finally:
                    os.close(descriptor)
            elif identity is not None:
                # Some Python builds lack pidfd_open. Check exact process identity
                # quietly every 30 s; never poll the GPU or call an LLM.
                while process_identity(args.previous_pid) == identity:
                    time.sleep(30)
            checkpoint = verify_completion(args.previous_run, args.steps)
            environment = os.environ.copy()
            # Never inherit full-data weights or a critic resume checkpoint.
            for key in ("IQL_RESUME", "IQL_BC_MODEL"):
                environment.pop(key, None)
            environment.update(
                IQL_GPU=args.gpu, IQL_STEPS=str(args.steps), IQL_OUTPUT=str(args.next_output)
            )
            command = [
                "bash",
                str(Path(__file__).with_name("run_iql_n17.sh")),
                "50per-task",
                "--execute",
            ]
            status("launching", completed_checkpoint=str(checkpoint))
            result = subprocess.run(command, env=environment, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"50/task launcher exited {result.returncode}")
            verify_completion(args.next_output, args.steps)
            status("completed")
        except BaseException as exc:
            status("blocked", error=str(exc))
            raise


if __name__ == "__main__":
    main()
