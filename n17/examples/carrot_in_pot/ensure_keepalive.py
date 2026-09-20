"""Start or reuse user-owned, low-duty keepalives on the two allocated GPUs."""

import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time


def reusable_pid(script, device):
    pidfile = script.with_suffix(".pid" if device == 0 else f".gpu{device}.pid")
    try:
        pid = int(pidfile.read_text().strip())
        guard = runpy.run_path(str(Path(__file__).with_name("gpu_guard.py")))["allowed_keepalives"]
        guard(str(pid), expected_script=script)
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        if command[command.index(b"--device") + 1] != str(device).encode():
            return None
        return pid
    except (OSError, ValueError, IndexError):
        return None


def main():
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True, timeout=10
    ).splitlines()
    if len(devices) != 2:
        raise SystemExit("This recipe expects exactly two GPUs in the allocated container")
    script = Path.home() / "vla_finetune/gpu_keepalive.py"
    log_root = Path(os.environ["VLA_STORAGE_ROOT"]) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = None
    pids = []
    for device in (0, 1):
        pid = reusable_pid(script, device)
        if pid is not None:
            print(f"Reusing GPU {device} keepalive PID {pid}", file=sys.stderr)
            pids.append(pid)
            continue
        if log_dir is None:
            log_dir = Path(tempfile.mkdtemp(prefix="carrot-keepalive-", dir=log_root))
        log_path = log_dir / f"gpu{device}.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(script),
                    "--device",
                    str(device),
                    "--duration",
                    "0",
                    "--interval",
                    "0.2",
                    "--duty-cycle",
                    "0.1",
                    "--yield-to-others",
                    "--active-duty-cycle",
                    "0.01",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Keepalive failed; inspect {log_path}")
            if f"Started PID={process.pid} " in log_path.read_text():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Keepalive startup not confirmed; inspect {log_path}")
        print(f"Started GPU {device} keepalive PID {process.pid}; log: {log_path}", file=sys.stderr)
        pids.append(process.pid)
    print(" ".join(map(str, pids)))


if __name__ == "__main__":
    main()
