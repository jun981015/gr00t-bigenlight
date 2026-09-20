"""Refuse concurrent training jobs, except explicitly identified user-owned keepalives."""

import argparse
import os
from pathlib import Path
import subprocess


def allowed_keepalives(value, *, proc_root=Path("/proc"), expected_script=None):
    expected = str(expected_script or (Path.home() / "vla_finetune/gpu_keepalive.py"))
    allowed = set()
    for token in value.split():
        if not token.isdecimal() or int(token) <= 1:
            raise ValueError("CARROT_KEEPALIVE_PIDS must contain positive process IDs")
        pid = int(token)
        process = proc_root / str(pid)
        if process.stat().st_uid != os.getuid():
            raise ValueError(f"Keepalive PID {pid} is not owned by the current user")
        command = (process / "cmdline").read_bytes().split(b"\0")
        if os.fsencode(expected) not in command:
            raise ValueError(f"PID {pid} is not the expected gpu_keepalive.py process")
        allowed.add(pid)
    return allowed


def busy_pids(rows, allowed, gpu_uuid=None):
    busy = set()
    for row in rows.splitlines():
        if not row.strip():
            continue
        pid, uuid = (part.strip() for part in row.split(",", 1))
        if gpu_uuid is None or uuid == gpu_uuid:
            busy.add(int(pid))
    return busy - allowed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", help="Check only this GPU; default checks every GPU")
    args = parser.parse_args()
    if args.gpu_uuid is not None:
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10
        ).splitlines()
        if args.gpu_uuid not in {uuid.strip() for uuid in devices}:
            raise ValueError("Requested GPU UUID is not visible in this container")
    allowed = allowed_keepalives(os.environ.get("CARROT_KEEPALIVE_PIDS", ""))
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    )
    busy = busy_pids(result, allowed, args.gpu_uuid)
    if busy:
        raise SystemExit(f"Existing GPU jobs detected: {sorted(busy)}. No processes were stopped.")
    print(f"GPU guard passed; permitted keepalive PIDs: {sorted(allowed)}", flush=True)


if __name__ == "__main__":
    main()
