"""Run a low-duty-cycle CUDA workload without requiring PyTorch."""

import argparse
import ctypes
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--duration", type=float, default=7200, help="Seconds; 0 runs until stopped"
    )
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--duty-cycle", type=float, default=0.2)
    parser.add_argument("--yield-to-others", action="store_true")
    parser.add_argument("--active-duty-cycle", type=float, default=0.01)
    parser.add_argument("--kernel", type=Path, default=Path(__file__).with_suffix(".ptx"))
    args = parser.parse_args()
    if args.duration < 0 or args.interval <= 0:
        parser.error("duration must be nonnegative and interval must be positive")
    if not 0 < args.duty_cycle <= 1:
        parser.error("duty-cycle must be in (0, 1]")
    if not 0 < args.active_duty_cycle <= args.duty_cycle:
        parser.error("active-duty-cycle must be in (0, duty-cycle]")
    if not args.kernel.is_file():
        parser.error(f"compiled CUDA kernel not found: {args.kernel}")

    # One lock per GPU; preserve the lock held by the existing GPU 0 process.
    pid_suffix = ".pid" if args.device == 0 else f".gpu{args.device}.pid"
    with Path(__file__).with_suffix(pid_suffix).open("a+") as pidfile:
        try:
            fcntl.flock(pidfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"A keepalive for GPU {args.device} is already running.")
        pidfile.seek(0)
        pidfile.truncate()
        pidfile.write(str(os.getpid()) + "\n")
        pidfile.flush()

        stopped = threading.Event()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stopped.set())

        cuda = ctypes.CDLL("libcuda.so.1")
        signatures = {
            "cuInit": [ctypes.c_uint],
            "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
            "cuCtxSetCurrent": [ctypes.c_void_p],
            "cuMemAlloc_v2": [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
            "cuModuleLoad": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p],
            "cuModuleGetFunction": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_char_p,
            ],
            "cuLaunchKernel": [ctypes.c_void_p]
            + [ctypes.c_uint] * 7
            + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)],
            "cuModuleUnload": [ctypes.c_void_p],
            "cuCtxSynchronize": [],
            "cuMemFree_v2": [ctypes.c_uint64],
            "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
        }
        for name, argtypes in signatures.items():
            getattr(cuda, name).argtypes = argtypes
            getattr(cuda, name).restype = ctypes.c_int

        def call(name, *values):
            status = getattr(cuda, name)(*values)
            if status:
                raise RuntimeError(f"{name} failed with CUDA status {status}")

        call("cuInit", 0)
        device = ctypes.c_int()
        context = ctypes.c_void_p()
        memory = ctypes.c_uint64()
        module = ctypes.c_void_p()
        kernel = ctypes.c_void_p()
        call("cuDeviceGet", ctypes.byref(device), args.device)
        call("cuDevicePrimaryCtxRetain", ctypes.byref(context), device)
        try:
            call("cuCtxSetCurrent", context)
            size = 16 * 1024 * 1024
            call("cuMemAlloc_v2", ctypes.byref(memory), size)
            call("cuModuleLoad", ctypes.byref(module), os.fsencode(args.kernel))
            call("cuModuleGetFunction", ctypes.byref(kernel), module, b"keepalive_compute")
            iterations = ctypes.c_int(65536)
            kernel_args = (ctypes.c_void_p * 2)(
                ctypes.cast(ctypes.byref(memory), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(iterations), ctypes.c_void_p),
            )
            print(
                f"Started PID={os.getpid()} GPU={args.device}, buffer=16 MiB, "
                f"compute duty={args.duty_cycle:.0%}, interval={args.interval}s, "
                f"duration={args.duration}s",
                flush=True,
            )
            deadline = time.monotonic() + args.duration if args.duration else float("inf")
            gpu_uuid = None
            if args.yield_to_others:
                gpu_uuid = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={args.device}",
                        "--query-gpu=uuid",
                        "--format=csv,noheader",
                    ],
                    text=True,
                    timeout=5,
                ).strip()
            next_check, duty_cycle = 0.0, args.duty_cycle
            while not stopped.is_set() and time.monotonic() < deadline:
                cycle_start = time.monotonic()
                if gpu_uuid and cycle_start >= next_check:
                    try:
                        apps = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-compute-apps=gpu_uuid,pid",
                                "--format=csv,noheader",
                            ],
                            text=True,
                            timeout=5,
                        )
                        other_job = any(
                            fields[0].strip() == gpu_uuid and int(fields[1]) != os.getpid()
                            for line in apps.splitlines()
                            if len(fields := line.split(",")) == 2
                        )
                        next_duty = args.active_duty_cycle if other_job else args.duty_cycle
                        if next_duty != duty_cycle:
                            print(
                                f"GPU={args.device} other_job={other_job}; target duty={next_duty:.0%}",
                                flush=True,
                            )
                        duty_cycle = next_duty
                    except (subprocess.SubprocessError, ValueError, OSError) as exc:
                        print(
                            f"Activity check failed ({type(exc).__name__}); retaining low duty",
                            flush=True,
                        )
                        duty_cycle = args.active_duty_cycle
                    next_check = cycle_start + 10
                active_until = min(deadline, cycle_start + args.interval * duty_cycle)
                while not stopped.is_set() and time.monotonic() < active_until:
                    call("cuLaunchKernel", kernel, 132, 1, 1, 128, 1, 1, 0, None, kernel_args, None)
                    call("cuCtxSynchronize")
                stopped.wait(max(0, min(deadline, cycle_start + args.interval) - time.monotonic()))
        finally:
            if memory.value:
                call("cuMemFree_v2", memory)
            if module.value:
                call("cuModuleUnload", module)
            call("cuDevicePrimaryCtxRelease_v2", device)
            print("Stopped; GPU context released.", flush=True)


if __name__ == "__main__":
    main()
