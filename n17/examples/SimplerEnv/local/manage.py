"""Local, RAID-backed SimplerEnv preparation and N1.7 evaluation commands."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIMPLER = ROOT / "external_dependencies/SimplerEnv"
MANISKILL = SIMPLER / "ManiSkill2_real2sim"
PINS = {
    SIMPLER: "8a2d286c926c1371927caa7651a412b4cc331756",
    MANISKILL: "c2a9e87c186300b694da6f2497dd68d2c347a4b7",
}
ROBOTS = {
    "widowx": {
        "embodiment": "SIMPLER_ENV_WIDOWX",
        "checkpoint": "nvidia/GR00T-N1.7-SimplerEnv-Bridge",
        "revision": "940134b3c2948ccfdf8e7393f2d2ca869dc42833",
        "action_steps": 4,
        "tasks": [
            "widowx_spoon_on_towel",
            "widowx_carrot_on_plate",
            "widowx_stack_cube",
            "widowx_put_eggplant_in_basket",
            "widowx_put_eggplant_in_sink",
            "widowx_open_drawer",
            "widowx_close_drawer",
        ],
    },
    "google": {
        "embodiment": "SIMPLER_ENV_GOOGLE",
        "checkpoint": "nvidia/GR00T-N1.7-SimplerEnv-Fractal",
        "revision": "fb8357ff66e3cc8dec369e2bf7dbe9e6740511b2",
        "action_steps": 1,
        "tasks": [
            "google_robot_pick_coke_can",
            "google_robot_pick_object",
            "google_robot_move_near",
            "google_robot_open_drawer",
            "google_robot_close_drawer",
            "google_robot_place_in_closed_drawer",
        ],
    },
}


def storage():
    return Path(os.environ.get("VLA_STORAGE_ROOT", str(Path.home() / "raid/vla_finetune")))


def python_path(server=False):
    name = "gr00t-n1.7" if server else "simpler-n17-client"
    return storage() / "envs" / name / "bin/python"


def check_sources():
    for path, expected in PINS.items():
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            raise RuntimeError(f"Source revision mismatch: {path}: {actual} != {expected}")
        print(f"Source OK: {path.name} @ {actual}", flush=True)
    for relative in (
        "data/custom",
        "data/hab2_bench_assets",
        "data/real_inpainting",
        "mani_skill2_real2sim/assets",
    ):
        path = MANISKILL / relative
        if not path.is_dir() or not any(path.iterdir()):
            raise FileNotFoundError(f"Missing bundled assets: {path}")


def require_gpu():
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            "GPU not accessible in this session; run inside the allocated GPU container. "
            + result.stderr
            + result.stdout
        )


def build_command(args):
    robot = ROBOTS[args.robot]
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    if args.command == "server":
        model = args.model_path or str(storage() / "models" / robot["checkpoint"].split("/")[-1])
        return [
            str(python_path(True)),
            str(ROOT / "gr00t/eval/run_gr00t_server.py"),
            "--model-path",
            model,
            "--embodiment-tag",
            robot["embodiment"],
            "--use-sim-policy-wrapper",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
    task = args.task or robot["tasks"][0]
    if task not in robot["tasks"]:
        raise ValueError(f"Task {task!r} is not registered for {args.robot}")
    if args.episodes < 1 or not 1 <= args.n_envs <= args.episodes or args.max_episode_steps < 1:
        raise ValueError("Require episodes >= n-envs >= 1 and max-episode-steps >= 1")
    action_steps = args.action_steps if args.action_steps is not None else robot["action_steps"]
    if action_steps < 1:
        raise ValueError("action-steps must be positive")
    return [
        str(python_path()),
        str(ROOT / "gr00t/eval/rollout_policy.py"),
        "--policy-client-host",
        "127.0.0.1",
        "--policy-client-port",
        str(args.port),
        "--env-name",
        f"simpler_env_{args.robot}/{task}",
        "--n-action-steps",
        str(action_steps),
        "--n-episodes",
        str(args.episodes),
        "--n-envs",
        str(args.n_envs),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--seed",
        str(args.seed),
        "--video-dir",
        str(Path(args.output_dir) / "videos"),
    ]


def download(robot_name):
    # Executed by N1.7 Python; simulator env never installs model-training dependencies.
    from huggingface_hub import snapshot_download

    robot = ROBOTS[robot_name]
    destination = storage() / "models" / robot["checkpoint"].split("/")[-1]
    snapshot_download(
        robot["checkpoint"],
        revision=robot["revision"],
        local_dir=destination,
        max_workers=2,
        ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state*.pth", "global_step*/*"],
    )
    print(f"Checkpoint ready: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sources", "doctor", "tasks"):
        sub.add_parser(name)
    p = sub.add_parser("download")
    p.add_argument("--robot", choices=ROBOTS, default="widowx")
    p = sub.add_parser("smoke")
    p.add_argument("--robot", choices=ROBOTS, default="widowx")
    p.add_argument("--all-tasks", action="store_true")
    for name in ("server", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--robot", choices=ROBOTS, default="widowx")
        p.add_argument("--port", type=int, default=5555)
        p.add_argument("--execute", action="store_true", help="Default: print command only")
        if name == "server":
            p.add_argument("--model-path", help="Local BC checkpoint directory (not an SVF .pt)")
        else:
            p.add_argument("--task")
            p.add_argument("--episodes", type=int, default=10)
            p.add_argument("--n-envs", type=int, default=1)
            p.add_argument("--action-steps", type=int)
            p.add_argument("--max-episode-steps", type=int, default=300)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument(
                "--output-dir",
                default=str(
                    storage() / "outputs/simpler-n17" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                ),
            )
    args = parser.parse_args()
    if args.command == "sources":
        check_sources()
    elif args.command == "tasks":
        print(json.dumps(ROBOTS, indent=2))
    elif args.command == "download":
        download(args.robot)
    elif args.command in ("doctor", "smoke"):
        check_sources()
        command = [str(python_path()), str(HERE / "check.py"), args.command]
        if args.command == "smoke":
            require_gpu()
            command += ["--robot", args.robot]
            if args.all_tasks:
                command += ["--all-tasks"]
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        command = build_command(args)
        print(shlex.join(command), flush=True)
        if args.execute:
            check_sources()
            require_gpu()
            if args.command == "server":
                model = Path(command[command.index("--model-path") + 1])
                if not (model / "config.json").is_file():
                    raise FileNotFoundError(
                        f"Download a checkpoint first, or set --model-path: {model}"
                    )
                os.execv(command[0], command)
            else:
                output = Path(args.output_dir)
                output.mkdir(parents=True, exist_ok=False)
                (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                with (output / "eval.log").open("x") as log:
                    print(f"Evaluation log: {output / 'eval.log'}", flush=True)
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                (output / "exit_code").write_text(str(result.returncode) + "\n")
                sys.exit(result.returncode)


if __name__ == "__main__":
    main()
