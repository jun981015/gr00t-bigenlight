"""RAID-backed LIBERO setup and N1.7 BC-policy evaluation (stdlib CLI)."""

import argparse
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "external_dependencies/LIBERO"
PACKAGE = SOURCE / "libero/libero"
SOURCE_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
MODEL_REPO = "nvidia/GR00T-N1.7-LIBERO"
MODEL_REVISION = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MODEL_FILES = (
    "config.json",
    "embodiment_id.json",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "processor_config.json",
    "statistics.json",
)


def storage():
    return Path(os.environ.get("VLA_STORAGE_ROOT", str(Path.home() / "raid/vla_finetune")))


def python_path(server=False):
    return storage() / "envs" / ("gr00t-n1.7" if server else "libero-n17-client") / "bin/python"


def task_map():
    # Read the pinned source literal without importing LIBERO's interactive initializer.
    path = PACKAGE / "benchmark/libero_suite_task_map.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"Missing task map: {path}")


def check_sources():
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"LIBERO source revision mismatch: {revision} != {SOURCE_REVISION}")
    if not (PACKAGE / "assets").is_dir():
        raise FileNotFoundError("LIBERO bundled assets are missing")
    tasks = task_map()
    for suite, names in tasks.items():
        for name in names:
            for folder, suffix in (("bddl_files", ".bddl"), ("init_files", ".pruned_init")):
                path = PACKAGE / folder / suite / (name + suffix)
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(path)
    print(
        f"Source OK: LIBERO @ {revision}; {sum(map(len, tasks.values()))} task definitions / init files",
        flush=True,
    )


def config_dict():
    return {
        "benchmark_root": str(PACKAGE),
        "bddl_files": str(PACKAGE / "bddl_files"),
        "init_states": str(PACKAGE / "init_files"),
        "assets": str(PACKAGE / "assets"),
        "datasets": str(storage() / "datasets/libero"),
    }


def configure():
    destination = storage() / "config/libero-n17/config.yaml"
    expected = json.dumps(config_dict(), indent=2) + "\n"  # JSON is valid YAML.
    if destination.exists():
        if destination.read_text() != expected:
            raise RuntimeError(f"Refusing to replace a different config: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as handle:
            handle.write(expected)
    Path(config_dict()["datasets"]).mkdir(parents=True, exist_ok=True)
    print(f"Config ready: {destination}; ~/.libero left untouched", flush=True)


def require_gpu():
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Cannot query GPU; run inside the allocated GPU container") from error
    if result.returncode:
        raise RuntimeError(
            "GPU unavailable in this session; use the allocated container. "
            + result.stderr
            + result.stdout
        )


def selected_tasks(args):
    names = task_map()[args.suite]
    if args.task:
        if args.task not in names:
            raise ValueError(f"Task {args.task!r} does not belong to {args.suite}")
        return [args.task]
    return names if args.all_tasks else names[:1]


def model_path(args):
    return (
        Path(args.model_path).expanduser().resolve()
        if args.model_path
        else storage() / "models/GR00T-N1.7-LIBERO" / args.suite
    )


def build_command(args, task=None, output=None):
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    if args.command == "server":
        return [
            str(python_path(True)),
            str(ROOT / "gr00t/eval/run_gr00t_server.py"),
            "--model-path",
            str(model_path(args)),
            "--embodiment-tag",
            "LIBERO_PANDA",
            "--use-sim-policy-wrapper",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
    if (
        args.episodes < 1
        or not 1 <= args.n_envs <= args.episodes
        or args.max_episode_steps < 1
        or args.action_steps < 1
    ):
        raise ValueError("Require episodes >= n-envs >= 1, positive episode/action steps")
    task = task or selected_tasks(args)[0]
    if task not in task_map()[args.suite]:
        raise ValueError(f"Task {task!r} does not belong to {args.suite}")
    output = Path(output or args.output_dir).expanduser().resolve()
    return [
        str(python_path()),
        str(ROOT / "gr00t/eval/rollout_policy.py"),
        "--policy-client-host",
        "127.0.0.1",
        "--policy-client-port",
        str(args.port),
        "--env-name",
        f"libero_sim/{task}",
        "--n-action-steps",
        str(args.action_steps),
        "--n-episodes",
        str(args.episodes),
        "--n-envs",
        str(args.n_envs),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--seed",
        str(args.seed),
        "--video-dir",
        str(output / "videos"),
    ]


def download(suite):
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=storage() / "models/GR00T-N1.7-LIBERO",
        allow_patterns=[f"{suite}/{name}" for name in MODEL_FILES],
        max_workers=2,
    )
    print(f"Model ready: {suite}; optimizer states excluded")


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sources", "configure", "doctor"):
        sub.add_parser(name)
    for name in ("tasks", "download", "cpu-smoke", "smoke", "server", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--suite", choices=SUITES, default="libero_spatial")
        if name in ("smoke", "cpu-smoke", "eval"):
            group = p.add_mutually_exclusive_group()
            group.add_argument("--task")
            group.add_argument("--all-tasks", action="store_true")
        if name in ("server", "eval"):
            p.add_argument("--port", type=int, default=5556)
            p.add_argument("--execute", action="store_true", help="Default: print commands only")
        if name == "server":
            p.add_argument("--model-path", help="Local BC checkpoint directory, not an SVF .pt")
        if name == "eval":
            p.add_argument("--episodes", type=int, default=10, help="Episodes per task")
            p.add_argument("--n-envs", type=int, default=1)
            p.add_argument("--action-steps", type=int, default=8)
            p.add_argument("--max-episode-steps", type=int, default=720)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument(
                "--output-dir",
                default=str(
                    storage() / "outputs/libero-n17" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                ),
            )
    return parser


def main():
    args = parser().parse_args()
    if args.command == "sources":
        check_sources()
    elif args.command == "configure":
        check_sources()
        configure()
    elif args.command == "tasks":
        print(json.dumps(task_map()[args.suite], indent=2))
    elif args.command == "download":
        download(args.suite)
    elif args.command in ("doctor", "cpu-smoke", "smoke"):
        check_sources()
        if args.command == "smoke":
            require_gpu()
        command = [str(python_path()), str(HERE / "check.py"), args.command]
        if args.command != "doctor":
            command += ["--suite", args.suite]
            if args.task:
                command += ["--task", args.task]
            elif args.all_tasks:
                command += ["--all-tasks"]
        subprocess.run(command, cwd=ROOT, check=True, timeout=1800)
    elif args.command == "server":
        command = build_command(args)
        print(shlex.join(command), flush=True)
        if args.execute:
            require_gpu()
            if not (model_path(args) / "config.json").is_file():
                raise FileNotFoundError(f"Download {args.suite} first or set --model-path")
            os.execv(command[0], command)
    else:
        output = Path(args.output_dir).expanduser().resolve()
        commands = [
            (task, build_command(args, task, output / task)) for task in selected_tasks(args)
        ]
        for _, command in commands:
            print(shlex.join(command), flush=True)
        if args.execute:
            check_sources()
            require_gpu()
            output.mkdir(parents=True, exist_ok=False)
            (output / "run.json").write_text(
                json.dumps(
                    {"args": vars(args), "source_revision": SOURCE_REVISION, "commands": commands},
                    indent=2,
                )
                + "\n"
            )
            for task, command in commands:
                task_dir = output / task
                task_dir.mkdir()
                print(f"Evaluation log: {task_dir / 'eval.log'}", flush=True)
                with (task_dir / "eval.log").open("x") as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                (task_dir / "exit_code").write_text(str(result.returncode) + "\n")
                if result.returncode:
                    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
