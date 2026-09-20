"""CPU-safe preparation and explicit, dry-run-first launch for the DEAS recipe."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECIPE = json.loads((HERE / "recipe.json").read_text())


def storage_root():
    return Path(os.environ.get("VLA_STORAGE_ROOT", Path.home() / "raid/vla_finetune")).resolve()


def asset_path(name, root=None):
    return (root or storage_root()) / RECIPE["assets"][name]["directory"]


def datasets(stage, root=None):
    if stage == "bc24":
        return [asset_path("bc24", root)]
    groups = ("demos", "success_rollouts" if stage == "filtered-bc" else "rollouts")
    return [asset_path("offline4", root) / group / task for group in groups for task in RECIPE["tasks"]]


def validate_dataset(path, require_rl=False, full=False):
    """Validate metadata, task annotations, and optionally every declared episode file."""
    path = Path(path)
    info = json.loads((path / "meta/info.json").read_text())
    modality = json.loads((path / "meta/modality.json").read_text())
    if not info["codebase_version"].startswith("v2"):
        raise ValueError(f"Expected LeRobot v2: {path}")
    required = {"observation.state", "action", "task_index"}
    if require_rl:
        required |= {"next.reward", "next.done"}
        if not modality.get("reward") or not modality.get("done"):
            raise ValueError(f"Missing reward/done modality: {path}")
    if not required <= info["features"].keys():
        raise ValueError(f"Missing fields {required - info['features'].keys()}: {path}")
    if "human.action.task_description" not in modality.get("annotation", {}):
        raise ValueError(f"Missing task description mapping: {path}")
    tasks = [json.loads(line) for line in (path / "meta/tasks.jsonl").read_text().splitlines() if line]
    if not tasks or any(not str(t.get("task", "")).strip() for t in tasks):
        raise ValueError(f"Missing task instructions: {path}")
    episodes = [json.loads(line) for line in (path / "meta/episodes.jsonl").read_text().splitlines() if line]
    if len(episodes) != info["total_episodes"]:
        raise ValueError(f"Episode inventory mismatch: {path}")
    for episode in (episodes if full else episodes[:1]):
        idx = episode["episode_index"]
        fields = {"episode_index": idx, "episode_chunk": idx // info["chunks_size"]}
        expected = [path / info["data_path"].format(**fields)]
        # GR00T uses external mp4 files even where legacy info.json labels images as images.
        for key, spec in modality.get("video", {}).items():
            video_key = spec.get("original_key", f"observation.images.{key}")
            expected.append(path / info["video_path"].format(**fields, video_key=video_key))
        for filename in expected:
            if not filename.is_file() or filename.stat().st_size == 0:
                raise ValueError(f"Missing/empty episode asset: {filename}")
    return {"path": str(path), "episodes": len(episodes), "frames": info["total_frames"],
            "language_variants": len(tasks), "rl": require_rl}


def download(names, metadata_only=False):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    for name in names:
        spec = RECIPE["assets"][name]
        print(f"Downloading {name}: {spec['repo_id']} @ {spec['revision']}", flush=True)
        target = asset_path(name)
        target.mkdir(parents=True, exist_ok=True)
        for attempt in range(20):
            try:
                snapshot_download(repo_id=spec["repo_id"], repo_type=spec["repo_type"], revision=spec["revision"],
                                  local_dir=str(target), max_workers=2,
                                  allow_patterns=["meta/*", "*/meta/*"] if metadata_only else None)
                break
            except HfHubHTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status not in (429, 500, 502, 503, 504) or attempt == 19:
                    raise
                # Respect server rate limits; cached/partial files are reused on retry.
                retry_after = exc.response.headers.get("Retry-After", "300")
                delay = max(300, int(retry_after)) if retry_after.isdigit() else 300
                print(f"HTTP {status}: retry {attempt + 1}/20 after {delay}s (preserving downloaded files)", flush=True)
                time.sleep(delay)
        # Only a fully returned snapshot is marked ready; metadata-only downloads are not ready.
        if not metadata_only:
            (target / "download_complete.json").write_text(json.dumps(spec, indent=2) + "\n")
        print(f"Completed {name}: {target}", flush=True)


def training_config(args):
    cfg = dict(RECIPE["training"])
    cfg.update(dataset_path=[str(p) for p in datasets(args.stage)], output_dir=str(args.output.resolve()),
               run_name=args.output.name, data_config="single_panda_gripper")
    if args.stage == "bc24":
        cfg["base_model_path"] = str(asset_path("model"))
    else:
        if args.base_model is None:
            raise ValueError("--base-model is required: initial BC checkpoint for filtered-bc; actor checkpoint for critic")
        cfg["base_model_path"] = str(args.base_model.resolve())
    if args.stage == "critic":
        cfg.update(RECIPE["critic"])
        cfg["data_config"] = "single_panda_gripper_rl"
        algorithm = getattr(args, "critic_algorithm", None) or "deas"
        if algorithm == "iql":
            cfg.update(critic_algorithm="iql", iql_discount=0.99,
                       negative_reward=False, num_atoms=1, nstep=1, q_agg="min")
    elif getattr(args, "critic_algorithm", None) is not None:
        raise ValueError("--critic-algorithm is only valid for the critic stage")
    for key in ("max_steps", "batch_size", "num_gpus"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    cfg["resume"] = bool(args.resume)
    if getattr(args, "save_steps", None) is not None:
        if args.save_steps <= 0:
            raise ValueError("save_steps must be positive")
        cfg["save_steps"] = args.save_steps
    if getattr(args, "video_backend", None) is not None:
        cfg["video_backend"] = args.video_backend
    if getattr(args, "logging_steps", None) is not None:
        if args.logging_steps <= 0:
            raise ValueError("logging_steps must be positive")
        cfg["logging_steps"] = args.logging_steps
    return cfg


def launch_train(args):
    cfg = training_config(args)
    python = storage_root() / "envs/deas-gr00t-n1.5/bin/python"
    command = [str(python), "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={cfg['num_gpus']}", str(HERE / "worker.py"),
               "--stage", args.stage, "--config", json.dumps(cfg)]
    print(json.dumps({"stage": args.stage, "config": cfg, "recovery": RECIPE["recovery"],
                      "effective_batch": cfg["num_gpus"] * cfg["batch_size"]}, indent=2))
    print(shlex.join(command))
    if not args.execute:
        print("DRY RUN: add --execute inside the GPU container to start.")
        return
    if not Path("/.dockerenv").exists():
        raise ValueError("Run GPU training inside the allocated container, not the login host.")
    if not python.is_file():
        raise ValueError("Install the isolated environment with bootstrap.sh first.")
    for path in datasets(args.stage):
        print(validate_dataset(path, require_rl=args.stage == "critic", full=True))
    if not (Path(cfg["base_model_path"]) / "config.json").is_file():
        raise ValueError("Missing base model/checkpoint config.json")
    if args.resume:
        if not (args.output / "latest_resumable.json").is_file():
            raise ValueError("No committed full checkpoint to resume")
        previous = json.loads((args.output / "launch_recipe.json").read_text())["config"]
        # Checkpoint cadence can change without changing optimizer/scheduler semantics.
        resume_mutable = {"resume", "save_steps", "logging_steps"}
        if {k: v for k, v in previous.items() if k not in resume_mutable} != {k: v for k, v in cfg.items() if k not in resume_mutable}:
            raise ValueError("Resume must use the original configuration, including batch size and max steps")
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty output directory; use --resume or a new path")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO), WANDB_PROJECT=RECIPE["wandb_project"], WANDB_DIR=str(args.output.resolve()),
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", USE_TF="0", NO_ALBUMENTATIONS_UPDATE="1")
    with (args.output / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.output / "launch_recipe.json").write_text(json.dumps({"recipe": RECIPE, "config": cfg}, indent=2))
        print(f"Training log: {args.output / 'train.log'}", flush=True)
        with (args.output / "train.log").open("a") as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        (args.output / "train.exit").write_text(str(result.returncode) + "\n")
    raise SystemExit(result.returncode)


def launch_eval(args):
    python = storage_root() / "envs/deas-gr00t-n1.5/bin/python"
    command = [str(python), str(REPO / "scripts/eval_policy_robocasa.py"),
               "--actor_model_path", str(args.actor.resolve()), "--model_type", "deas" if args.critic else "gr00tn15",
               "--data_config", "single_panda_gripper_rl_inference", "--embodiment_tag", "new_embodiment",
               "--action_horizon", "16", "--env_name", args.task, "--noise", "0.0", "--seed", str(args.seed),
               "--output_path", str(args.output.resolve())]
    for key, value in RECIPE["evaluation"].items():
        command += [f"--{key}", str(value)]
    if args.critic:
        command += ["--critic_model_path", str(args.critic.resolve())]
    print(shlex.join(command))
    if args.execute:
        if not Path("/.dockerenv").exists():
            raise ValueError("Run evaluation inside the GPU container after simulator smoke testing")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Use a new evaluation output directory")
        env = os.environ.copy()
        env.update(MUJOCO_GL="egl", PYTHONPATH=str(REPO))
        raise SystemExit(subprocess.run(command, cwd=REPO, env=env).returncode)
    print("DRY RUN: simulator installation and EGL smoke test are required before --execute.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("assets", nargs="+", choices=list(RECIPE["assets"]))
    dl.add_argument("--metadata-only", action="store_true")
    check = sub.add_parser("check")
    check.add_argument("--full", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("stage", choices=["bc24", "filtered-bc", "critic"])
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--base-model", type=Path)
    for key in ("max-steps", "batch-size", "num-gpus"):
        train.add_argument(f"--{key}", type=int)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--save-steps", type=int)
    train.add_argument("--logging-steps", type=int)
    train.add_argument("--critic-algorithm", choices=["deas", "iql"])
    train.add_argument("--video-backend", choices=["decord", "torchvision_av"])
    train.add_argument("--execute", action="store_true")
    ev = sub.add_parser("eval")
    ev.add_argument("--task", choices=RECIPE["tasks"], required=True)
    ev.add_argument("--actor", type=Path, required=True)
    ev.add_argument("--critic", type=Path)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        download(args.assets, args.metadata_only)
    elif args.command == "check":
        for stage in ("bc24", "critic", "filtered-bc"):
            for path in datasets(stage):
                print(json.dumps(validate_dataset(path, require_rl=stage == "critic", full=args.full)))
    elif args.command == "train":
        launch_train(args)
    elif args.command == "eval":
        launch_eval(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))
