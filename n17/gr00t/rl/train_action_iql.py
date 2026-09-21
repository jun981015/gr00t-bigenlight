"""Train only action-reinjected Q/V on the existing frozen BC feature cache."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import signal

import numpy as np
import torch

from .action_conditioned_iql import (
    ActionConditionedQEnsemble,
    ActionIQLConfig,
    ActionIQLLearner,
    StateValue,
)
from .dataset import EpisodeBatchSampler
from .feature_cache import CachedFeatureDataset, atomic_json
from .trainer import OfflineTrainer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reward", choices=["step-cost", "terminal-success"], default="step-cost")
    p.add_argument(
        "--assume-all-success",
        action="store_true",
        required=True,
        help="Required: current cache reward presets assume every episode succeeds",
    )
    p.add_argument("--gamma", type=float, default=0.99, help="Per-primitive-action discount")
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--value-lr", type=float, default=1e-4)
    p.add_argument("--expectile-tau", type=float, default=0.8)
    p.add_argument("--target-tau", type=float, default=0.005)
    p.add_argument("--num-q-heads", type=int, default=10)
    p.add_argument(
        "--exclude-proprio",
        action="store_true",
        help="Drop cached N1.7 state columns 2048:2180 in Q, V and EMA V",
    )
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 512, 256])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--perturb-sigma", type=float, default=0.05)
    p.add_argument("--diagnostic-beta", type=float, default=1.0)
    p.add_argument("--wandb-project")
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--debug-fixed-batch", action="store_true", help="Overfit one batch, NOT an experiment"
    )
    args = p.parse_args(argv)
    if (
        min(args.steps, args.batch_size, args.log_every, args.cpu_threads) < 1
        or args.save_every < 0
        or args.seed < 0
    ):
        raise ValueError("Invalid run settings")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError("Use a fresh output directory")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds = CachedFeatureDataset(args.cache, reward=args.reward, gamma=args.gamma)
    config = ActionIQLConfig(
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        expectile_tau=args.expectile_tau,
        target_tau=args.target_tau,
        diagnostic_every=args.log_every,
        perturb_sigma=args.perturb_sigma,
        diagnostic_beta=args.diagnostic_beta,
        assume_all_success=True,
    )
    q = ActionConditionedQEnsemble(
        ds.feature_dim,
        ds.action_shape,
        ds.action_indices,
        args.hidden_dims,
        args.num_q_heads,
        exclude_proprio=args.exclude_proprio,
    ).to(args.device)
    v = StateValue(ds.feature_dim, args.hidden_dims, exclude_proprio=args.exclude_proprio).to(
        args.device
    )
    learner = ActionIQLLearner(q, v, config)
    semantic = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("output_dir", "resume", "steps", "save_every", "wandb_project", "cpu_threads")
    }
    if not args.exclude_proprio:
        semantic.pop("exclude_proprio")  # Preserve existing run/checkpoint resume identity.
    metadata = {
        "backend": "action-conditioned-iql-cache-v1",
        "args": semantic,
        "cache_manifest_sha256": hashlib.sha256(
            (args.cache / "manifest.json").read_bytes()
        ).hexdigest(),
        "bc_identity": ds.manifest["identity"],
        "config": asdict(config),
        "feature_dim": ds.feature_dim,
        "action_shape": list(ds.action_shape),
        "action_indices": ds.action_indices.tolist(),
        "q_layout": "batch,heads",
        "discount": "sum(gamma**i * r_i) + gamma**H * (1-terminal) * V_target(s_next)",
        "success_labels": "assumed all episodes successful, NOT measured",
        "trainable": "Q ensemble and V only; no actor/backbone loaded; FP32",
    }
    trainer = OfflineTrainer(learner, args.device, metadata=metadata)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    if args.steps <= trainer.step:
        raise ValueError("Target steps must exceed resumed step")
    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists() and any(
        json.loads(s)["step"] > trainer.step
        for s in metrics_path.read_text().splitlines()
        if s.strip()
    ):
        raise ValueError("Metrics ahead of resume point; use a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_file = args.output_dir / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()) != metadata:
        raise ValueError("Output metadata mismatch")
    if not run_file.exists():
        atomic_json(run_file, metadata)
    wb = None
    if args.wandb_project:
        import wandb

        identity_path = args.output_dir / "wandb_resume.json"
        if not identity_path.exists():
            atomic_json(
                identity_path, {"id": wandb.util.generate_id(), "project": args.wandb_project}
            )
        identity = json.loads(identity_path.read_text())
        if identity["project"] != args.wandb_project:
            raise ValueError("W&B project mismatch")
        wb = wandb.init(
            project=args.wandb_project,
            id=identity["id"],
            resume="allow",
            mode="online",
            name=args.output_dir.name,
            dir=str(args.output_dir),
            config=metadata,
        )

    def log(metrics):
        if wb and (
            metrics["step"] == 1
            or metrics["step"] % args.log_every == 0
            or metrics["step"] == args.steps
        ):
            wb.log(metrics, step=metrics["step"])

    previous = {}

    def stop(signum, frame):
        trainer.stop_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, stop)
    completed = False
    try:
        sampler = EpisodeBatchSampler(
            ds, args.batch_size, args.steps - trainer.step, args.seed, trainer.step
        )
        if args.debug_fixed_batch:
            indices = next(iter(EpisodeBatchSampler(ds, args.batch_size, 1, args.seed)))
            fixed = ds.batch(indices)
            batches = (fixed for _ in range(args.steps - trainer.step))
        else:
            batches = (ds.batch(indices) for indices in sampler)
        trainer.fit(
            batches,
            log_path=metrics_path,
            checkpoint_dir=args.output_dir / "checkpoints",
            save_every=args.save_every,
            max_run_seconds=28800,
            keep_latest_training_state=True,
            metrics_callback=log,
        )
        if not (args.output_dir / f"checkpoints/step-{trainer.step}.pt").exists():
            trainer.save_recovery_checkpoint(
                args.output_dir / "checkpoints", True, archive_model=True
            )
        completed = True
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if wb:
            wb.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
