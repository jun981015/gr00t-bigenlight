"""IQL on completed frozen-BC caches. Does not load a VLM, DiT or tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import signal

import numpy as np
import torch

from .dataset import EpisodeBatchSampler
from .feature_cache import CachedFeatureDataset, atomic_json
from .iql import FeatureValue, IQLConfig, IQLCriticLearner
from .networks import FeatureCritic
from .trainer import OfflineTrainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["iql", "deas"], default="iql")
    parser.add_argument("--iql-encoder", choices=["none", "deas-mlp"], default="none")
    parser.add_argument("--discount1", type=float, default=0.9)
    parser.add_argument("--discount2", type=float, default=0.99)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reward", choices=["terminal-success", "step-cost"], default="terminal-success"
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-project")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Trusted checkpoint from this cached trainer, not a live-VLM run",
    )
    args = parser.parse_args(argv)
    if args.algorithm != "iql" and args.iql_encoder != "none":
        raise ValueError("--iql-encoder applies only to scalar IQL")
    if (
        min(args.steps, args.batch_size, args.hidden_dim, args.hidden_layers, args.log_every) < 1
        or args.seed < 0
        or args.save_every < 0
    ):
        raise ValueError("Invalid training settings")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise FileExistsError("Use a fresh output directory")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.algorithm == "deas":
        from .deas_cached import DEASCachedLearner, DEASConfig, DEASDataset

        if args.reward != "step-cost":
            raise ValueError("DEAS cached backend requires step-cost rewards")
        dataset = DEASDataset(args.cache, discount1=args.discount1, discount2=args.discount2)
        bc_config = json.loads(
            (Path(dataset.manifest["identity"]["model_path"]) / "config.json").read_text()
        )
        config = DEASConfig(
            learning_rate=args.learning_rate,
            expectile=args.expectile,
            target_tau=args.target_tau,
            discount1=args.discount1,
            discount2=args.discount2,
            vlm_dim=bc_config["backbone_embedding_dim"],
            state_dim=bc_config["max_state_dim"],
            embodiment_dim=bc_config["max_num_embodiments"],
            hidden_dim=args.hidden_dim,
            depth=args.hidden_layers,
        )
        learner = DEASCachedLearner(dataset.action_indices, config, args.device)
    else:
        dataset = CachedFeatureDataset(args.cache, reward=args.reward, gamma=args.gamma)
        hidden = (args.hidden_dim,) * args.hidden_layers
        config = IQLConfig(args.learning_rate, args.expectile, args.target_tau)
        if args.iql_encoder == "deas-mlp":
            from .projected_iql import ProjectedIQLQ, ProjectedIQLV, pooled_encoder

            if dataset.feature_dim != 2212:
                raise ValueError("DEAS-style IQL encoder requires N1.7 2048+132+32 features")
            projection = pooled_encoder().to(args.device)
            critic = ProjectedIQLQ(projection, dataset.feature_dim, dataset.action_shape, hidden)
            value = ProjectedIQLV(projection, dataset.feature_dim, hidden)
        else:
            critic = FeatureCritic(dataset.feature_dim, dataset.action_shape, hidden)
            value = FeatureValue(dataset.feature_dim, hidden)
        learner = IQLCriticLearner(critic.to(args.device), value.to(args.device), config)
    semantic = {
        k: v
        for k, v in vars(args).items()
        if k
        not in [
            "output_dir",
            "cache",
            "steps",
            "resume",
            "wandb_project",
            "log_every",
            "save_every",
        ]
    }
    if args.iql_encoder == "none":
        semantic.pop("iql_encoder")  # Backwards-compatible checkpoint metadata.
    metadata = {
        "backend": "frozen-bc-cache-v1",
        "cache_manifest_sha256": hashlib.sha256(
            (args.cache / "manifest.json").read_bytes()
        ).hexdigest(),
        "args": semantic,
        "bc_identity": dataset.manifest["identity"],
    }
    if args.algorithm == "iql":
        # Preserve exact metadata identity of existing IQL runs/checkpoints.
        for key in ("algorithm", "discount1", "discount2"):
            semantic.pop(key)
    else:
        semantic.pop("gamma")
        from dataclasses import asdict

        metadata["deas_config"] = asdict(config)
        metadata["frontend"] = "N1.7 frozen masked pooled VLM; no N1.5 token attention"
    trainer = OfflineTrainer(learner, args.device, metadata=metadata)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    if args.steps <= trainer.step:
        raise ValueError("Target steps must exceed checkpoint step")
    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists() and any(
        json.loads(line)["step"] > trainer.step
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ):
        raise ValueError("Metrics are ahead of checkpoint; use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["metadata"] != metadata:
        raise ValueError("Output run metadata mismatch")
    if not run_path.exists():
        atomic_json(
            run_path,
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "metadata": metadata,
                "feature_dim": dataset.feature_dim,
                "action_shape": list(dataset.action_shape),
                "trainable": (
                    "new DEAS projection, Q1/Q2 and residual V; BC frozen"
                    if args.algorithm == "deas"
                    else "new Q1/Q2, IQL V and optional shared pooled encoder; no BC model loaded"
                ),
            },
        )
    wandb_run = None
    if args.wandb_project:
        import wandb

        identity_path = args.output_dir / "wandb_resume.json"
        if not identity_path.exists():
            atomic_json(
                identity_path, {"id": wandb.util.generate_id(), "project": args.wandb_project}
            )
        identity = json.loads(identity_path.read_text())
        if identity["project"] != args.wandb_project:
            raise ValueError("W&B project changed")
        wandb_run = wandb.init(
            project=args.wandb_project,
            id=identity["id"],
            resume="allow",
            mode="online",
            name=args.output_dir.name,
            dir=str(args.output_dir),
            config=vars(args),
        )
    previous = {}

    def stop(signum, frame):
        trainer.stop_requested = True

    for signum in [signal.SIGTERM, signal.SIGINT]:
        previous[signum] = signal.signal(signum, stop)
    completed = False
    try:
        batches = (
            dataset.batch(indices)
            for indices in EpisodeBatchSampler(
                dataset, args.batch_size, args.steps - trainer.step, args.seed, trainer.step
            )
        )
        trainer.fit(
            batches,
            log_path=metrics_path,
            checkpoint_dir=args.output_dir / "checkpoints",
            save_every=args.save_every,
            first_save_step=0 if args.algorithm == "deas" else 100,
            save_interval_seconds=0 if args.algorithm == "deas" else 1800,
            max_run_seconds=28800,
            keep_latest_training_state=True,
            metrics_callback=(
                lambda m: (
                    wandb_run.log(m, step=m["step"])
                    if m["step"] == 1 or m["step"] % args.log_every == 0 or m["step"] == args.steps
                    else None
                )
            )
            if wandb_run
            else None,
        )
        final = args.output_dir / "checkpoints" / f"step-{trainer.step}.pt"
        if not final.exists():
            trainer.save_recovery_checkpoint(final.parent, True, archive_model=True)
        completed = True
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if wandb_run:
            wandb_run.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
