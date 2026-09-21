"""SVF from disk-projected BC tokens: DiT LoRA + inner, optionally train env Q."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import signal

import numpy as np
import torch

from .actor_cache import ProjectedFeatureDataset, load_action_head
from .algorithms import SoftValueFlow, SVFConfig
from .dataset import EpisodeBatchSampler
from .feature_cache import atomic_json, cache_identity, identities_match
from .frozen_q import FrozenQSoftValueFlow, load_frozen_iql_q
from .networks import FeatureCritic
from .projected_actor import lora_actor_pair
from .trainer import OfflineTrainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-cache", type=Path, required=True)
    parser.add_argument("--actor-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-q", choices=["fixed-iql", "td"], default="fixed-iql")
    parser.add_argument("--iql-checkpoint", type=Path)
    parser.add_argument("--reward", choices=["step-cost", "terminal-success"], default="step-cost")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dit-lora-rank", type=int, default=16)
    parser.add_argument("--dit-lora-alpha", type=float, default=32.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--soft-lambda", type=float)
    parser.add_argument("--q-aggregation", choices=["min", "mean"], default="min")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-project")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if (args.env_q == "fixed-iql") != (args.iql_checkpoint is not None):
        raise ValueError("Supply --iql-checkpoint exactly for --env-q fixed-iql")
    if (
        min(
            args.steps,
            args.batch_size,
            args.dit_lora_rank,
            args.hidden_dim,
            args.hidden_layers,
            args.log_every,
        )
        < 1
        or min(args.seed, args.save_every) < 0
    ):
        raise ValueError("Invalid training settings")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError("Require fresh output or explicit resume")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = ProjectedFeatureDataset(
        args.pooled_cache, args.actor_cache, reward=args.reward, gamma=args.gamma
    )
    identity = dataset.base.manifest["identity"]
    current_identity = cache_identity(
        identity["model_path"],
        identity["dataset_path"],
        identity["horizon"],
        identity["embodiment"],
    )
    if not identities_match(identity, current_identity):
        raise ValueError("BC/data differs from cached conditioning")
    head = load_action_head(identity["model_path"], args.device)
    actor, reference, targets = lora_actor_pair(head, args.dit_lora_rank, args.dit_lora_alpha)
    feature_dim, shape = dataset.base.feature_dim, dataset.base.action_shape
    hidden = (args.hidden_dim,) * args.hidden_layers
    provenance = None
    if args.env_q == "fixed-iql":
        critic, provenance = load_frozen_iql_q(
            args.iql_checkpoint,
            args.pooled_cache,
            identity=current_identity,
            feature_dim=feature_dim,
            action_mask=torch.from_numpy(dataset.base.mask)[None],
            gamma=args.gamma,
            reward=args.reward,
            device=args.device,
        )
    else:
        critic = FeatureCritic(feature_dim, shape, hidden).to(args.device)
    config = SVFConfig(
        learning_rate=args.learning_rate,
        flow_steps=args.flow_steps,
        candidates=args.candidates,
        soft_lambda=args.soft_lambda,
        q_aggregation=args.q_aggregation,
        freeze_reference=True,
    )
    algorithm_class = FrozenQSoftValueFlow if args.env_q == "fixed-iql" else SoftValueFlow
    algorithm = algorithm_class(
        actor,
        critic,
        FeatureCritic(feature_dim, shape, hidden, time_embed_dim=16).to(args.device),
        config,
        reference=reference,
    )
    serialized = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata = {
        "backend": "projected-cache-dit-lora-v1",
        "args": {
            k: v
            for k, v in serialized.items()
            if k
            not in ("output_dir", "steps", "resume", "wandb_project", "log_every", "save_every")
        },
        "actor_manifest_sha256": hashlib.sha256(
            (args.actor_cache / "manifest.json").read_bytes()
        ).hexdigest(),
        "iql_source": provenance,
        "svf_config": asdict(config),
        "lora_targets": targets,
    }
    trainer = OfflineTrainer(algorithm, args.device, metadata=metadata)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    if args.steps <= trainer.step:
        raise ValueError("Target steps must exceed restored checkpoint")
    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists() and any(
        json.loads(line)["step"] > trainer.step
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ):
        raise ValueError("Metrics ahead of checkpoint; use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["metadata"] != metadata:
        raise ValueError("Run metadata mismatch")
    if not run_path.exists():
        atomic_json(
            run_path,
            {
                "args": serialized,
                "metadata": metadata,
                "actor_trainable_parameters": sum(
                    p.numel() for p in actor.parameters() if p.requires_grad
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
        wb_identity = json.loads(identity_path.read_text())
        if wb_identity["project"] != args.wandb_project:
            raise ValueError("W&B project changed")
        wandb_run = wandb.init(
            project=args.wandb_project,
            id=wb_identity["id"],
            resume="allow",
            mode="online",
            name=args.output_dir.name,
            dir=str(args.output_dir),
            config=serialized,
        )

    def stop(signum, frame):
        trainer.stop_requested = True

    previous = {signum: signal.signal(signum, stop) for signum in [signal.SIGTERM, signal.SIGINT]}
    success = False
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
            first_save_step=100,
            save_interval_seconds=1800,
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
        checkpoint_dir = args.output_dir / "checkpoints"
        if not (checkpoint_dir / f"step-{trainer.step}.pt").exists():
            trainer.save_recovery_checkpoint(checkpoint_dir, True, archive_model=True)
        success = True
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if wandb_run:
            wandb_run.finish(exit_code=0 if success else 1)


if __name__ == "__main__":
    main()
