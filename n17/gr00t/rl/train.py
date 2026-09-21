"""Single-device offline RL entry point. Run python -m gr00t.rl.train --help."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import signal

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapters import (
    FrozenBCGR00TEncoder,
    FrozenGR00TEncoder,
    Gr00tFlowActor,
    Gr00tTransitionCollator,
    StateActionTransitionCollator,
)
from .algorithms import FlowBC, SoftValueFlow, SVFConfig
from .dataset import (
    AllSuccessStepCostAnnotations,
    AllSuccessTerminalAnnotations,
    ColumnRLAnnotations,
    DEASRoboCasaAnnotations,
    EpisodeBatchSampler,
    LeRobotOfflineRLDataset,
    OfflineRLConcatDataset,
)
from .feature_cache import cache_identity
from .frozen_q import (
    FrozenQConditioningEncoder,
    FrozenQInnerOnly,
    FrozenQSoftValueFlow,
    load_frozen_iql_q,
)
from .iql import FeatureValue, IQLConfig, IQLCriticLearner
from .networks import FeatureCritic, FeatureFlowActor
from .trainer import OfflineTrainer


def initialize_loader_worker(worker_id):
    """Worker CPU budgets; never inherit a CUDA context (spawn is required)."""
    torch.set_num_threads(1)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_batches(dataset, sampler, collator, *, workers=0, prefetch=2, seed=0):
    options = {}
    if workers:
        options.update(
            multiprocessing_context="spawn",
            prefetch_factor=prefetch,
            persistent_workers=True,
            worker_init_fn=initialize_loader_worker,
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
        **options,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Use your RAID output directory"
    )
    parser.add_argument("--backend", choices=("state", "gr00t"), default="state")
    parser.add_argument("--algorithm", choices=("bc", "svf", "iql"), default="svf")
    parser.add_argument("--iql-expectile", type=float, default=0.7)
    parser.add_argument("--iql-target-tau", type=float, default=0.005)
    parser.add_argument(
        "--model-path", type=Path, help="GR00T checkpoint already configured for this robot"
    )
    parser.add_argument(
        "--modality-config-path",
        type=Path,
        help="Trusted Python file registering NEW_EMBODIMENT (state backend)",
    )
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument(
        "--annotation-format",
        choices=("columns", "deas-robocasa", "all-success-terminal", "all-success-step-cost"),
        default="columns",
    )
    parser.add_argument("--reward-column")
    parser.add_argument("--terminated-column")
    parser.add_argument("--truncated-column")
    parser.add_argument("--last-row-is-observation", action="store_true")
    parser.add_argument(
        "--bootstrap-on-truncation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--relative-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="State backend; GR00T always uses checkpoint normalization",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Number of actions actually executed per RL transition",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--bootstrap-gamma",
        type=float,
        help="Inter-chunk discount; defaults to --gamma. Separate DEAS-style discounts are opt-in",
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Total target updates, including resumed updates"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--loader-prefetch", type=int, default=2)
    parser.add_argument("--episode-cache-count", type=int, default=1)
    parser.add_argument("--episode-cache-gib", type=float, default=0)
    parser.add_argument("--decoder-threads", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--lambda-multiplier", type=float, default=1.0)
    parser.add_argument("--q-aggregation", choices=("mean", "min"), default="mean")
    parser.add_argument("--freeze-reference", action="store_true")
    parser.add_argument(
        "--dit-lora-rank",
        type=int,
        default=0,
        help="Positive rank: train only DiT LoRA, freeze all projections/reference",
    )
    parser.add_argument("--dit-lora-alpha", type=float, default=32.0)
    parser.add_argument(
        "--critic-feature-cache",
        type=Path,
        help="Reuse pooled Q/inner features with live DiT tokens; requires DiT LoRA",
    )
    parser.add_argument(
        "--fixed-iql-checkpoint",
        "--fixed-q-checkpoint",
        type=Path,
        help="Trusted cached IQL or DEAS full checkpoint; freezes env Q and BC reference",
    )
    parser.add_argument(
        "--fixed-iql-cache", "--fixed-q-cache", type=Path, help="Completed source critic cache"
    )
    parser.add_argument("--first-save-step", type=int, default=0)
    parser.add_argument("--allow-batch-size-change", action="store_true")
    parser.add_argument(
        "--inner-only", action="store_true", help="Freeze actor/reference/env Q; train inner only"
    )
    parser.add_argument(
        "--initialize-svf",
        type=Path,
        help="Trusted fixed-Q SVF checkpoint to fork joint or inner-only runs",
    )
    parser.add_argument("--save-interval-seconds", type=float, default=0)
    parser.add_argument("--max-run-seconds", type=float, default=0)
    parser.add_argument("--keep-latest-training-state", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-log-every", type=int, default=1)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="0: final checkpoint only; full GR00T optimizer checkpoints are large",
    )
    parser.add_argument(
        "--resume", type=Path, help="Trusted local training checkpoint, not an HF model checkpoint"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fixed_q = args.fixed_iql_checkpoint is not None
    if args.inner_only and not (fixed_q and args.algorithm == "svf"):
        raise ValueError("Inner-only requires fixed-Q SVF")
    if args.initialize_svf and not (fixed_q and args.algorithm == "svf"):
        raise ValueError("--initialize-svf requires fixed-Q SVF")
    if args.dit_lora_rank < 0:
        raise ValueError("LoRA rank cannot be negative")
    if args.dit_lora_rank:
        if args.backend != "gr00t" or args.algorithm != "svf":
            raise ValueError("DiT LoRA currently requires GR00T SVF")
        args.freeze_reference = True
    if args.critic_feature_cache and not args.dit_lora_rank:
        raise ValueError("--critic-feature-cache currently requires --dit-lora-rank")
    if fixed_q != (args.fixed_iql_cache is not None):
        raise ValueError("Provide both --fixed-iql-checkpoint and --fixed-iql-cache")
    if fixed_q:
        if args.algorithm != "svf" or args.backend != "gr00t" or len(args.dataset_path) != 1:
            raise ValueError("Frozen IQL Q requires GR00T SVF and its single original dataset")
        if args.annotation_format not in ("all-success-terminal", "all-success-step-cost"):
            raise ValueError("Frozen cached IQL Q requires the original all-success reward preset")
        if args.bootstrap_gamma is not None and args.bootstrap_gamma != args.gamma:
            raise ValueError("Frozen cached IQL Q requires identical chunk/bootstrap gamma")
        args.freeze_reference = True
    if args.annotation_format == "columns" and args.reward_column is None:
        raise ValueError("Column annotations require --reward-column")
    if args.annotation_format != "columns" and any(
        (
            args.reward_column,
            args.terminated_column,
            args.truncated_column,
            args.last_row_is_observation,
        )
    ):
        raise ValueError("Do not combine an annotation preset with manual column mappings")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(
            "Offline RL currently supports one process; do not launch independent learners with torchrun"
        )
    for key in (
        "steps",
        "batch_size",
        "horizon",
        "hidden_dim",
        "hidden_layers",
        "cpu_threads",
        "wandb_log_every",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.seed < 0 or args.save_every < 0:
        raise ValueError("seed/save_every must be nonnegative")
    if (
        args.loader_workers < 0
        or args.loader_prefetch < 1
        or args.episode_cache_count < 1
        or args.episode_cache_gib < 0
        or args.decoder_threads < 0
    ):
        raise ValueError("Invalid loader/cache configuration")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise FileExistsError("Use a new output directory or explicitly --resume an existing run")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; restore the GPU allocation or use --backend state --device cpu"
        )
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.state_action.state_action_processor import StateActionProcessor

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    encoder = None
    reference = None
    fixed_q_identity = None
    q_provenance = None
    if args.backend == "gr00t":
        if args.model_path is None:
            raise ValueError("--backend gr00t requires --model-path")
        if args.modality_config_path is not None:
            raise ValueError(
                "GR00T backend uses checkpoint modality/normalization; do not override with a new config"
            )
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(tag, str(args.model_path), device=args.device)
        if args.horizon > len(policy.modality_configs["action"].delta_indices):
            raise ValueError("RL horizon must not exceed the robot's trained action horizon")
        modalities = policy.modality_configs
        if args.algorithm == "iql":
            encoder = FrozenBCGR00TEncoder(policy.model)
            assert not any(p.requires_grad for p in policy.model.parameters())
        elif args.dit_lora_rank:
            from .feature_cache import CachedFeatureDataset, identities_match
            from .projected_actor import FrozenProjectedEncoder, lora_actor_pair

            fixed_q_identity = (
                cache_identity(
                    args.model_path, args.dataset_path[0], args.horizon, args.embodiment_tag
                )
                if fixed_q
                else None
            )
            actor, reference, _ = lora_actor_pair(
                policy.model.action_head, args.dit_lora_rank, args.dit_lora_alpha
            )
            pooled_path = args.critic_feature_cache or (args.fixed_iql_cache if fixed_q else None)
            pooled = None
            if pooled_path:
                if len(args.dataset_path) != 1 or args.annotation_format not in (
                    "all-success-terminal",
                    "all-success-step-cost",
                ):
                    raise ValueError(
                        "Critic cache requires its original single all-success dataset"
                    )
                if fixed_q and pooled_path.resolve() != args.fixed_iql_cache.resolve():
                    raise ValueError("IQL Q and critic features must use the same cache")
                pooled = CachedFeatureDataset(
                    pooled_path,
                    gamma=args.gamma,
                    reward="step-cost"
                    if args.annotation_format == "all-success-step-cost"
                    else "terminal-success",
                )
                expected = cache_identity(
                    args.model_path, args.dataset_path[0], args.horizon, args.embodiment_tag
                )
                if not identities_match(pooled.manifest["identity"], expected):
                    raise ValueError("Critic cache BC/data identity mismatch")
            encoder = FrozenProjectedEncoder(policy.model, critic_cache=pooled)
        else:
            if fixed_q:
                from copy import deepcopy

                fixed_q_identity = cache_identity(
                    args.model_path, args.dataset_path[0], args.horizon, args.embodiment_tag
                )
                encoder = FrozenQConditioningEncoder(policy.model)
                reference = Gr00tFlowActor(deepcopy(policy.model.action_head))
                reference.requires_grad_(False).eval()
            # Keep Adam trainable weights in FP32; VLM remains frozen BF16.
            policy.model.action_head.float()
            policy.model.action_head.set_trainable_parameters(True, True, True)
            actor = Gr00tFlowActor(policy.model.action_head)
            if not fixed_q:
                encoder = FrozenGR00TEncoder(policy.model)
        collator = Gr00tTransitionCollator(policy.processor, args.gamma)
    else:
        if args.modality_config_path is not None:
            runpy.run_path(str(args.modality_config_path.resolve()))
        if tag.value not in MODALITY_CONFIGS:
            raise ValueError(
                "Unknown state-backend modality config; provide --modality-config-path"
            )
        # State-only path does not decode unused images.
        modalities = {
            key: value
            for key, value in MODALITY_CONFIGS[tag.value].items()
            if key not in ("video", "mask", "rl_info")
        }

    loaders = [
        LeRobotEpisodeLoader(
            path, modalities, decoder_kwargs={"num_ffmpeg_threads": args.decoder_threads}
        )
        for path in args.dataset_path
    ]
    labels = (
        AllSuccessStepCostAnnotations()
        if args.annotation_format == "all-success-step-cost"
        else (
            AllSuccessTerminalAnnotations()
            if args.annotation_format == "all-success-terminal"
            else (
                DEASRoboCasaAnnotations()
                if args.annotation_format == "deas-robocasa"
                else ColumnRLAnnotations(
                    args.reward_column,
                    args.terminated_column,
                    args.truncated_column,
                    args.last_row_is_observation,
                )
            )
        )
    )
    single_datasets = [
        LeRobotOfflineRLDataset(
            loader,
            tag,
            labels,
            horizon=args.horizon,
            gamma=args.gamma,
            bootstrap_gamma=args.bootstrap_gamma,
            bootstrap_on_truncation=args.bootstrap_on_truncation,
            cache_episodes=args.episode_cache_count,
            cache_bytes=int(args.episode_cache_gib * 1024**3),
        )
        for loader in loaders
    ]
    dataset = (
        single_datasets[0] if len(single_datasets) == 1 else OfflineRLConcatDataset(single_datasets)
    )
    if args.backend == "state":
        from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics

        all_stats = [loader.get_dataset_statistics() for loader in loaders]
        statistics = {
            modality: merge_statistics(
                [stats[modality] for stats in all_stats],
                [len(ds) for ds in single_datasets],
                is_relative_stats=modality == "relative_action",
            )
            for modality in all_stats[0]
        }
        normalization = StateActionProcessor(
            {tag.value: modalities},
            {tag.value: statistics},
            use_relative_action=args.relative_actions,
        )
        normalization.eval()
        collator = StateActionTransitionCollator(normalization, {tag.value: modalities}, args.gamma)

    example = collator([dataset[0]])
    example = encoder(example) if encoder is not None else example.to(args.device)
    example.validate()
    shape, feature_dim = (
        tuple(example.actions.shape[1:]),
        example.observations["features"].shape[-1],
    )
    hidden_dims = (args.hidden_dim,) * args.hidden_layers
    if args.backend == "state" and args.algorithm != "iql":
        actor = FeatureFlowActor(feature_dim, shape, hidden_dims).to(args.device)
    config = (
        IQLConfig(args.learning_rate, args.iql_expectile, args.iql_target_tau)
        if args.algorithm == "iql"
        else SVFConfig(
            learning_rate=args.learning_rate,
            flow_steps=args.flow_steps,
            candidates=args.candidates,
            kappa=args.kappa,
            lambda_multiplier=args.lambda_multiplier,
            q_aggregation=args.q_aggregation,
            freeze_reference=args.freeze_reference,
        )
    )
    if args.algorithm == "iql":
        algorithm = IQLCriticLearner(
            FeatureCritic(feature_dim, shape, hidden_dims).to(args.device),
            FeatureValue(feature_dim, hidden_dims).to(args.device),
            config,
        )
    elif args.algorithm == "bc":
        algorithm = FlowBC(actor, config)
    elif fixed_q:
        critic, q_provenance = load_frozen_iql_q(
            args.fixed_iql_checkpoint,
            args.fixed_iql_cache,
            identity=fixed_q_identity,
            feature_dim=feature_dim,
            action_mask=example.action_mask,
            gamma=args.gamma,
            reward=(
                "step-cost"
                if args.annotation_format == "all-success-step-cost"
                else "terminal-success"
            ),
            device=args.device,
        )
        learner_class = FrozenQInnerOnly if args.inner_only else FrozenQSoftValueFlow
        algorithm = learner_class(
            actor,
            critic,
            FeatureCritic(feature_dim, shape, hidden_dims, time_embed_dim=16).to(args.device),
            config,
            reference=reference,
        )
    else:
        algorithm = SoftValueFlow(
            actor,
            FeatureCritic(feature_dim, shape, hidden_dims).to(args.device),
            FeatureCritic(feature_dim, shape, hidden_dims, time_embed_dim=16).to(args.device),
            config,
            reference=reference,
        )
    semantic_args = {
        key: value
        for key, value in vars(args).items()
        if (args.algorithm == "iql" or not key.startswith("iql_"))
        and key
        not in (
            "steps",
            "output_dir",
            "resume",
            "save_every",
            "cpu_threads",
            "first_save_step",
            "save_interval_seconds",
            "max_run_seconds",
            "keep_latest_training_state",
            "wandb_project",
            "wandb_log_every",
            "loader_workers",
            "loader_prefetch",
            "episode_cache_count",
            "episode_cache_gib",
            "decoder_threads",
        )
        and (fixed_q or key not in ("fixed_iql_checkpoint", "fixed_iql_cache"))
        and (args.dit_lora_rank or key not in ("dit_lora_rank", "dit_lora_alpha"))
        and (args.critic_feature_cache is not None or key != "critic_feature_cache")
        and (args.inner_only or key != "inner_only")
        and key != "initialize_svf"
        and key != "allow_batch_size_change"
    }
    signal_hash = hashlib.sha256()
    for annotation in dataset.annotations:
        for values in (annotation.rewards, annotation.terminated, annotation.truncated):
            signal_hash.update(values.tobytes())
    normalization_statistics = (
        policy.processor.state_action_processor.statistics
        if args.backend == "gr00t"
        else normalization.statistics
    )
    metadata = json.loads(
        json.dumps(
            {
                "args": semantic_args,
                "normalization": normalization_statistics,
                "modalities": {key: asdict(value) for key, value in modalities.items()},
                "episodes": [loader.episodes_metadata for loader in loaders],
                "rl_signal_sha256": signal_hash.hexdigest(),
            },
            default=str,
        )
    )
    if fixed_q:
        metadata["fixed_iql_q"] = q_provenance
    if args.initialize_svf:
        source = torch.load(args.initialize_svf, map_location="cpu", weights_only=False)
        expected = json.loads(json.dumps(metadata))
        expected["args"].pop("inner_only", None)
        source_metadata = json.loads(json.dumps(source.get("metadata")))
        if isinstance(source_metadata, dict):
            source_metadata.get("args", {}).pop("inner_only", None)
            source_metadata.pop("initial_svf", None)
        if args.allow_batch_size_change:
            expected["args"].pop("batch_size", None)
            source_metadata["args"].pop("batch_size", None)
        if (
            source.get("format_version") != 1
            or source_metadata != expected
            or source.get("step", 0) < 1
            or source["step"] != source["algorithm"]["updates"]
        ):
            raise ValueError("Initialization SVF provenance differs from this run")
        if not args.resume:
            algorithm.initialize_from_svf(source["algorithm"])
        with args.initialize_svf.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        metadata["initial_svf"] = {
            "checkpoint": str(args.initialize_svf.resolve()),
            "sha256": digest,
            "step": source["step"],
            "optimizer": "fresh inner-only Adam" if args.inner_only else "fresh actor+inner Adam",
            "new_step_counter": True,
        }
        del source
    trainer = OfflineTrainer(algorithm, args.device, encoder, metadata=metadata)
    if args.resume is not None:
        if args.allow_batch_size_change:
            restored = torch.load(args.resume, map_location="cpu", weights_only=False)
            previous_metadata = restored["metadata"]
            del restored
            comparable = json.loads(json.dumps(previous_metadata))
            transitions = comparable.pop("batch_size_transitions", [])
            old_batch_size = comparable["args"]["batch_size"]
            comparable["args"]["batch_size"] = args.batch_size
            if comparable != metadata:
                raise ValueError("Only batch-size changes are allowed during this resume")
            trainer.metadata = previous_metadata
            trainer.load_checkpoint(args.resume)
            if old_batch_size != args.batch_size:
                transitions = transitions + [
                    {"step": trainer.step, "old": old_batch_size, "new": args.batch_size}
                ]
            metadata["batch_size_transitions"] = transitions
            trainer.metadata = json.loads(json.dumps(metadata))
        else:
            trainer.load_checkpoint(args.resume)
    if args.steps <= trainer.step:
        raise ValueError("--steps must exceed the restored update count")
    manifest = args.output_dir / "run.json"
    if manifest.exists():
        with manifest.open() as previous:
            if json.load(previous).get("metadata") != metadata:
                raise ValueError("Output run metadata differs; resume into a new output directory")
    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists():
        # Appending to a later run after rewinding a checkpoint creates ambiguous
        # duplicate steps. Fork into a new output directory instead.
        with metrics_path.open() as previous:
            for line in previous:
                if line.strip() and json.loads(line)["step"] > trainer.step:
                    raise ValueError(
                        "Output metrics are ahead of checkpoint; use a new output directory"
                    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        with manifest.open("x") as output:
            json.dump(
                {
                    "args": vars(args),
                    "algorithm_config" if args.algorithm == "iql" else "svf_config": asdict(config),
                    "action_shape": shape,
                    "feature_dim": feature_dim,
                    "metadata": metadata,
                    "fmrl_reference": "7304b615e6f1bbb07d462ea24bf008e8ac019a17",
                },
                output,
                default=str,
                indent=2,
            )
        if args.backend == "gr00t":
            policy.processor.save_pretrained(args.output_dir / "processor")
        if args.annotation_format in ("all-success-terminal", "all-success-step-cost"):
            with (args.output_dir / "reward_assumption.json").open("x") as output:
                json.dump(
                    {
                        "assumption": (
                            "User confirms every episode succeeds; final recorded action receives 0 and terminated=true; every other environment action receives -1"
                            if args.annotation_format == "all-success-step-cost"
                            else "User confirms every episode succeeds; final recorded action receives +1 and terminated=true; all other rewards zero"
                        ),
                        "source_data_modified": False,
                        "episodes": [
                            {
                                "dataset": str(loader.dataset_path),
                                "episode_index": meta["episode_index"],
                                "terminal_row": len(annotation.rewards) - 1,
                            }
                            for loader, ds in zip(loaders, single_datasets)
                            for meta, annotation in zip(loader.episodes_metadata, ds.annotations)
                        ],
                    },
                    output,
                    indent=2,
                )
    sampler = EpisodeBatchSampler(
        dataset, args.batch_size, args.steps - trainer.step, args.seed, trainer.step
    )
    batches = build_batches(
        dataset,
        sampler,
        collator,
        workers=args.loader_workers,
        prefetch=args.loader_prefetch,
        seed=args.seed,
    )
    wandb_run = None
    if args.wandb_project:
        import wandb

        from gr00t.experiment.checkpoint_policy import atomic_json

        identity_path = args.output_dir / "wandb_resume.json"
        if not identity_path.exists():
            atomic_json(
                identity_path, {"id": wandb.util.generate_id(), "project": args.wandb_project}
            )
        identity = json.loads(identity_path.read_text())
        if identity["project"] != args.wandb_project:
            raise ValueError("W&B project differs from the saved run identity")
        wandb_run = wandb.init(
            project=identity["project"],
            id=identity["id"],
            resume="allow",
            name=args.output_dir.name,
            dir=str(args.output_dir),
            config=vars(args),
        )
    completed = False
    old_handlers = {}

    def request_stop(signum, frame):
        trainer.stop_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, request_stop)
    try:
        trainer.fit(
            batches,
            log_path=args.output_dir / "metrics.jsonl",
            checkpoint_dir=args.output_dir / "checkpoints",
            save_every=args.save_every,
            first_save_step=args.first_save_step,
            save_interval_seconds=args.save_interval_seconds,
            max_run_seconds=args.max_run_seconds,
            keep_latest_training_state=args.keep_latest_training_state,
            metrics_callback=(
                lambda metrics: (
                    wandb_run.log(metrics, step=metrics["step"])
                    if metrics["step"] == 1
                    or metrics["step"] % args.wandb_log_every == 0
                    or metrics["step"] == args.steps
                    else None
                )
            )
            if wandb_run
            else None,
        )
        final = args.output_dir / "checkpoints" / f"step-{trainer.step}.pt"
        if not final.exists():
            trainer.save_recovery_checkpoint(
                final.parent, args.keep_latest_training_state, archive_model=True
            )
        completed = True
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if wandb_run is not None:
            wandb_run.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
