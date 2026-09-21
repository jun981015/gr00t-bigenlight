"""Heldout demonstration diagnostics for scalar IQL, not a policy success evaluator."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .feature_cache import CachedFeatureDataset, atomic_json, identities_match
from .iql import FeatureValue
from .networks import FeatureCritic
from .prepare_critic_holdout import source_key


def discounted_returns(rewards, gamma):
    result = np.empty(len(rewards), dtype=np.float64)
    value = 0.0
    for index in reversed(range(len(rewards))):
        value = float(rewards[index]) + gamma * value
        result[index] = value
    return result


def validate_provenance(checkpoint, train_root, dataset, expected_backend="frozen-bc-cache-v1"):
    train_root = Path(train_root)
    train_manifest = json.loads((train_root / "manifest.json").read_text())
    metadata = checkpoint["metadata"]
    if metadata["backend"] != expected_backend:
        raise ValueError("Expected cached IQL checkpoint")
    if (
        metadata["cache_manifest_sha256"]
        != hashlib.sha256((train_root / "manifest.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Wrong training cache")
    left, right = train_manifest["identity"], dataset.manifest["identity"]
    if not identities_match(metadata["bc_identity"], left):
        raise ValueError("Checkpoint BC provenance differs from training cache")
    for field in ("model_path", "weight_file_stats", "horizon", "embodiment"):
        if left[field] != right[field]:
            raise ValueError(f"Holdout BC mismatch: {field}")

    def enc(identity):
        return {p.rsplit("/gr00t/", 1)[-1]: h for p, h in identity["encoder_sha256"].items()}

    if enc(left) != enc(right):
        raise ValueError("Holdout feature encoder differs")

    def model_configs(identity):
        return {
            p: h
            for p, h in identity["configs_sha256"].items()
            if p.startswith(identity["model_path"] + "/")
        }

    if model_configs(left) != model_configs(right):
        raise ValueError("BC processor/model config differs")
    for field in ("feature_dim", "action_shape", "action_indices"):
        if train_manifest[field] != dataset.manifest[field]:
            raise ValueError(f"Holdout dimensions/mask differ: {field}")
    split_path = Path(right["dataset_path"]) / "meta/holdout_split.json"
    if (
        right["configs_sha256"].get(str(split_path))
        != hashlib.sha256(split_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Holdout split changed since extraction")
    split = json.loads(split_path.read_text())
    if split["train_dataset"] != left["dataset_path"]:
        raise ValueError("Wrong BC training split")
    train_mapping_path = Path(left["dataset_path"]).parent / "source_mapping.json"
    full_mapping_path = Path(split["full_dataset"]).parent / "source_mapping.json"
    for path, key in [
        (train_mapping_path, "train_mapping_sha256"),
        (full_mapping_path, "full_mapping_sha256"),
    ]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != split[key]:
            raise ValueError("Source mapping changed")
    train_sources = {source_key(e) for e in json.loads(train_mapping_path.read_text())}
    heldout_sources = {source_key(e) for e in split["heldout_sources"]}
    full_sources = {source_key(e) for e in json.loads(full_mapping_path.read_text())}
    if train_sources & heldout_sources or train_sources | heldout_sources != full_sources:
        raise ValueError("Holdout is not the disjoint complement of training data")
    if [e["episode_index"] for e in dataset.episodes] != split["heldout_episode_ids"]:
        raise ValueError("Holdout cache episode selection differs")
    full_by_id = {e["episode_index"]: e for e in json.loads(full_mapping_path.read_text())}
    if [full_by_id[i] for i in split["heldout_episode_ids"]] != split["heldout_sources"]:
        raise ValueError("Source episode IDs differ from holdout mapping")
    return split


def summarize(data):
    q, v, mc = data["q_min"], data["v"], data["mc_return"]
    error = q - mc
    result = {
        "q_mc_mae": float(np.abs(error).mean()),
        "q_mc_rmse": float(np.sqrt(np.square(error).mean())),
        "q_mc_bias": float(error.mean()),
        "q_mc_over_fraction": float((error > 0).mean()),
        "td_mae": float(data["td_abs"].mean()),
        "td_mse": float(data["td_squared"].mean()),
        "v_expectile_loss": float(data["v_expectile_loss"].mean()),
        "twin_q_gap": float(data["q_gap"].mean()),
    }
    for name, values in [("q", q), ("v", v), ("mc_return", mc)]:
        result.update(
            {
                f"{name}_{stat}": float(fn(values))
                for stat, fn in [
                    ("min", np.min),
                    ("mean", np.mean),
                    ("max", np.max),
                    ("std", np.std),
                ]
            }
        )
    # IQL V is an expectile, so MC error is deliberately not a V selection criterion.
    result["q_mc_correlation"] = (
        float(np.corrcoef(q, mc)[0, 1]) if q.std() > 1e-8 and mc.std() > 1e-8 else None
    )
    return result


@torch.inference_mode()
def evaluate(dataset, critic, value, target_critic, expectile, tasks, device="cpu", batch_size=256):
    all_data, episodes = [], []
    for episode, spec in enumerate(dataset.episodes):
        size = dataset.episode_sizes[episode]
        if size < 1:
            continue
        mc = discounted_returns(dataset.annotations[episode].rewards, dataset.gamma)[:size]
        chunks = []
        for start in range(0, size, batch_size):
            local = np.arange(start, min(start + batch_size, size))
            batch = dataset.batch(dataset.offsets[episode] + local).to(device)
            batch.validate()
            action = batch.actions * batch.action_mask
            q = critic(batch.observations, action)
            v = value(batch.observations)
            target_q = target_critic(batch.observations, action).min(0).values
            td_target = batch.rewards + batch.discounts * value(batch.next_observations)
            diff = target_q - v
            columns = {
                "q_min": q.min(0).values,
                "q_mean": q.mean(0),
                "q_max": q.max(0).values,
                "v": v,
                "td_target": td_target,
                "q_gap": q.max(0).values - q.min(0).values,
                "td_abs": (q - td_target[None]).abs().mean(0),
                "td_squared": (q - td_target[None]).square().mean(0),
                "v_expectile_loss": torch.where(diff > 0, expectile, 1 - expectile) * diff.square(),
                "terminal": batch.terminated,
            }
            if any(not torch.isfinite(t).all() for t in columns.values()):
                raise FloatingPointError("Nonfinite evaluation output")
            row = {k: t.cpu().numpy() for k, t in columns.items()}
            row.update(
                mc_return=mc[local],
                frame=local,
                episode=np.full(len(local), spec["episode_index"]),
                progress=local / max(1, spec["rows"] - 1),
            )
            chunks.append(row)
        merged = {k: np.concatenate([r[k] for r in chunks]) for k in chunks[0]}
        all_data.append(merged)
        episodes.append(
            {
                "episode": spec["episode_index"],
                "task": tasks[str(spec["episode_index"])],
                "transitions": size,
                **summarize(merged),
            }
        )
    if not all_data:
        raise ValueError("No evaluable action chunks")
    data = {k: np.concatenate([r[k] for r in all_data]) for k in all_data[0]}
    macro_keys = ["q_mc_mae", "q_mc_rmse", "q_mc_bias", "td_mae", "td_mse", "v_expectile_loss"]

    def macro(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in macro_keys}

    per_task = {}
    for task in sorted({str(e["task"]) for e in episodes}):
        rows = [e for e in episodes if str(e["task"]) == task]
        per_task[task] = {"episodes": len(rows), **macro(rows)}
    progress = []
    for index in range(10):
        selected = (data["progress"] >= index / 10) & (data["progress"] < (index + 1) / 10)
        if selected.any():
            progress.append(
                {
                    "bin": index,
                    "transitions": int(selected.sum()),
                    **summarize({k: v[selected] for k, v in data.items()}),
                }
            )
    return {
        "transition_weighted": summarize(data),
        "episode_weighted": macro(episodes),
        "per_task": per_task,
        "progress_bins": progress,
        "episodes": episodes,
        "transitions": len(data["frame"]),
    }, data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Trusted model-only IQL archive"
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--wandb-project")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh evaluation output directory")
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    algorithm = checkpoint["algorithm"]
    if algorithm["algorithm"] != "iql-critic-only-v1":
        raise ValueError("Expected IQL checkpoint")
    config = checkpoint["metadata"]["args"]
    dataset = CachedFeatureDataset(args.eval_cache, reward=config["reward"], gamma=config["gamma"])
    split = validate_provenance(checkpoint, args.train_cache, dataset)
    hidden = (config["hidden_dim"],) * config["hidden_layers"]
    critic = FeatureCritic(dataset.feature_dim, dataset.action_shape, hidden).to(args.device).eval()
    target = FeatureCritic(dataset.feature_dim, dataset.action_shape, hidden).to(args.device).eval()
    value = FeatureValue(dataset.feature_dim, hidden).to(args.device).eval()
    for model, key in [(critic, "critic"), (target, "target_critic"), (value, "value")]:
        model.load_state_dict(algorithm[key], strict=True)
        model.requires_grad_(False)
    report, data = evaluate(
        dataset,
        critic,
        value,
        target,
        algorithm["config"]["expectile"],
        split["tasks"],
        args.device,
        args.batch_size,
    )
    report.update(
        checkpoint=str(args.checkpoint.resolve()),
        step=checkpoint["step"],
        expectile=algorithm["config"]["expectile"],
        reward=config["reward"],
        gamma=config["gamma"],
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        eval_cache_manifest_sha256=hashlib.sha256(
            (args.eval_cache / "manifest.json").read_bytes()
        ).hexdigest(),
        caveat="MC return follows recorded successful demonstrations, not the improved IQL policy. No failure/OOD-action or robot success evaluation. Full chunks only; final H-1 starts excluded.",
    )
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "report.json", report)
    with (args.output_dir / "predictions.csv").open("x") as f:
        writer = csv.writer(f)
        writer.writerow(data)
        writer.writerows(zip(*data.values(), strict=True))
    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            mode="online",
            name=args.output_dir.name,
            job_type="critic-holdout-eval",
            dir=str(args.output_dir),
            config={
                "checkpoint": str(args.checkpoint),
                "step": checkpoint["step"],
                "expectile": report["expectile"],
                "heldout_episodes": len(report["episodes"]),
            },
        )
        metrics = {
            f"validation/{group}/{key}": value
            for group in ["transition_weighted", "episode_weighted"]
            for key, value in report[group].items()
            if value is not None
        }
        for task, values in report["per_task"].items():
            metrics.update({f"validation/task/{task}/{key}": v for key, v in values.items()})
        metrics["validation/progress_bins"] = wandb.Table(
            dataframe=__import__("pandas").DataFrame(report["progress_bins"])
        )
        run.log(metrics)
        run.summary.update(
            {
                "report_path": str(args.output_dir / "report.json"),
                "training_step": checkpoint["step"],
            }
        )
        atomic_json(args.output_dir / "wandb.json", {"url": run.url})
        run.finish()
    print(
        json.dumps(
            {"step": report["step"], "expectile": report["expectile"], **report["episode_weighted"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
