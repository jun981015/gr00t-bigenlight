"""Holdout evaluation for action-reinjected IQL, using ensemble MEAN and EMA V."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .action_conditioned_iql import load_action_iql_models
from .eval_cached_iql import discounted_returns, summarize, validate_provenance
from .feature_cache import CachedFeatureDataset, atomic_json


def summarize_mean(data):
    # Reuse legacy statistical formulas, but explicitly score ensemble MEAN.
    result = summarize({**data, "q_min": data["q_mean"], "q_gap": data["q_range"]})
    result["ensemble_range"] = result.pop("twin_q_gap")
    result["ensemble_std"] = float(data["q_std"].mean())
    result["action_grad_norm"] = float(data["action_grad_norm"].mean())
    result["perturb_abs_delta"] = float(data["perturb_abs_delta"].mean())
    return result


def evaluate(
    dataset, q, v, target_v, expectile, tasks, device="cpu", batch_size=256, perturb_sigma=0.05
):
    episodes, arrays = [], []
    generator = torch.Generator(device=device).manual_seed(1729)
    for episode, spec in enumerate(dataset.episodes):
        size = dataset.episode_sizes[episode]
        if not size:
            continue
        mc = discounted_returns(dataset.annotations[episode].rewards, dataset.gamma)[:size]
        chunks = []
        for start in range(0, size, batch_size):
            local = np.arange(start, min(start + batch_size, size))
            batch = dataset.batch(dataset.offsets[episode] + local).to(device)
            batch.validate()
            with torch.enable_grad():
                action = batch.actions.detach().requires_grad_(True)
                values = q(batch.observations, action * batch.action_mask)
                mean = values.mean(-1)
                grad = torch.autograd.grad(mean.sum(), action)[0]
            with torch.no_grad():
                value = v(batch.observations)
                target = batch.rewards + batch.discounts * target_v(batch.next_observations)
                diff = mean.detach() - value
                noisy = (
                    action
                    + perturb_sigma * torch.randn(action.shape, generator=generator, device=device)
                ) * batch.action_mask
                delta = (q(batch.observations, noisy).mean(-1) - mean.detach()).abs()
                tensors = {
                    "q_mean": mean.detach(),
                    "q_min": values.detach().min(-1).values,
                    "q_max": values.detach().max(-1).values,
                    "q_std": values.detach().std(-1, correction=0),
                    "q_range": values.detach().max(-1).values - values.detach().min(-1).values,
                    "v": value,
                    "td_target": target,
                    "td_abs": (values.detach() - target[:, None]).abs().mean(-1),
                    "td_squared": (values.detach() - target[:, None]).square().mean(-1),
                    "v_expectile_loss": torch.where(diff > 0, expectile, 1 - expectile)
                    * diff.square(),
                    "action_grad_norm": grad.flatten(1).norm(dim=-1),
                    "perturb_abs_delta": delta,
                    "terminal": batch.terminated,
                }
            if any(not torch.isfinite(t).all() for t in tensors.values()):
                raise FloatingPointError("Nonfinite evaluation output")
            row = {k: t.cpu().numpy() for k, t in tensors.items()}
            row.update(
                mc_return=mc[local],
                frame=local,
                episode=np.full(len(local), spec["episode_index"]),
                progress=local / max(1, spec["rows"] - 1),
            )
            chunks.append(row)
        data = {k: np.concatenate([r[k] for r in chunks]) for k in chunks[0]}
        arrays.append(data)
        episodes.append(
            {
                "episode": spec["episode_index"],
                "task": tasks[str(spec["episode_index"])],
                "transitions": size,
                **summarize_mean(data),
            }
        )
    if not arrays:
        raise ValueError("Empty holdout")
    data = {k: np.concatenate([r[k] for r in arrays]) for k in arrays[0]}
    keys = (
        "q_mc_mae",
        "q_mc_rmse",
        "q_mc_bias",
        "td_mae",
        "td_mse",
        "v_expectile_loss",
        "action_grad_norm",
        "perturb_abs_delta",
        "ensemble_std",
    )

    def macro(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    per_task = {}
    for task in sorted({str(e["task"]) for e in episodes}):
        selected = [e for e in episodes if str(e["task"]) == task]
        per_task[task] = {"episodes": len(selected), **macro(selected)}
    terminal = data["terminal"].astype(bool)
    return {
        "q_aggregation": "mean",
        "transition_weighted": summarize_mean(data),
        "episode_weighted": macro(episodes),
        "per_task": per_task,
        "episodes": episodes,
        "transitions": len(data["frame"]),
        "terminal": summarize_mean({k: v[terminal] for k, v in data.items()})
        if terminal.any()
        else None,
        "nonterminal": summarize_mean({k: v[~terminal] for k, v in data.items()})
        if (~terminal).any()
        else None,
    }, data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--wandb-project")
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.batch_size < 1:
        raise ValueError("Use a fresh output directory and positive batch size")
    torch.set_num_threads(2)
    archive = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    q, v, metadata = load_action_iql_models(args.checkpoint, args.device)
    settings = metadata["args"]
    ds = CachedFeatureDataset(args.eval_cache, reward=settings["reward"], gamma=settings["gamma"])
    split = validate_provenance(
        archive, args.train_cache, ds, expected_backend="action-conditioned-iql-cache-v1"
    )
    target_v = deepcopy(v)
    target_v.load_state_dict(archive["algorithm"]["target_value"], strict=True)
    report, data = evaluate(
        ds,
        q,
        v,
        target_v,
        settings["expectile_tau"],
        split["tasks"],
        args.device,
        args.batch_size,
        settings["perturb_sigma"],
    )
    report.update(
        step=archive["step"],
        expectile=settings["expectile_tau"],
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        eval_cache_manifest_sha256=hashlib.sha256(
            (args.eval_cache / "manifest.json").read_bytes()
        ).hexdigest(),
        caveat="Successful demonstration diagnostics, not policy success. Ensemble mean, EMA V bootstrap. Last H-1 starts excluded. Perturbation is diagnostic, not a labeled failure.",
    )
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "report.json", report)
    np.savez_compressed(args.output_dir / "predictions.npz", **data)
    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            mode="online",
            name=args.output_dir.name,
            job_type="action-iql-holdout",
            dir=str(args.output_dir),
            config={
                "checkpoint": str(args.checkpoint),
                "training_step": archive["step"],
                "q_aggregation": "mean",
            },
        )
        run.log(
            {
                f"validation/{group}/{k}": val
                for group in ("transition_weighted", "episode_weighted")
                for k, val in report[group].items()
                if val is not None
            }
        )
        atomic_json(args.output_dir / "wandb.json", {"url": run.url})
        run.finish()
    print(json.dumps({"step": archive["step"], **report["transition_weighted"]}), flush=True)


if __name__ == "__main__":
    main()
