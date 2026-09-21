"""Export inference-only SVF LoRA/critics without duplicating BC weights."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


def split_actor(actor):
    adapters, frozen = {}, {}
    for key, value in actor.items():
        if key.endswith((".lora_A", ".lora_B")):
            adapters[key] = value.detach().cpu().contiguous().clone()
        else:
            base_key = (
                key.removeprefix("head.")
                .replace(".base.weight", ".weight")
                .replace(".base.bias", ".bias")
            )
            frozen["action_head." + base_key] = value
    if not adapters:
        raise ValueError("No LoRA tensors; full actor training cannot use adapter-only export")
    return adapters, frozen


def load_actor_adapter(actor, path):
    """Actor must already wrap the exact BC head with the manifest's LoRA config."""
    weights = load_file(str(path), device="cpu")
    expected = {k for k in actor.state_dict() if k.endswith((".lora_A", ".lora_B"))}
    if not expected or set(weights) != expected:
        raise ValueError("LoRA key mismatch: check architecture, targets and base model")
    if any(not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError("Nonfinite LoRA weights")
    actor.load_state_dict(weights, strict=False)
    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--base-repo", required=True)
    parser.add_argument("--base-revision", required=True, help="Exact HF commit, never main")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--upload-to", help="Optional existing HF repository; upload under adapter/"
    )
    args = parser.parse_args()
    if len(args.base_revision) != 40 or any(
        c not in "0123456789abcdef" for c in args.base_revision
    ):
        raise ValueError("Require a pinned HF commit hash")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    algorithm, metadata = checkpoint["algorithm"], checkpoint["metadata"]
    settings = metadata["args"]
    adapters, frozen = split_actor(algorithm["actor"])
    seen = set()
    # Exact comparison refuses export if ANY frozen actor weight changed.
    for shard in sorted(args.base_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key in frozen:
                    # Training loads the BC head as BF16; compare after the
                    # same dtype conversion, not against its FP32 storage.
                    if key in seen or not torch.equal(
                        source.get_tensor(key).to(frozen[key].dtype), frozen[key]
                    ):
                        raise ValueError(f"Actor differs from base BC outside LoRA: {key}")
                    seen.add(key)
    if seen != set(frozen):
        raise ValueError("Base checkpoint missing frozen actor tensors")
    for key, tensor in {**adapters, **algorithm["critic"], **algorithm["inner_critic"]}.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite export tensor: {key}")
    config = {
        "format": "gr00t-svf-adapter-v1",
        "step": checkpoint["step"],
        "base_model": args.base_repo,
        "base_revision": args.base_revision,
        "lora_rank": settings["dit_lora_rank"],
        "base_head_dtype": "bfloat16",
        "lora_alpha": settings["dit_lora_alpha"],
        "lora_targets": metadata["lora_targets"],
        "svf_config": algorithm["config"],
        "env_q_mode": settings.get("env_q", "see_source_metadata"),
        "critic_hidden_dim": settings["hidden_dim"],
        "critic_hidden_layers": settings["hidden_layers"],
        "inner_time_embed_dim": 16,
        "critic_source": metadata.get("iql_source", metadata.get("fixed_iql_q")),
        "action_horizon": 16,
        "includes": ["actor LoRA A/B", "online env Q", "online inner critic"],
        "excludes": ["BC/VLM weights", "reference", "EMA targets", "optimizer", "RNG"],
        "frozen_actor_matches_local_bc": True,
        "note": "Inference export only. Not a training resume checkpoint or native BC model directory.",
    }
    args.output_dir.mkdir(parents=True)
    save_file(adapters, str(args.output_dir / "actor_lora.safetensors"))
    for name, key in [("env_q", "critic"), ("inner_critic", "inner_critic")]:
        save_file(
            {k: v.detach().cpu().contiguous().clone() for k, v in algorithm[key].items()},
            str(args.output_dir / f"{name}.safetensors"),
        )
    (args.output_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    (args.output_dir / "README.md").write_text("""# Lightweight SVF inference bundle

Download the exact base_model/base_revision in adapter_config.json once.
Load the original BC processor/statistics from that revision, and construct its
BC action head. Apply inject_dit_lora(head, lora_rank, lora_alpha), wrap it in
ProjectedFlowActor, and call load_actor_adapter(actor, actor_lora.safetensors).
Helpers are in gr00t.rl.projected_actor and gr00t.rl.export_adapter.
Do NOT merge these adapters onto already fine-tuned SVF weights.

actor_lora.safetensors is sufficient for the learned actor when combined with
the exact BC. env_q and inner_critic are optional for scoring/guided sampling;
their model architectures must match adapter_config.json and critic_source.
Live observations still require BC VLM and frozen observation projection.
This is not a standalone robot controller or a training recovery checkpoint.
""")
    print(json.dumps({p.name: p.stat().st_size for p in args.output_dir.iterdir()}), flush=True)
    if args.upload_to:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=args.upload_to,
            folder_path=args.output_dir,
            path_in_repo="adapter",
            commit_message="Add BC-deduplicated SVF adapter and optional critics",
        )


if __name__ == "__main__":
    main()
