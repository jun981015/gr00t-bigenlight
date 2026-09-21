from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.rl import train_cached_svf
from gr00t.rl.actor_cache import (
    FIELDS,
    ProjectedFeatureDataset,
    load_action_head,
    manifest_for,
    mark_complete,
    write_projected_episode,
)
from gr00t.rl.adapters import FrozenBCGR00TEncoder, Gr00tFlowActor
from gr00t.rl.algorithms import SoftValueFlow, SVFConfig
from gr00t.rl.feature_cache import CachedFeatureDataset, atomic_json
from gr00t.rl.frozen_q import FrozenQSoftValueFlow
from gr00t.rl.networks import FeatureCritic
from gr00t.rl.projected_actor import (
    FrozenProjectedEncoder,
    lora_actor_pair,
    project_observation,
    reuse_critic_cache,
)
from gr00t.rl.train_cached import main as train_iql
from gr00t.rl.types import OfflineRLBatch
import pytest
from safetensors.torch import save_file
import torch

from .test_feature_cache import make_cache
from .test_gr00t_adapter import TinyModel, observations, small_head


@pytest.fixture(autouse=True)
def cpu_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(21)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("alternate", [False, True])
def test_projected_lora_matches_bc_and_never_reprojects(alternate, monkeypatch):
    head = small_head(alternate).requires_grad_(False)
    obs = observations()
    raw = obs["backbone_features"].clone()
    projected = project_observation(head, obs)
    actions, time = torch.randn(2, 3, 2), torch.tensor([0.3, 0.7])
    expected = Gr00tFlowActor(head)(obs, actions, time)
    actor, reference, targets = lora_actor_pair(head, rank=4, alpha=8)
    assert targets and all(
        "lora_" in name for name, p in actor.named_parameters() if p.requires_grad
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Projection must be bypassed after caching")

    monkeypatch.setattr(head, "_encode_features", forbidden)
    monkeypatch.setattr(reference.head, "_encode_features", forbidden)
    torch.testing.assert_close(actor(projected, actions, time), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reference(projected, actions, time), expected, rtol=1e-5, atol=1e-6)
    actor(projected, actions, time).square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in actor.named_parameters()
        if "lora_B" in name
    )
    assert all(p.grad is None for name, p in actor.named_parameters() if "lora_" not in name)
    torch.testing.assert_close(raw, obs["backbone_features"], rtol=0, atol=0)


def test_live_projected_encoder_matches_iql_features():
    model = TinyModel().requires_grad_(False)
    obs = observations()
    expected = FrozenBCGR00TEncoder(deepcopy(model)).encode_observation(obs)
    actual = FrozenProjectedEncoder(model).encode_observation(obs)
    assert "backbone_features" not in actual
    torch.testing.assert_close(actual["features"], expected["features"], rtol=0, atol=0)
    assert model.backbone.calls == 1


@pytest.mark.parametrize("fixed", [True, False])
def test_svf_updates_only_lora_inner_and_optional_env_q(fixed):
    head = small_head(True).requires_grad_(False)
    obs = project_observation(head, observations())
    obs["features"] = torch.randn(2, 5)
    actor, reference, _ = lora_actor_pair(head, 4, 8)
    cls = FrozenQSoftValueFlow if fixed else SoftValueFlow
    algorithm = cls(
        actor,
        FeatureCritic(5, (3, 2), (8,)),
        FeatureCritic(5, (3, 2), (8,), time_embed_dim=16),
        SVFConfig(freeze_reference=True, flow_steps=1, candidates=1, soft_lambda=1, t_min=1e-6),
        reference,
    )
    before = deepcopy(actor.state_dict())
    q_before = deepcopy(algorithm.critic.state_dict())
    data = OfflineRLBatch(
        obs,
        obs,
        torch.randn(2, 3, 2),
        torch.ones(2, 3, 2),
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.full((2,), 3),
    )
    metrics = algorithm.update(data)
    changed = [k for k, v in actor.state_dict().items() if not torch.equal(v, before[k])]
    assert changed and all("lora_" in k for k in changed)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert (
        all(torch.equal(v, q_before[k]) for k, v in algorithm.critic.state_dict().items()) == fixed
    )


def make_projected_cache(pooled, root):
    base = CachedFeatureDataset(pooled, reward="step-cost")
    root.mkdir()
    manifest = manifest_for(base)
    atomic_json(root / "manifest.json", manifest)
    for spec in manifest["episodes"]:
        rows = spec["rows"]
        payload = {
            "projected_vl_features": torch.randn(rows, 8, 64).bfloat16(),
            "projected_state_features": torch.randn(rows, 1, 64).bfloat16(),
            "backbone_attention_mask": torch.ones(rows, 8, dtype=torch.bool),
            "image_mask": torch.tensor([[True] * 4 + [False] * 4] * rows),
            "embodiment_id": torch.zeros(rows, dtype=torch.int64),
        }
        write_projected_episode(root / spec["file"], payload, rows)
    mark_complete(root, manifest)
    return base


def test_disk_cache_preserves_transitions_and_rejects_incomplete(tmp_path):
    pooled, root = tmp_path / "pooled", tmp_path / "projected"
    make_cache(pooled)
    base = make_projected_cache(pooled, root)
    dataset = ProjectedFeatureDataset(pooled, root)
    expected, actual = base.batch([0, 1, 2]), dataset.batch([0, 1, 2])
    for key in ("actions", "action_mask", "rewards", "discounts", "terminated"):
        torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
    torch.testing.assert_close(actual.observations["features"], expected.observations["features"])
    assert actual.observations["projected_vl_features"].dtype == torch.bfloat16
    assert set(actual.observations) == set(FIELDS) | {"features"}
    with pytest.raises(ValueError, match="one episode"):
        dataset.batch([0, 5])
    (root / "COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError):
        ProjectedFeatureDataset(pooled, root)


def test_existing_critic_cache_is_reused_without_replacing_actor_tokens(tmp_path):
    pooled = tmp_path / "pooled"
    make_cache(pooled)
    cache = CachedFeatureDataset(pooled, reward="step-cost")
    batch = cache.batch([0, 1])
    tokens = torch.randn(2, 8, 64)
    live = replace(
        batch,
        observations={
            "features": torch.zeros_like(batch.observations["features"]),
            "projected_vl_features": tokens,
        },
    )
    transitions = [SimpleNamespace(episode_index=0, step_index=i) for i in (0, 1)]
    reused = reuse_critic_cache(live, transitions, cache)
    torch.testing.assert_close(reused.observations["features"], batch.observations["features"])
    assert reused.observations["projected_vl_features"] is tokens
    with pytest.raises(ValueError, match="rewards"):
        reuse_critic_cache(replace(live, rewards=live.rewards + 1), transitions, cache)


@pytest.mark.parametrize("env_q", ["td", "fixed-iql"])
def test_cached_svf_cli_and_resume_without_vlm(tmp_path, monkeypatch, env_q):
    pooled, root, model_root = tmp_path / "pooled", tmp_path / "projected", tmp_path / "bc"
    make_cache(pooled)
    model_root.mkdir()
    config = small_head(True).config
    config.max_action_dim, config.action_horizon = 3, 4
    head = Gr00tN1d7ActionHead(config).eval()
    config.save_pretrained(model_root)
    save_file(
        {
            **{"action_head." + k: v for k, v in head.state_dict().items()},
            "backbone.unused_sentinel": torch.ones(1),
        },
        model_root / "model.safetensors",
    )
    loaded = load_action_head(model_root)
    for k, value in head.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], value.bfloat16())
    manifest = json.loads((pooled / "manifest.json").read_text())
    manifest["identity"].update(
        model_path=str(model_root), dataset_path="unused-data", embodiment="NEW_EMBODIMENT"
    )
    atomic_json(pooled / "manifest.json", manifest)
    complete = json.loads((pooled / "COMPLETE.json").read_text())
    complete["manifest_sha256"] = hashlib.sha256(
        (pooled / "manifest.json").read_bytes()
    ).hexdigest()
    atomic_json(pooled / "COMPLETE.json", complete)
    make_projected_cache(pooled, root)
    monkeypatch.setattr(train_cached_svf, "cache_identity", lambda *args: manifest["identity"])
    args = [
        "--pooled-cache",
        str(pooled),
        "--actor-cache",
        str(root),
        "--output-dir",
        str(tmp_path / "svf"),
        "--env-q",
        env_q,
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--dit-lora-rank",
        "4",
        "--hidden-dim",
        "8",
        "--hidden-layers",
        "1",
        "--flow-steps",
        "1",
        "--candidates",
        "1",
        "--soft-lambda",
        "1",
        "--steps",
        "1",
    ]
    if env_q == "fixed-iql":
        train_iql(
            [
                "--cache",
                str(pooled),
                "--output-dir",
                str(tmp_path / "iql"),
                "--reward",
                "step-cost",
                "--device",
                "cpu",
                "--steps",
                "1",
                "--batch-size",
                "2",
                "--hidden-dim",
                "8",
                "--hidden-layers",
                "1",
            ]
        )
        args += ["--iql-checkpoint", str(tmp_path / "iql/checkpoints/step-1.pt")]
    train_cached_svf.main(args)
    train_cached_svf.main(
        args + ["--steps", "2", "--resume", str(tmp_path / "svf/checkpoints/step-1.pt")]
    )
    assert (tmp_path / "svf/checkpoints/step-2.pt").is_file()
