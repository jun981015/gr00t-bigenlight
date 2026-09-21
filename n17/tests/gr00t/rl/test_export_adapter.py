from copy import deepcopy

from gr00t.rl.export_adapter import load_actor_adapter, split_actor
from gr00t.rl.projected_actor import LoRALinear
import pytest
from safetensors.torch import save_file
import torch


def test_adapter_reconstructs_output_without_base_weights(tmp_path):
    torch.manual_seed(1)
    base = torch.nn.Linear(5, 4)
    trained = LoRALinear(deepcopy(base), 2, 4)
    with torch.no_grad():
        trained.lora_B.normal_()
    # Match deployed actor key layout.
    actor = torch.nn.ModuleDict({"head": torch.nn.ModuleDict({"layer": trained})})
    fresh = torch.nn.ModuleDict(
        {"head": torch.nn.ModuleDict({"layer": LoRALinear(deepcopy(base), 2, 4)})}
    )
    adapters, frozen = split_actor(actor.state_dict())
    assert set(frozen) == {"action_head.layer.weight", "action_head.layer.bias"}
    assert len(adapters) == 2
    path = tmp_path / "adapter.safetensors"
    save_file(adapters, str(path))
    load_actor_adapter(fresh, path)
    x = torch.randn(7, 5)
    torch.testing.assert_close(actor["head"]["layer"](x), fresh["head"]["layer"](x), rtol=0, atol=0)
    save_file({"wrong": torch.ones(1)}, str(path))
    with pytest.raises(ValueError):
        load_actor_adapter(fresh, path)
