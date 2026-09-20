"""Check real GR00T weights with synthetic observations, without robot control."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from gr00t.policy.gr00t_policy import Gr00tPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=str(Path.home() / "raid/vla_finetune/models/GR00T-N1.7-3B"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    started = time.monotonic()
    policy = Gr00tPolicy(
        embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        model_path=args.model_path,
        device=args.device,
    )
    print(f"Real checkpoint loaded in {time.monotonic() - started:.1f}s", flush=True)
    config = policy.modality_configs
    observation = {"video": {}, "state": {}, "language": {}}
    for key in config["video"].modality_keys:
        shape = (1, len(config["video"].delta_indices), 256, 256, 3)
        observation["video"][key] = rng.integers(0, 256, shape, dtype=np.uint8)

    state_config = policy.processor.state_action_processor.norm_params[
        policy.embodiment_tag.value
    ]["state"]
    for key in config["state"].modality_keys:
        shape = (1, len(config["state"].delta_indices), int(state_config[key]["dim"]))
        state = np.zeros(shape, dtype=np.float32)
        if key == "eef_9d":
            state[..., 3:9] = [1, 0, 0, 0, 1, 0]
        observation["state"][key] = state
    observation["language"][config["language"].modality_keys[0]] = [["pick up the red cube"]]

    with torch.inference_mode():
        actions, _ = policy.get_action(observation)
    expected_keys = set(config["action"].modality_keys)
    assert set(actions) == expected_keys
    for key, value in actions.items():
        assert value.ndim == 3 and value.shape[0] == 1, (key, value.shape)
        assert value.shape[1] == len(config["action"].delta_indices)
        assert np.isfinite(value).all(), f"Non-finite output: {key}"
        print(f"Action OK: {key}, shape={value.shape}, dtype={value.dtype}", flush=True)
    print("PASS: real-weight inference with synthetic input; no robot connected.", flush=True)


if __name__ == "__main__":
    main()
