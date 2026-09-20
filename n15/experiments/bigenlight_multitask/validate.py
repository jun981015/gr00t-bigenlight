"""Exercise real DEAS N1.5 transforms and padding on all four tasks, CPU only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.bigenlight_multitask.config import UR7eDataConfig
from experiments.robocasa_deas.recovery import atomic_json
from gr00t.data.dataset import LeRobotSingleDataset


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path.home() / "raid/vla_finetune/datasets/bigenlight_multitask_gr00t/n15",
    )
    root = parser.parse_args().dataset_path
    info = json.loads((root / "meta/info.json").read_text())
    config = UR7eDataConfig()
    dataset = LeRobotSingleDataset(
        root,
        config.modality_config(),
        "new_embodiment",
        video_backend="torchvision_av",
        transforms=config.transform(),
    )
    seen, offset = set(), 0
    for row in map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines()):
        task = row["tasks"][0]
        if task not in seen:
            for step in (offset, offset + row["length"] - 1):
                sample = dataset[step]
                assert sample["action"].shape == (16, 32)
                assert sample["state"].shape == (1, 64)
                assert sample["action_mask"].sum() == 16 * 7
                assert np.isfinite(sample["action"]).all() and np.isfinite(sample["state"]).all()
                assert sample["eagle_content"]
            seen.add(task)
        offset += row["length"]
    assert len(seen) == info["total_tasks"] and len(dataset) == offset == info["total_frames"]
    result = {
        "tasks": sorted(seen),
        "frames": offset,
        "episodes": info["total_episodes"],
        "action_shape": [16, 32],
        "state_shape": [1, 64],
        "action_horizon": 16,
        "action_representation": "absolute joint targets",
        "gpu_forward_backward_tested": False,
    }
    atomic_json(root / "VALIDATION.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
