"""Bounded CPU benchmark of full vs seek decoding; never starts training.

One first/middle episode per dataset, sampled at its middle timestep. Compare
every raw video/state/action/language value and time raw loading vs transforms.
Both modes disable frame caching for a conservative cold-frame comparison.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path.home() / "raid/vla_finetune/outputs/deas-n15-filtered-bc-direct-1gpu-b32/launch_recipe.json",
    )
    args = parser.parse_args()
    config = json.loads(args.recipe.read_text())["config"]
    data_config = DATA_CONFIG_MAP[config["data_config"]]
    torch.set_num_threads(1)
    os.environ["GR00T_PYAV_CACHE_BYTES"] = "0"
    totals = {"full": 0.0, "seek": 0.0, "transform": 0.0}
    count = 0
    for path in config["dataset_path"]:
        if not (Path(path) / "meta/stats.json").exists():
            raise ValueError("Precomputed stats required; benchmark must not generate data")
        dataset = LeRobotSingleDataset(
            path,
            data_config.modality_config(),
            config["embodiment_tag"],
            video_backend="torchvision_av",
            transforms=data_config.transform(),
        )
        for episode_index in (0, len(dataset.trajectory_ids) // 2):
            episode = int(dataset.trajectory_ids[episode_index])
            step = int(dataset.trajectory_lengths[episode_index] // 2)
            samples, times = {}, {}
            # Alternate order to avoid always warming filesystem cache for seek.
            modes = ("full", "seek") if count % 2 == 0 else ("seek", "full")
            for mode in modes:
                os.environ["GR00T_PYAV_DECODE_MODE"] = mode
                dataset.curr_traj_id, dataset.curr_traj_data = None, None
                started = time.perf_counter()
                samples[mode] = dataset.get_step_data(episode, step)
                times[mode] = time.perf_counter() - started
                totals[mode] += times[mode]
            for key in samples["full"]:
                np.testing.assert_array_equal(samples["full"][key], samples["seek"][key])
            started = time.perf_counter()
            result = dataset.transforms(samples["seek"])
            totals["transform"] += time.perf_counter() - started
            assert np.isfinite(result["action"]).all()
            count += 1
            print(
                json.dumps(
                    {"dataset": path, "episode": episode, "seconds": times, "all_raw_values_bit_exact": True}
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "samples": count,
                "total_seconds": totals,
                "raw_loading_speedup": totals["full"] / totals["seek"],
                "cpu_preparation_speedup": (totals["full"] + totals["transform"])
                / (totals["seek"] + totals["transform"]),
                "gpu_training_tested": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
