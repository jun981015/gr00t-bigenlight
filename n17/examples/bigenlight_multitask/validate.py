"""Check vectors/IDs/language, RGB alignment and real N1.7 preprocessing on CPU."""

import argparse
import gc
import json
from pathlib import Path

from examples.bigenlight_multitask.config import CONFIG
from examples.bigenlight_multitask.prepare import DATA_PATH, VIDEO_KEYS, VIDEO_PATH, atomic_json
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, MessageType
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
import numpy as np
import pandas as pd
import torch
from torchcodec.decoders import VideoDecoder


def validate(storage, output=None):
    output = Path(output) if output is not None else storage / "datasets/bigenlight_multitask_gr00t"
    ready = json.loads((output / "READY.json").read_text())
    mapping = json.loads((output / "source_mapping.json").read_text())
    root = output / "n17"
    tasks = {r["task_index"]: r["task"] for r in ready["tasks"]}
    seen, source_tables, values = {}, {}, {"action": [], "observation.state": []}
    offset = 0
    for row in mapping:
        episode, source_id = row["episode_index"], row["source_episode_index"]
        source = storage / "datasets" / row["source_repo"].split("/")[-1]
        source_file = source / row["source_data_path"]
        if source_file not in source_tables:
            source_tables[source_file] = pd.read_parquet(source_file)
        raw = source_tables[source_file]
        raw = raw[raw.episode_index == source_id]
        frame = pd.read_parquet(
            root / DATA_PATH.format(episode_chunk=episode // 1000, episode_index=episode)
        )
        np.testing.assert_array_equal(frame.episode_index, np.full(len(raw), episode))
        np.testing.assert_array_equal(frame["index"], np.arange(offset, offset + len(raw)))
        for key in ("frame_index", "timestamp", "action", "observation.state"):
            np.testing.assert_array_equal(np.stack(frame[key]), np.stack(raw[key]))
            if key in values:
                values[key].append(np.stack(frame[key]))
        source_tasks = pd.read_parquet(source / "meta/tasks.parquet")
        source_tasks = {int(r.task_index): str(text) for text, r in source_tasks.iterrows()}
        assert [tasks[int(t)] for t in frame.task_index] == [
            source_tasks[int(t)] for t in raw.task_index
        ]
        seen.setdefault(row["source_repo"], []).append(row)
        offset += len(frame)
    assert len(mapping) == ready["episodes"] and offset == ready["frames"]
    stats = json.loads((root / "meta/stats.json").read_text())
    for key, chunks in values.items():
        array = np.concatenate(chunks)
        for field, expected in (
            ("mean", array.mean(0)),
            ("std", array.std(0)),
            ("min", array.min(0)),
            ("max", array.max(0)),
            ("q01", np.quantile(array, 0.01, axis=0)),
            ("q99", np.quantile(array, 0.99, axis=0)),
        ):
            np.testing.assert_allclose(stats[key][field], expected, atol=1e-6, rtol=1e-5)
    del values, source_tables
    max_mae, checked = 0.0, 0
    # First/last episodes of EVERY input source, including file-boundary offsets.
    for rows in seen.values():
        for row in (rows[0], rows[-1]):
            episode = row["episode_index"]
            source = storage / "datasets" / row["source_repo"].split("/")[-1]
            for key in VIDEO_KEYS:
                reference = VideoDecoder(
                    str(source / row["videos"][key]["path"]),
                    dimension_order="NHWC",
                    num_ffmpeg_threads=2,
                )
                actual = VideoDecoder(
                    str(
                        root
                        / VIDEO_PATH.format(
                            episode_chunk=episode // 1000, episode_index=episode, video_key=key
                        )
                    ),
                    dimension_order="NHWC",
                    num_ffmpeg_threads=2,
                )
                assert len(actual) == row["length"]
                indices = [0, row["length"] // 2, row["length"] - 1]
                a = actual.get_frames_at(indices).data.numpy()
                b = reference.get_frames_at(
                    [row["videos"][key]["start_frame"] + i for i in indices]
                ).data.numpy()
                mae = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
                assert mae < 4, (episode, key, mae)
                max_mae, checked = max(max_mae, mae), checked + len(indices)
                del reference, actual, a, b
    print(
        f"All {len(mapping)} episode vectors/labels and sampled RGB boundaries verified", flush=True
    )
    loader = LeRobotEpisodeLoader(root, CONFIG, decoder_kwargs={"num_ffmpeg_threads": 2})
    processor = Gr00tN1d7Processor.from_pretrained(
        storage / "models/GR00T-N1.7-3B",
        modality_configs={EmbodimentTag.NEW_EMBODIMENT.value: CONFIG},
        use_relative_action=True,
        use_percentiles=True,
        shortest_image_edge=256,
        crop_fraction=1.0,
        image_crop_size=None,
        image_target_size=None,
    )
    processor.set_statistics(
        {EmbodimentTag.NEW_EMBODIMENT.value: loader.get_dataset_statistics()}, override=True
    )
    processor.eval()
    shapes = {}
    for rows in seen.values():
        episode = loader[rows[0]["episode_index"]]
        for t in (0, len(episode) - 1):
            step = extract_step_data(
                episode, t, CONFIG, EmbodimentTag.NEW_EMBODIMENT, allow_padding=True
            )
            processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            batch = processor.collator([processed])["inputs"]
            assert tuple(batch["action"].shape) == (1, 40, 132)
            assert tuple(batch["state"].shape) == (1, 1, 132)
            assert batch["action_mask"].sum() == 16 * 7
            assert all(
                torch.isfinite(v).all() for v in batch.values() if isinstance(v, torch.Tensor)
            )
            shapes = {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        del episode, step, processed, batch
        gc.collect()
    report = {
        "episodes": len(mapping),
        "frames": offset,
        "tasks": len(tasks),
        "vectors_bit_exact": True,
        "task_labels_preserved": True,
        "statistics_recomputed": True,
        "rgb_sampled_frames": checked,
        "rgb_max_mae": max_mae,
        "action_horizon": 16,
        "action_representation": "absolute joint targets",
        "processor_shapes": shapes,
        "gpu_forward_backward_tested": False,
    }
    atomic_json(root / "VALIDATION.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    validate((Path.home() / "raid/vla_finetune").resolve(), args.dataset_root)
