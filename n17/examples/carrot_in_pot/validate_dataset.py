"""CPU verification: unchanged vectors, RGB alignment, train-only stats, GR00T batch."""

import argparse
import gc
import json
from pathlib import Path
import runpy

from examples.carrot_in_pot.prepare_dataset import DATA_PATH, VIDEO_KEYS, VIDEO_PATH, write_json
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, MessageType
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
import numpy as np
import pandas as pd
import torch
from torchcodec.decoders import VideoDecoder


def validate(source, root, model):
    source, root, model = Path(source), Path(root), Path(model)
    if not (root / "PREPARATION_COMPLETE.json").exists():
        raise ValueError("Preparation has not completed")
    mapping = json.loads((root / "source_mapping.json").read_text())["episodes"]
    original = pd.read_parquet(source / "data/chunk-000/file-000.parquet")
    src_decoders = {
        key: VideoDecoder(
            str(source / f"videos/{key}/chunk-000/file-000.mp4"),
            device="cpu",
            dimension_order="NHWC",
            num_ffmpeg_threads=2,
        )
        for key in VIDEO_KEYS
    }
    max_mae, min_correlation, checked_frames = 0.0, 1.0, 0
    train_values = {"observation.state": [], "action": []}
    for number, row in enumerate(mapping, 1):
        split, index, length = row["split"], row["episode_index"], row["length"]
        frame = pd.read_parquet(
            root / split / DATA_PATH.format(episode_chunk=index // 1000, episode_index=index)
        )
        raw = original[original["episode_index"] == row["source_episode_index"]]
        for key in ("observation.state", "action"):
            values = np.stack(frame[key])
            np.testing.assert_array_equal(values, np.stack(raw[key]))
            if split == "train":
                train_values[key].append(values)
        indices = [0, length // 2, length - 1]
        for key in VIDEO_KEYS:
            path = (
                root
                / split
                / VIDEO_PATH.format(episode_chunk=index // 1000, video_key=key, episode_index=index)
            )
            decoder = VideoDecoder(
                str(path), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=2
            )
            if len(decoder) != length:
                raise ValueError(f"Decoded frame count mismatch: {path}")
            current = decoder.get_frames_at(indices).data.numpy()
            reference = (
                src_decoders[key]
                .get_frames_at([row["source_from_index"] + i for i in indices])
                .data.numpy()
            )
            for actual, expected in zip(current, reference, strict=True):
                mae = float(np.abs(actual.astype(np.float32) - expected.astype(np.float32)).mean())
                corr = float(
                    np.corrcoef(actual[::4, ::4].ravel(), expected[::4, ::4].ravel())[0, 1]
                )
                if not np.isfinite(corr) or mae > 4.0 or corr < 0.99:
                    raise ValueError(
                        f"Possible frame misalignment: {path}: MAE={mae}, correlation={corr}"
                    )
                max_mae, min_correlation = max(max_mae, mae), min(min_correlation, corr)
                checked_frames += 1
            del decoder, current, reference
        if number % 10 == 0:
            print(f"Verified vectors + RGB alignment: {number}/{len(mapping)} episodes", flush=True)
    for key, chunks in train_values.items():
        values = np.concatenate(chunks)
        stats = json.loads((root / "train/meta/stats.json").read_text())[key]
        np.testing.assert_allclose(stats["min"], values.min(0), rtol=0, atol=1e-7)
        np.testing.assert_allclose(stats["max"], values.max(0), rtol=0, atol=1e-7)
    for name in ("stats.json", "relative_stats.json"):
        assert (root / "train/meta" / name).read_bytes() == (root / "val/meta" / name).read_bytes()
    del src_decoders, train_values, original
    gc.collect()
    config = runpy.run_path(str(Path(__file__).with_name("carrot_config.py")))["carrot_config"]
    loader = LeRobotEpisodeLoader(root / "train", config, decoder_kwargs={"num_ffmpeg_threads": 2})
    print("Loading real GR00T processor (no model weights / GPU)", flush=True)
    processor = Gr00tN1d7Processor.from_pretrained(
        model,
        modality_configs={EmbodimentTag.NEW_EMBODIMENT.value: config},
        use_relative_action=True,
        use_percentiles=True,
        shortest_image_edge=256,
        crop_fraction=1.0,
        image_crop_size=None,
        image_target_size=None,
    )
    if processor.shortest_image_edge != 256 or processor.crop_fraction != 1.0:
        raise ValueError("Pretrained processor ignored requested image preprocessing overrides")
    processor.set_statistics(
        {EmbodimentTag.NEW_EMBODIMENT.value: loader.get_dataset_statistics()}, override=True
    )
    processor.eval()
    episode = loader[0]
    examples = [
        extract_step_data(episode, t, config, EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)
        for t in (0, 16)
    ]
    processed = [
        processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]) for step in examples
    ]
    batch = processor.collator(processed)["inputs"]
    if tuple(batch["action"].shape) != (2, 40, 132):
        raise ValueError(f"Unexpected action shape: {batch['action'].shape}")
    assert tuple(batch["state"].shape) == (2, 1, 132)
    assert torch.all(batch["action_mask"][:, :16, :7] == 1)
    assert batch["action_mask"].sum().item() == 2 * 16 * 7
    assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v, torch.Tensor))
    report = {
        "episodes": len(mapping),
        "frames": sum(row["length"] for row in mapping),
        "state_action_bit_exact": True,
        "rgb_sampled_frames": checked_frames,
        "rgb_max_mae_0_255": max_mae,
        "rgb_min_correlation": min_correlation,
        "train_only_statistics": True,
        "gr00t_processor": type(processor).__name__,
        "batch_shapes": {
            key: list(value.shape)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        },
        "model_forward_backward_tested": False,
    }
    write_json(root / "VALIDATION.json", report)
    processor.save_pretrained(root / "processor_preview")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    storage = Path.home() / "raid/vla_finetune"
    parser.add_argument(
        "--source", type=Path, default=storage / "datasets/carrot_in_pot_lerobot_v3"
    )
    parser.add_argument("--root", type=Path, default=storage / "datasets/carrot_in_pot_gr00t")
    parser.add_argument("--model", type=Path, default=storage / "models/GR00T-N1.7-3B")
    args = parser.parse_args()
    torch.set_num_threads(2)
    validate(args.source, args.root, args.model)
