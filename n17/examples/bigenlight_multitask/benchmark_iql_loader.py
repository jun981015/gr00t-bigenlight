"""Bounded CPU-only real-processor prefetch check; never loads BC model weights."""

import argparse
import json
from pathlib import Path
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    import gr00t.model  # noqa: F401
    from gr00t.rl.adapters import Gr00tTransitionCollator
    from gr00t.rl.dataset import (
        AllSuccessTerminalAnnotations,
        EpisodeBatchSampler,
        LeRobotOfflineRLDataset,
    )
    from gr00t.rl.train import build_batches
    from transformers import AutoProcessor

    torch.set_num_threads(2)
    run = json.loads((args.run / "run.json").read_text())["args"]
    root = Path(run["model_path"])
    processor = AutoProcessor.from_pretrained(
        root if (root / "processor_config.json").exists() else root / "processor"
    )
    processor.eval()
    tag = EmbodimentTag.resolve(run["embodiment_tag"])
    ds = LeRobotOfflineRLDataset(
        LeRobotEpisodeLoader(
            run["dataset_path"][0],
            processor.get_modality_configs()[tag.value],
            decoder_kwargs={"num_ffmpeg_threads": 2},
        ),
        tag,
        AllSuccessTerminalAnnotations(),
        horizon=16,
        cache_episodes=16,
        cache_bytes=4 * 1024**3,
    )
    loader = build_batches(
        ds,
        EpisodeBatchSampler(ds, 32, args.batches, 0, 600),
        Gr00tTransitionCollator(processor),
        workers=args.workers,
    )
    start = previous = time.perf_counter()
    try:
        for i, batch in enumerate(loader):
            now = time.perf_counter()
            print(
                json.dumps(
                    {
                        "batch": i,
                        "wait_s": now - previous,
                        "elapsed_s": now - start,
                        "actions": list(batch.current["action"].shape),
                    }
                ),
                flush=True,
            )
            previous = now
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


if __name__ == "__main__":
    main()
