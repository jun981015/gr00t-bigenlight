"""Generate a small, entirely synthetic LeRobot v2 SO100 training dataset."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "raid/vla_finetune/datasets/synthetic_so100_vram",
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1 or args.frames < 40:
        parser.error("Use at least one episode and 40 frames per episode")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing dataset.
    meta = root / "meta"
    meta.mkdir()
    repo = Path(__file__).resolve().parents[2] / "n17"
    schema = repo / "demo_data/cube_to_bowl_5/meta"
    info = json.loads((schema / "info.json").read_text())
    info.update(
        robot_type="synthetic_so100_NOT_REAL_ROBOT_DATA",
        total_episodes=args.episodes,
        total_frames=args.episodes * args.frames,
        total_tasks=1,
        total_chunks=1,
        total_videos=args.episodes * 2,
        splits={"train": f"0:{args.episodes}"},
    )
    for feature in info["features"].values():
        if feature["dtype"] == "video":
            feature["shape"] = [256, 256, 3]
            feature["info"].update(
                {
                    "video.height": 256,
                    "video.width": 256,
                    "video.codec": "h264",
                    "video.fps": 30,
                }
            )
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (meta / "modality.json").write_text((schema / "modality.json").read_text())
    task = "Pick up the red cube and place it in the bowl."
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    rng = np.random.default_rng(args.seed)
    yy, xx = np.mgrid[:256, :256]
    episodes = []
    data_dir = root / "data/chunk-000"
    data_dir.mkdir(parents=True)
    for episode in range(args.episodes):
        t = np.arange(args.frames, dtype=np.float32) / 30
        phase = rng.uniform(0, 2 * np.pi, size=6)
        state = (25 * np.sin(t[:, None] * np.linspace(0.5, 1.5, 6) + phase)).astype(np.float32)
        state[:, 5] = 50 + 40 * np.sin(t + phase[5])
        action = np.roll(state, -1, axis=0) + rng.normal(0, 0.15, state.shape)
        action[-1] = action[-2]
        frame_index = np.arange(args.frames, dtype=np.int64)
        pd.DataFrame(
            {
                "observation.state": list(state),
                "action": list(action.astype(np.float32)),
                "timestamp": t,
                "frame_index": frame_index,
                "episode_index": np.full(args.frames, episode, dtype=np.int64),
                "index": frame_index + episode * args.frames,
                "task_index": np.zeros(args.frames, dtype=np.int64),
            }
        ).to_parquet(data_dir / f"episode_{episode:06d}.parquet", index=False)
        for camera_index, camera in enumerate(("front", "wrist")):
            video_dir = root / f"videos/chunk-000/observation.images.{camera}"
            video_dir.mkdir(parents=True, exist_ok=True)
            target = video_dir / f"episode_{episode:06d}.mp4"
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "256x256",
                "-r",
                "30",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "veryfast",
                "-crf",
                "25",
                "-pix_fmt",
                "yuv420p",
                str(target),
            ]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
                for frame in range(args.frames):
                    rgb = np.stack(
                        [
                            (xx + frame * 2 + episode * 23) % 256,
                            (yy + camera_index * 60 + frame) % 256,
                            ((xx + yy) // 2 + frame * 3) % 256,
                        ],
                        axis=-1,
                    ).astype(np.uint8)
                    left, top = (frame * 3 + 20) % 200, (frame * 2 + 30) % 200
                    rgb[top : top + 32, left : left + 32] = [240, 25, 25]
                    encoder.stdin.write(rgb.tobytes())
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed: {target}")
        episodes.append({"episode_index": episode, "tasks": [task], "length": args.frames})
        print(
            f"Generated synthetic episode {episode}: {args.frames} frames, two cameras", flush=True
        )
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (root / "SYNTHETIC_DATA.json").write_text(
        json.dumps(
            {
                "purpose": "VRAM / training pipeline smoke test only; NOT a quality benchmark",
                "seed": args.seed,
                "episodes": args.episodes,
                "frames_per_episode": args.frames,
                "cameras": 2,
                "height": 256,
                "width": 256,
                "fps": 30,
                "state_and_action_dimensions": 6,
            },
            indent=2,
        )
        + "\n"
    )
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"Dataset ready: {root} ({size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
