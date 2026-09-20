"""Dependency, CPU-physics, and GPU-camera checks are deliberately separate."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

from manage import PACKAGE, SUITES, config_dict, selected_tasks, storage, task_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("doctor", "cpu-smoke", "smoke"))
    parser.add_argument("--suite", choices=SUITES, default="libero_spatial")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task")
    group.add_argument("--all-tasks", action="store_true")
    args = parser.parse_args()

    config_path = Path(os.environ.get("LIBERO_CONFIG_PATH", "/nonexistent")) / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError("Run local/setup.sh or local/run.sh configure before importing LIBERO")
    import yaml

    if yaml.safe_load(config_path.read_text()) != config_dict():
        raise RuntimeError(f"Unexpected LIBERO configuration: {config_path}")

    from gr00t.eval import rollout_policy  # noqa: F401
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    from gr00t.policy.server_client import PolicyClient  # noqa: F401
    import gymnasium as gym
    from libero.libero.envs.env_wrapper import ControlEnv
    import numpy as np

    register_libero_envs()
    for names in task_map().values():
        for name in names:
            gym.spec(f"libero_sim/{name}")
    print(
        json.dumps(
            {
                name: importlib.metadata.version(name)
                for name in ("mujoco", "robosuite", "gymnasium", "torch", "numpy")
            },
            indent=2,
        )
    )
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg missing from PATH")
    print(
        "Imports / RPC client / rollout / ffmpeg / 130 task registrations OK; rendering NOT checked.",
        flush=True,
    )
    if args.mode == "doctor":
        return
    for task in selected_tasks(args):
        if args.mode == "cpu-smoke":
            env = ControlEnv(
                bddl_file_name=str(PACKAGE / "bddl_files" / args.suite / (task + ".bddl")),
                use_camera_obs=False,
                has_renderer=False,
                has_offscreen_renderer=False,
                ignore_done=True,
            )
            try:
                env.seed(42)
                obs = env.reset()
                assert np.isfinite(obs["robot0_eef_pos"]).all()
                for _ in range(3):
                    obs, reward, done, info = env.step(np.zeros(7))
                    assert np.isfinite(reward) and np.isfinite(obs["robot0_eef_pos"]).all()
                    if done:
                        break
                print(f"CPU physics reset/step OK (NO images): {args.suite}/{task}", flush=True)
            finally:
                env.close()
        else:
            env = gym.make(f"libero_sim/{task}")
            try:
                obs, info = env.reset(seed=42)
                for key in ("video.image", "video.wrist_image"):
                    assert obs[key].shape == (256, 256, 3) and obs[key].dtype == np.uint8
                    assert np.std(obs[key]) > 0, f"Blank image: {key}"
                assert obs["annotation.human.action.task_description"]
                for _ in range(3):
                    action = {
                        key: np.zeros(space.shape, dtype=space.dtype)
                        for key, space in env.action_space.items()
                    }
                    obs, reward, terminated, truncated, info = env.step(action)
                    assert np.isfinite(reward) and "success" in info
                    if terminated or truncated:
                        break
                from PIL import Image

                output = storage() / "logs/libero-n17/smoke" / args.suite / task
                output.mkdir(parents=True, exist_ok=True)
                for key in ("video.image", "video.wrist_image"):
                    Image.fromarray(obs[key]).save(output / f"{key}.png")
                print(f"GPU RGB/reset/step OK: {task}; images: {output}", flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    main()
