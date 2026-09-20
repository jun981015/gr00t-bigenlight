"""CPU dependency check and explicitly requested GPU reset/step smoke test."""

import argparse
import importlib.metadata
import json
import shutil

from manage import ROBOTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("doctor", "smoke"))
    parser.add_argument("--robot", choices=ROBOTS, default="widowx")
    parser.add_argument("--all-tasks", action="store_true")
    args = parser.parse_args()

    from gr00t.eval import rollout_policy  # noqa: F401
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
    from gr00t.policy.server_client import PolicyClient  # noqa: F401
    import gymnasium as gym
    import numpy as np
    import sapien.core  # noqa: F401

    register_simpler_envs()
    for robot, config in ROBOTS.items():
        for task in config["tasks"]:
            gym.spec(f"simpler_env_{robot}/{task}")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("sapien", "gymnasium", "numpy", "torch", "transformers")
    }
    print(json.dumps(versions, indent=2))
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg missing from PATH; source run.sh environment")
    print(
        "CPU imports, policy client, rollout module, ffmpeg and 13 registrations OK. GPU rendering NOT checked."
    )
    if args.mode == "doctor":
        return
    tasks = ROBOTS[args.robot]["tasks"]
    for task in tasks if args.all_tasks else tasks[:1]:
        env = gym.make(f"simpler_env_{args.robot}/{task}")
        try:
            obs, info = env.reset(seed=42)
            image_key = "video.image_0" if args.robot == "widowx" else "video.image"
            expected = (256, 256, 3) if args.robot == "widowx" else (256, 320, 3)
            assert obs[image_key].shape == expected
            assert obs[image_key].dtype == np.uint8
            assert obs["annotation.human.action.task_description"]
            for _ in range(3):
                action = {
                    key: np.zeros(space.shape, dtype=space.dtype)
                    for key, space in env.action_space.items()
                }
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(reward)
                assert "success" in info
                if terminated or truncated:
                    break
            print(f"GPU reset/RGB/step OK: {task}; success={info['success']}", flush=True)
        finally:
            # Existing GR00T adapter has no close override; close actual simulator too.
            env.unwrapped.env.close()
            env.close()


if __name__ == "__main__":
    main()
