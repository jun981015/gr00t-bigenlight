#!/usr/bin/env bash
# Install the RoboCasa simulator into the isolated DEAS/GR00T-N1.5 env.
# This intentionally does not install or upgrade torch.
set -euo pipefail

RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEAS_REPO=$(cd -- "$RECIPE_DIR/../.." && pwd)
source "$DEAS_REPO/../storage_env.sh"

DEAS_ENV="${DEAS_ENV:-$VLA_STORAGE_ROOT/envs/deas-gr00t-n1.5}"
UV_BIN="${UV_BIN:-$HOME/raid/uv/bin/uv}"
ROBOCASA_REPO="${ROBOCASA_REPO:-$DEAS_REPO/../n17/external_dependencies/robocasa}"
ROBO_SUITE_REF="85abee228d1c43ab1939bce33028099945d453b4"

# Keep this list compatible with Python 3.10 and the existing N1.5 torch 2.5.1.
# In particular, robocasa's setup.py pins numpy/numba versions that are not
# installable together on Python 3.10 in this environment.
SIM_DEPS=(
  "gymnasium==0.29.1"
  "numpy==1.26.4"
  "numba"
  "scipy"
  "mujoco==3.2.6"
  "pygame"
  "Pillow"
  "opencv-python-headless==4.11.0.86"
  "pyyaml"
  "pynput"
  "tqdm"
  "termcolor"
  "imageio"
  "h5py"
  "lxml"
  "hidapi"
  "tianshou"
  "pydantic"
  "av==12.3.0"
  "pyzmq"
  "msgpack"
  "msgpack-numpy"
)

[[ -x "$DEAS_ENV/bin/python" ]] || {
  echo "Missing N1.5 environment: $DEAS_ENV" >&2
  echo "Run experiments/robocasa_deas/bootstrap.sh first." >&2
  exit 1
}
[[ -x "$UV_BIN" ]] || { echo "Missing uv: $UV_BIN" >&2; exit 1; }
[[ -d "$ROBOCASA_REPO/robocasa" ]] || {
  echo "Missing RoboCasa source: $ROBOCASA_REPO" >&2
  echo "Set ROBOCASA_REPO to a checked-out squarefk/robocasa tree." >&2
  exit 1
}

echo "Installing simulator dependencies into $DEAS_ENV"
"$UV_BIN" pip install --python "$DEAS_ENV/bin/python" "${SIM_DEPS[@]}"

# DEAS N1.5's RoboCasa wrapper expects the same robosuite commit as the
# official Isaac-GR00T RoboCasa setup. --no-deps prevents torch/other base
# packages from being re-resolved.
"$UV_BIN" pip install --python "$DEAS_ENV/bin/python" --no-deps \
  "git+https://github.com/ARISE-Initiative/robosuite.git@$ROBO_SUITE_REF"
"$UV_BIN" pip install --python "$DEAS_ENV/bin/python" --no-deps \
  --editable "$ROBOCASA_REPO"

if [[ "${SKIP_DOWNLOAD_ASSETS:-0}" == "1" ]]; then
  echo "Skipping RoboCasa assets (SKIP_DOWNLOAD_ASSETS=1)."
else
  # Kitchen assets are several GB and are stored inside the checked-out
  # RoboCasa tree, which is on the RAID-backed workspace in this setup.
  "$DEAS_ENV/bin/python" \
    "$ROBOCASA_REPO/robocasa/scripts/download_kitchen_assets.py"
fi

MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
  "$DEAS_ENV/bin/python" - <<'PY'
import gymnasium as gym
import mujoco
import robocasa
import robosuite
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

print("RoboCasa imports OK")
print("mujoco:", mujoco.__version__)
print("robocasa:", getattr(robocasa, "__version__", "unknown"))
print("robosuite:", getattr(robosuite, "__version__", "unknown"))
env = gym.make("robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env", enable_render=True)
print("RoboCasa environment OK:", type(env).__name__)
env.close()
PY

echo "DEAS N1.5 RoboCasa simulator is ready."
