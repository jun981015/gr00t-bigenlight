#!/usr/bin/env bash
# Isolated N1.5 environment; never install into the N1.7 environment.
set -euo pipefail
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEAS_REPO=$(cd -- "$RECIPE_DIR/../.." && pwd)
source "$DEAS_REPO/../storage_env.sh"
export TMPDIR="$VLA_STORAGE_ROOT/tmp"
export UV_PYTHON_INSTALL_DIR="$HOME/raid/uv/python"
DEAS_ENV="$VLA_STORAGE_ROOT/envs/deas-gr00t-n1.5"
UV_BIN="$HOME/raid/uv/bin/uv"
mkdir -p "$TMPDIR"
if [[ ! -x "$DEAS_ENV/bin/python" ]]; then
    "$UV_BIN" venv --python 3.10 "$DEAS_ENV"
fi
"$UV_BIN" pip install --python "$DEAS_ENV/bin/python" -e "$DEAS_REPO[base]" \
    'setuptools<81' wheel ninja packaging 'huggingface-hub<1' 'tyro<1'
# Both OpenCV distributions install the same cv2 files. Keep only headless for
# GPU containers without desktop GLib/Qt libraries; never modify the N1.7 env.
"$UV_BIN" pip uninstall --python "$DEAS_ENV/bin/python" opencv-python
"$UV_BIN" pip install --python "$DEAS_ENV/bin/python" --no-deps \
    --reinstall-package opencv-python-headless 'opencv-python-headless==4.11.0.86'
if [[ "${1:-}" == "--flash-attn" ]]; then
    # Run in the allocated GPU container with CUDA toolkit available.
    [[ -f /.dockerenv ]] || { echo 'FlashAttention setup requires the GPU container'; exit 1; }
    export MAX_JOBS=4
    "$UV_BIN" pip install --python "$DEAS_ENV/bin/python" --no-build-isolation 'flash-attn==2.7.1.post4'
fi
"$UV_BIN" pip freeze --python "$DEAS_ENV/bin/python" > "$DEAS_ENV/requirements-installed.txt"
echo "Environment: $DEAS_ENV (FlashAttention and simulator smoke tests still required)"
