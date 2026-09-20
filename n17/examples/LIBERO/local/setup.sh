#!/usr/bin/env bash
# Preserve both the training env and any existing ~/.libero configuration.
set -euo pipefail
LIBERO_HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_ROOT="$(cd -- "$LIBERO_HERE/../../.." && pwd)"
source "$LIBERO_ROOT/../storage_env.sh"
LIBERO_ENV="$VLA_STORAGE_ROOT/envs/libero-n17-client"
LIBERO_UV="${LIBERO_UV:-$HOME/raid/uv/bin/uv}"
export TMPDIR="$VLA_STORAGE_ROOT/tmp"
mkdir -p "$TMPDIR" "$VLA_STORAGE_ROOT/logs/libero-n17"
exec 9>"$VLA_STORAGE_ROOT/logs/libero-n17/setup.lock"
flock -n 9 || { echo 'Another LIBERO setup is running.' >&2; exit 1; }
cd "$LIBERO_ROOT"
python3 "$LIBERO_HERE/manage.py" sources
if [[ ! -e "$LIBERO_ENV" ]]; then
    # Reuse the RAID-backed interpreter, not the host-only /usr/bin/python3.12.
    "$LIBERO_UV" venv "$LIBERO_ENV" --python "$VLA_STORAGE_ROOT/envs/gr00t-n1.7/bin/python"
elif [[ ! -f "$LIBERO_ENV/pyvenv.cfg" ]]; then
    echo "Refusing to modify a non-venv directory: $LIBERO_ENV" >&2
    exit 1
fi
"$LIBERO_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
# Use the official PyPI and PyTorch indexes; torch is explicitly pinned to +cpu.
"$LIBERO_UV" pip install --python "$LIBERO_ENV/bin/python" --index-strategy unsafe-best-match \
    -r "$LIBERO_HERE/requirements.txt" -e "$LIBERO_ROOT/external_dependencies/LIBERO" \
    --config-settings editable_mode=compat
"$LIBERO_UV" pip check --python "$LIBERO_ENV/bin/python"
"$LIBERO_UV" pip freeze --python "$LIBERO_ENV/bin/python" \
    > "$VLA_STORAGE_ROOT/logs/libero-n17/installed-requirements.txt"
bash "$LIBERO_HERE/run.sh" configure
bash "$LIBERO_HERE/run.sh" doctor
