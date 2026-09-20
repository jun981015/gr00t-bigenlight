#!/usr/bin/env bash
# Non-destructive alternative to the upstream setup script (which deletes its venv).
set -euo pipefail
SIMPLER_HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIMPLER_ROOT="$(cd -- "$SIMPLER_HERE/../../.." && pwd)"
source "$SIMPLER_ROOT/../storage_env.sh"
SIMPLER_ENV="$VLA_STORAGE_ROOT/envs/simpler-n17-client"
SIMPLER_UV="${SIMPLER_UV:-$HOME/raid/uv/bin/uv}"
export TMPDIR="$VLA_STORAGE_ROOT/tmp"
mkdir -p "$TMPDIR" "$VLA_STORAGE_ROOT/logs/simpler-n17"
exec 9>"$VLA_STORAGE_ROOT/logs/simpler-n17/setup.lock"
flock -n 9 || { echo 'Another SimplerEnv setup is running.' >&2; exit 1; }
cd "$SIMPLER_ROOT"
python3 "$SIMPLER_HERE/manage.py" sources
if [[ ! -e "$SIMPLER_ENV" ]]; then
    "$SIMPLER_UV" venv "$SIMPLER_ENV" --python 3.10
elif [[ ! -f "$SIMPLER_ENV/pyvenv.cfg" ]]; then
    echo "Refusing to modify a non-venv directory: $SIMPLER_ENV" >&2
    exit 1
fi
"$SIMPLER_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
# Both indexes are official upstreams; torch/torchvision are pinned to +cpu.
"$SIMPLER_UV" pip install --python "$SIMPLER_ENV/bin/python" --index-strategy unsafe-best-match \
    -r "$SIMPLER_HERE/requirements.txt" \
    -e "$SIMPLER_ROOT/external_dependencies/SimplerEnv/ManiSkill2_real2sim" \
    -e "$SIMPLER_ROOT/external_dependencies/SimplerEnv"
"$SIMPLER_UV" pip check --python "$SIMPLER_ENV/bin/python"
"$SIMPLER_UV" pip freeze --python "$SIMPLER_ENV/bin/python" \
    > "$VLA_STORAGE_ROOT/logs/simpler-n17/installed-requirements.txt"
bash "$SIMPLER_HERE/run.sh" doctor
