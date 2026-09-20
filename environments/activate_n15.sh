# Usage: source environments/activate_n15.sh (bash).
GR00T_WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GR00T_WORKSPACE_ROOT GR00T_N15_ROOT="$GR00T_WORKSPACE_ROOT/n15"
source "$GR00T_WORKSPACE_ROOT/storage_env.sh"
export TMPDIR="$VLA_STORAGE_ROOT/tmp" USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1
export UV_PROJECT_ENVIRONMENT="$VLA_STORAGE_ROOT/envs/deas-gr00t-n1.5"
source "$UV_PROJECT_ENVIRONMENT/bin/activate" || return 1
export PYTHONPATH="$GR00T_N15_ROOT"
