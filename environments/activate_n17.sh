# Usage: source environments/activate_n17.sh (bash).
GR00T_WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GR00T_WORKSPACE_ROOT GR00T_N17_ROOT="$GR00T_WORKSPACE_ROOT/n17"
source "$GR00T_WORKSPACE_ROOT/environments/activate_gr00t.sh" || return 1
# Override the legacy editable install and any inherited N1.5 PYTHONPATH.
export PYTHONPATH="$GR00T_N17_ROOT"
