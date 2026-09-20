#!/usr/bin/env bash
set -eo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/activate_gr00t.sh"
cd "$GR00T_N17_ROOT"
exec python examples/robocasa_svf/launch.py "$@"
