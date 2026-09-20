#!/usr/bin/env bash
set -eo pipefail
LIBERO_N15_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$LIBERO_N15_DIR/../robocasa_deas/activate.sh"
export WANDB_PROJECT=libero-gr00t-n15 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
# At most 8 GiB per DataLoader worker, CPU only. The source demos stay unchanged.
export GR00T_PYAV_CACHE_BYTES=8589934592
cd "$DEAS_REPO_DIR"
exec python "$LIBERO_N15_DIR/manage.py" "$@"
