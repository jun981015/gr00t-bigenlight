#!/usr/bin/env bash
# Live N1.7 SVF: frozen DEAS Q, DiT LoRA + fresh inner soft value.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/environments/activate_n17.sh"
cd "$GR00T_N17_ROOT"
export CUDA_VISIBLE_DEVICES="${SVF_GPU:-0}" WANDB_MODE=online WANDB_ENTITY=junhyeong
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
output="${SVF_OUTPUT:-$VLA_STORAGE_ROOT/outputs/bigenlight-n17-svf-fixed-deas-all-lora16-20260921}"
command=(python -m gr00t.rl.train --backend gr00t --algorithm svf
  --model-path "$VLA_STORAGE_ROOT/outputs/bigenlight-n17-all-b32-10k-20260920T100453/checkpoint-10000"
  --dataset-path "$VLA_STORAGE_ROOT/datasets/bigenlight_multitask_gr00t/n17"
  --fixed-q-checkpoint "$VLA_STORAGE_ROOT/outputs/bigenlight-n17-deas-all-lr1e-4-30k-20260921-deas-dualdiscount/checkpoints/step-30000.pt"
  --fixed-q-cache "$VLA_STORAGE_ROOT/features/bigenlight-n17-all-bc10000-h16-v1"
  --output-dir "$output" --annotation-format all-success-step-cost --horizon 16 --gamma 0.99
  --dit-lora-rank 16 --dit-lora-alpha 32 --freeze-reference --q-aggregation min
  --batch-size 2 --learning-rate 3e-4 --flow-steps 10 --candidates 8
  --device cuda:0 --cpu-threads 2 --loader-workers 4 --loader-prefetch 2
  --episode-cache-count 16 --episode-cache-gib 2 --decoder-threads 2
  --wandb-project bigenlight-multitask-gr00t --wandb-log-every 50
  --save-every 5000 --first-save-step 100 --save-interval-seconds 1800
  --max-run-seconds 28800 --keep-latest-training-state)
if [[ "${1:-}" != --execute ]]; then
  printf '%q ' "${command[@]}" --steps 10000; printf '\n'
  exit 0
fi
test -f /.dockerenv
# A completed ten-step startup also supplies the first resumable checkpoint.
if [[ ! -f "$output/checkpoints/latest_resumable.json" ]]; then
  "${command[@]}" --steps 10
fi
resume=$(python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()); print(p.parent/s["file"])' "$output/checkpoints/latest_resumable.json")
exec "${command[@]}" --steps 10000 --resume "$resume"
