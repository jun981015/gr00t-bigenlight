# Frozen BC critic feature caches

This is a separate path: currently running live-VLM IQL jobs are not restarted or
switched to cache training. The extractor shares GPU 0 with the current learner,
using batch 8, two CPU/decoder threads and a 25 ms yield between batches. Both BC
variants are extracted sequentially in a persistent session, not simultaneously.
Expect some interference with live training while extraction runs.

## Contents and semantics

Each episode file stores all observation rows as FP32 pooled conditioning
(`2048 VLM + 132 normalized state + 32 embodiment = 2212` dimensions), plus compact
normalized **16-step action chunks** for valid starts. Action normalization is
performed on the entire chunk by the actual checkpoint processor, so relative
action configurations are not incorrectly reconstructed from per-frame actions.
The original `40 x 132` action shape and exact padding mask are restored on read.
Final incomplete action chunks are not training samples. The last observation is
retained for successors; a missing final successor is never bootstrapped.

Features use the existing frozen VLM, BC LN/self-attention exactly once, and masked
pooling. Extraction batches never mix episodes/tasks, preserving the live RL
sampler's language/padding context. BF16 computation can have small batch-size
dependent numerical differences; FP32 storage adds no further quantization.
No DiT inference, Q/V update, reward, or Fourier time feature is cached.

The full and 50/task **BC checkpoints have different weights**, so use separate
caches. Each identity records exact source paths, config/metadata/encoder hashes
and weight-file size/mtime fingerprints (not full multi-GB weight hashes). Treat
source data/checkpoints as immutable while extracting/using caches. Changing BC,
normalization, images, language, encoder semantics or horizon requires a new cache.
The pooled cache cannot substitute for token conditioning when training a DiT.

## Paths and extraction

RAID roots:

- `/raid/yoon/vla_finetune/features/bigenlight-n17-all-bc10000-h16-v1`
- `/raid/yoon/vla_finetune/features/bigenlight-n17-50per-task-bc10000-h16-v1`

Feature payloads are approximately 0.84 GB / 0.68 GB, plus about 41 MB / 33 MB of
packed actions and small metadata. Episode `.npz` files are uncompressed for fast
reads. A temp file is flushed before exclusive atomic publication. Re-running the
same extractor skips finished episodes; `COMPLETE.json` is written only after every
episode is present. A partial cache cannot be used for training.

```bash
cd /home/yoon/vla_finetune/gr00t-bigenlight/n17
bash examples/bigenlight_multitask/cache_features.sh both            # dry run
bash examples/bigenlight_multitask/cache_features.sh both --execute  # explicit GPU work
```

The extractor holds a GPU-specific cache lock. A future live-IQL launcher waits for
that lock to release before its idle-GPU check, preventing the existing queued
50/task job from failing if extraction is still running. Running learners are not
affected by the lock. Container termination stops extraction, but saved episodes
remain resumable. No periodic LLM monitoring is needed; `progress.json` records
episode progress, and extractor logs are updated once per episode.

## Cached critic training

The cached trainer loads the roughly 1 GB cache into RAM, assembles current/next
features and masked actions, and only trains new Q1/Q2/V MLPs. It does not load a
VLM, tokenizer, images or DiT. The explicit reward preset is applied at training
time, allowing both 0/+1 and -1/0 experiments on exactly the same feature cache.

```bash
bash examples/bigenlight_multitask/train_cached_iql.sh all
IQL_REWARD=step-cost bash examples/bigenlight_multitask/train_cached_iql.sh all
# Add --execute only after cache completion and when that GPU is free.
```

Default training is batch 32, 10k updates, gamma .99, Adam LR 3e-4, expectile .7,
scalar Q-MSE / V-expectile, W&B online every 50 steps, periodic/recovery saving.
It preserves the episode-weighted sampling and 16-step return/terminal contract.
`IQL_RESUME` accepts checkpoints from this cached trainer with matching metadata;
live-VLM runs are not silently imported or rewritten. Neither these commands nor
creating caches changes the existing learner or its queued experiment settings.

For dedicated extraction after stopping the live learner, use
`CACHE_BATCH_SIZE=32 CACHE_CPU_THREADS=4 CACHE_SLEEP_MS=0` before the extraction
command. Saved episodes are reused; `progress.json` records the current batch
size and resumed episode count. The manifest's extraction batch size describes
the initial session; a resumed session can use a different batch size (with small
BF16 rounding differences).

`after_feature_cache.py` explicitly queues fresh cached all-data training followed
by fresh 50/task training, using each dataset's own BC cache. It waits without
polling for the extractor's GPU lock to release, requires both complete caches,
and verifies the all-data 10k checkpoint before launching 50/task. Specify
`--storage-root`, `--state-path`, `--all-output`, `--subset-output`, and
`--reward step-cost` (per-action -1, successful final action 0). It never resumes
the old critic. Queue state and separate training logs are written to RAID.
Like extraction, the queue must be restarted explicitly after container loss.
