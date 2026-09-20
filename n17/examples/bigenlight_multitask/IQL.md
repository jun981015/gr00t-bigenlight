# N1.7 frozen BC → environment critic-only IQL

This first experiment fixes **every BC parameter**, including the VLM, projector,
feature LN/self-attention and DiT. Only new scalar twin-Q and state V(s) MLPs train.
No actor objective, actor sampling, flow-time input, SVF inner critic, or environment
interaction is used. It does not yet implement HL-Gauss in the N1.7 IQL path.

Features: frozen VLM → frozen BC LN/self-attention (once) → masked mean pooling →
concatenate checkpoint-normalized state and embodiment one-hot → independent Q/V
MLPs. Q also takes the masked, normalized dataset action chunk. This follows the
DEAS-style shared VLM conditioning idea, **not an exact DEAS critic reproduction**:
these heads are independent LayerNorm/GELU MLPs. Raw VLM features are never overwritten
or sent through BC attention twice. Checkpoint normalization is reused unchanged.

The explicitly selected `all-success-terminal` annotation assumes all episodes
succeed: +1 and termination after the last recorded action, zero reward elsewhere.
These are user-approved labels, not recorded success timestamps. Source data is
unchanged; `reward_assumption.json` records per-episode terminal rows in the run.
There is no last successor observation; terminal bootstrap is zero and the existing
loader supplies a marked-invalid placeholder. No action chunk crosses an episode.

Default RL horizon = 16 actual actions (model padding is masked), gamma = 0.99 per
action. Targets are chunk return + gamma^16 × V(next state), zero bootstrap at
termination. V fits the 0.7 expectile of min(target-Q1, target-Q2) on dataset actions.
Q uses scalar MSE and target Q receives post-update EMA with tau=0.005. A chunk ending
at success has return 0.99^15; this is not a +1 reward at every action. Only one
chunk-start per episode ends at the final reward; sparse rewards may train slowly.
All-success demos alone do not establish failure/OOD action discrimination.

## Separate per-action -1 / goal 0 comparison

The optional `IQL_REWARD=step-cost` recipe selects `all-success-step-cost`: -1 for
every recorded environment action except the last successful action, whose reward
is 0 with terminal bootstrap disabled. Goal timing is still the explicitly approved
last-row assumption. The 16-step return is the discounted sum of these **per-action**
rewards, not one -1 per chunk. Existing runs/data and the default 0/+1 recipe remain
unchanged. Start fresh Q/V heads from the matching frozen BC, not the previous IQL
critic; strict resume metadata prevents mixing reward definitions.

```bash
# Dry run; add --execute only when the GPU is free / scheduled for this run.
IQL_REWARD=step-cost bash examples/bigenlight_multitask/run_iql_n17.sh all
IQL_REWARD=step-cost bash examples/bigenlight_multitask/run_iql_n17.sh 50per-task
```

All other settings (batch 32, H=16, gamma=.99, LR=3e-4, 10k updates, prefetch,
W&B online and recovery saving) stay the same. The default output names include
`step-cost` so these experiments cannot be mistaken for the original reward runs.

## Run inside the allocated container

One GPU, batch 32, LR 3e-4, 10,000 optimizer updates; W&B **online**, logs every 50
updates plus first/final. The `all` model/dataset is the default, not both variants
automatically. `50per-task` selects the matching BC checkpoint and dataset.

Metrics are computed on the current batch, before the optimizer update:
`q/{min,mean,max}` across both Q heads and the batch, `v/{min,mean,max}` across the
batch, `critic/loss` (Q MSE, alias of `loss/q`), `loss/v` (expectile loss), and
`critic/abs_td_loss` (mean absolute Bellman residual, diagnostic only). Gradient
L2 norms are **before clipping**: `critic/grad_norm` is Q-only, `value/grad_norm`
is IQL V-only, and `grad_norm` is their combined norm. Local metrics record every
update; W&B receives the sampled logging steps, not a 50-step moving average.

### Asynchronous input pipeline

The launcher now uses 8 spawned CPU workers, prefetch factor 2 (up to 16 queued
batches), 2 FFmpeg threads per decoder and one Torch CPU thread per worker.
This overlaps video decoding/processing with GPU updates, like N1.7 BC's background
loading, but retains the episode-safe RL sampler and its exact batch order rather
than substituting the BC shard sampler. Each worker retains at most 16 episodes
within a 4 GiB decoded-data cache (32 GiB total cache budget). Active decoding,
oversized episodes, processors and queued batches require additional memory.
Oversized episodes are reused within a batch but not retained across batches.

Loader/cache controls are operational and can change on checkpoint resume without
changing the RL metadata contract. `time/data_wait_s`, `time/update_s` and
`time/step_s` make the remaining bottleneck visible; first-step worker startup is
not representative of steady-state throughput. SIGTERM/SIGINT request a checkpoint
and stop after the current update in newly launched processes; this does not protect
against SIGKILL or container destruction and does not retrofit older processes.

```bash
cd /home/yoon/vla_finetune/gr00t-bigenlight/n17
# Print the command without loading the model or starting training:
bash examples/bigenlight_multitask/run_iql_n17.sh all
# First do a bounded real GPU smoke test, in its own output directory:
IQL_STEPS=10 bash examples/bigenlight_multitask/run_iql_n17.sh all --execute
# Then start the real run (new critic initialization/output):
bash examples/bigenlight_multitask/run_iql_n17.sh all --execute
```

No other GPU process is stopped by this launcher; it refuses an occupied GPU.
Use the site's normal persistent session mechanism if disconnecting SSH. Frozen
VLM current/next encodings still need substantial compute; GPU VRAM/speed have not
yet been measured for this path.

Output is on RAID, separate from BC. Recovery includes Q, V, target-Q, optimizer and
RNG at step 100, every 30 min, and each 5,000 steps. Once a newer complete state is
saved, only the previous owned optimizer checkpoint is removed; milestone weights
remain. BC weights are referenced by `--model-path`, not duplicated. At 8 hours the
trainer saves and exits; it does not renew a GPU allocation. A hard container kill
can lose progress since the latest recovery checkpoint.

Resume with the **same** variant, BC path and output, using the checkpoint named by
`checkpoints/latest_resumable.json`. If metrics contain updates beyond that checkpoint,
choose a new output directory (the trainer refuses ambiguous duplicate steps):

```bash
IQL_RESUME=/raid/yoon/vla_finetune/outputs/OLD_RUN/checkpoints/step-N.pt \
IQL_OUTPUT=/raid/yoon/vla_finetune/outputs/NEW_RESUMED_RUN \
IQL_STEPS=10000 bash examples/bigenlight_multitask/run_iql_n17.sh all --execute
```

`IQL_GPU` (default 0) and `IQL_BC_MODEL` override GPU/checkpoint. For a different BC
checkpoint you must ensure robot configuration, dataset and normalization match.
