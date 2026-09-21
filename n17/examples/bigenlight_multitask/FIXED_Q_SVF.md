# SVF with frozen cached-IQL Q (N1.7)

## DEAS distributional Q support

The same loader now accepts `deas-cached-critic-v1` full checkpoints. It restores
the learned DEAS projection and online Q1/Q2, freezes them together, and returns
each head's expectation over the HL-Gauss bin centers. No DEAS V, target Q, or
optimizer is imported. `min`/`mean` aggregation happens on decoded head values.
Both source discounts are recorded; frozen-Q SVF performs no reward/TD backup.
The CLI aliases `--fixed-q-checkpoint` and `--fixed-q-cache` accept either format.

`bash examples/bigenlight_multitask/train_deas_fixed_q_svf.sh --execute` runs
the all-data N1.7 BC with all-data DEAS lr=1e-4 step-30000, LoRA rank16/alpha32,
batch2, Adam lr=3e-4, 8 candidates and 10 flow steps. A ten-step startup checkpoint
is resumed to 10k updates. Env Q/reference/projections are frozen; DiT LoRA and
fresh inner V train. W&B online logs every50; scheduled saves every5k plus recovery.
The launcher supports concurrent user-owned cache extraction and does not stop it.

## Inner-only continuation

`--inner-only` freezes the actor in addition to reference and env Q. Only the
inner network belongs to Adam; no actor forward/backward or guidance gradient
is computed. The frozen reference DiT still generates SDE endpoint targets.
`--initialize-svf PATH` copies the modules from a matching fixed-Q SVF full
checkpoint, with a fresh inner-only optimizer and step counter. Source BC,
dataset, critic, normalization and hyperparameters must match. `--resume` then
restores an inner-only checkpoint normally; it cannot silently load a joint run.

The 20260921-from114 sweep starts independent 10k/30k inner-only runs from joint
step 114. The actor/reference/DEAS Q and learned inner weights are copied from
that snapshot. The 30k run is not a continuation of the 10k run. W&B logs include
inner predictions/targets (min/mean/max), inner MSE/absolute error and inner-only
gradient norm. The existing 8-hour recovery stop still applies; rerun the same
sweep after allocation recovery to continue from its latest saved state.

Optional [DiT LoRA with frozen projections](DIT_LORA.md) is now available.
The full-action-head path described below remains the default.

This is policy improvement against an immutable IQL Q, not joint online Q-learning.
It does not start, stop, or replace the running cache/IQL queue.

## Trainable and frozen paths

- Load **online Q1/Q2** from the cached-IQL full training checkpoint. Do not load
  IQL V(s), its optimizer, or its EMA Q as the scorer. Env Q has no gradients,
  optimizer entries, TD objective, next-action TD rollout, or EMA updates.
- Freeze the pretrained BC reference flow and VLM.
- Snapshot BC LN/self-attention before converting the trainable actor to FP32.
  Q and the fresh inner V receive its masked pooled feature + state + embodiment,
  with the same conditioning definition/dtype as cached IQL. This snapshot is
  independent of the actor's trainable projection/attention.
- Keep **raw VLM tokens** separate for actor/reference. Never overwrite them
  with the Q attention output or feed the same attention output through BC twice.
- Train the actor action head and a fresh time-conditioned inner V. Inner targets
  are `lambda * logmeanexp(Q(reference SDE endpoints) / lambda)`; actor targets
  are data CFM velocity plus detached inner-value action gradients. IQL V(s)
  is not this inner value and is not reused.

The scalar Q recipe uses `min(Q1,Q2)` for endpoint scoring (`--q-aggregation min`).
The existing mean option is also available. Inner Fourier time embedding remains
16, candidates 8, integration steps 10. No inner-only warmup schedule is added.

## Prepare / execute

```bash
cd /home/yoon/vla_finetune/gr00t-bigenlight/n17
bash examples/bigenlight_multitask/train_fixed_q_svf.sh all
bash examples/bigenlight_multitask/train_fixed_q_svf.sh 50per-task
```

These are **dry runs**. Add `--execute` only inside the allocated container once
the corresponding IQL `checkpoints/step-10000.pt` exists and the GPU is free.
Defaults refer to the Sept 20 cached-IQL step-cost runs: matching all/50-task
N1.7 BC checkpoints and caches, H=16, gamma=.99, per-action -1 / goal 0.
The GPU guard refuses an occupied training GPU. Nothing is automatically queued.

Overrides: `SVF_Q_CHECKPOINT`, `SVF_OUTPUT`, `SVF_BATCH_SIZE` (default **2**, not
IQL's 32), `SVF_STEPS` (default 10000), `SVF_GPU`, `SVF_RESUME`.
W&B online logs every 50 steps. Checkpoints every 5000, first recovery at 100,
then 30-minute recovery and 8-hour stop, retaining the newest full optimizer state.

The general CLI flags are `--algorithm svf --backend gr00t
--fixed-iql-checkpoint PATH --fixed-iql-cache CACHE_DIR`; specifying these forces
the BC reference frozen. Both flags are required. Import validates checkpoint
type, update count, completed cache hash, original BC/data/normalization identity,
feature dimension, full action padding mask, reward preset and gamma. Only the
current cached-IQL and cached-DEAS formats are supported; legacy live-IQL states are rejected.
Use trusted local `.pt` checkpoints only. Source checkpoint SHA256 and metadata
are recorded; SVF resume must use the same original BC, IQL source and settings.
The frozen Q-conditioning copy is reconstructed from the verified original BC on
resume; actor, reference, Q, inner/target weights, optimizer and RNG are saved.

## Limits and verification

This currently uses **live VLM token extraction**, not the pooled-only feature
cache as the actor input. The cache is used to verify IQL provenance; a DiT cannot
train from a pooled critic vector alone. No tokenizer/video-free SVF is claimed.
Full pretrained N1.7 GPU VRAM, throughput, convergence and deployment remain to be
measured. Two action heads and repeated SDE forwards cost more than cached IQL.
The actor remains FP32; frozen VLM/reference/Q conditioning retain BC BF16.

CPU tests check fixed Q/reference weights and optimizer exclusion, no Q EMA/TD
rollout, actor/inner updates, BC feature parity and actor independence, raw-token
immutability, real tiny GR00T head backward, cached-IQL import mismatch guards,
and deterministic SVF checkpoint continuation. Freezing Q does not correct its
errors on out-of-dataset actions; success-only IQL is not evidence of policy gains.
