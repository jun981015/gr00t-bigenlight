# LIBERO BC with DEAS GR00T N1.5

This recipe supports independently launched BC jobs and optional unsaved post-BC activity.
It uses DEAS's original `LiberoDataConfig` and `scripts/gr00t_finetune.py:main`.
It does not train a critic, collect rollouts, or implement SVF yet.

## Model, data, and batch

| Profile | Existing demonstrations | Training |
| --- | --- | --- |
| `long` | Same 50 episodes as N1.7 Long: 5/task, seed 42 | One GPU, effective batch 256, 5,000 optimizer steps |
| `unified` | Same 40 episodes as N1.7 unified: 1/task across all four suites | Another GPU, effective batch 256, 5,000 optimizer steps |

- Initialize each new run from NVIDIA `GR00T-N1.5-3B`, **not** N1.7 or a carrot checkpoint.
- Data: `/raid/yoon/vla_finetune/datasets/libero_n15_bc/`.
  These are isolated metadata overlays. Parquet/MP4 files are symlinks to the existing
  selections, including the official Goal wrist-video repair. No dataset download.
- Reuse N1.5 environment `/raid/yoon/vla_finetune/envs/deas-gr00t-n1.5`
  (Python 3.10 / torch 2.5.1); never install into N1.7's environment.
- Default batch: **micro-batch 32 × accumulation 8 = effective 256 per model**.
  This is not 256 simultaneous samples. `--micro-batch-size 256` disables accumulation;
  its VRAM fit is unverified. Each profile is an independent one-GPU run, not DDP across two models.
- Freeze language model and vision tower; train the original DEAS projector/DiT paths,
  including its VL normalization/self-attention. No critic head in this BC stage.
- Keep the original DEAS **BF16 mixed precision** for both profiles. N1.7 Long was FP32;
  this difference must be reported in comparisons. This is not an exact Q-VGM reproduction.
- Action supervision/output horizon: **16**, action dimension **7**. Dataset rate 20 Hz.
  N1.5's default LIBERO preprocessing converts the absolute 3D axis-angle state rotation
  to 6D rotation, while actions remain 7D controller commands. State input before padding
  is therefore 11D (position 3 + rotation 6 + two gripper joints).
- The names `eef_pos_delta`/`eef_rot_delta` describe the controller commands already
  present in the dataset; do **not** subtract observations from these actions again.
- N1.7's existing recipes supervise 16 actions but pad internally to 40. These are
  distinct from the evaluation `n_action_steps` (how many actions execute before replanning).

## Prepare and validate without GPU or network downloads

From the DEAS repository root:

```bash
bash experiments/libero_bc/run.sh prepare --profile long --validate
bash experiments/libero_bc/run.sh prepare --profile unified --validate
```

Validation uses the actual N1.5 loader and DEAS transforms on CPU, decoding one
episode per task. It checks `(16, 32)` padded action tensors, `(1, 64)` padded
states, 112 valid action-mask entries, finite values, and visual/language inputs.
Statistics remain limited to the selected trajectories; original data/metadata is untouched.

## Remaining prerequisites before GPU training

The environment and pinned model weights were installed on 2026-09-17. FlashAttention
is `2.7.1.post4+cu12torch2.5cxx11abifalse` for Python 3.10 / torch 2.5.1. The commands
below reproduce preparation when moving to another environment:

```bash
source experiments/robocasa_deas/activate.sh
# Model only. Do not run robocasa_deas/download.sh: that also downloads RoboCasa data.
python experiments/robocasa_deas/manage.py download model
# Pinned model revision in that recipe: 869830fc749c35f34771aa5209f923ac57e4564e

# In the allocated GPU container, with its CUDA toolkit:
bash experiments/robocasa_deas/bootstrap.sh --flash-attn
```

The launcher fails early if weights or FlashAttention are missing. A brief GPU
smoke test is still required before declaring the model/optimizer path validated.
GPU 0/1 selection uses UUIDs and the same per-GPU locks/guards as N1.7. Existing
training on the chosen GPU blocks launch; verified keepalive processes are preserved.

## Launch (dry run by default)

```bash
bash experiments/libero_bc/run.sh train --profile long --gpu 0 \
  --output /raid/yoon/vla_finetune/outputs/libero-n15-long-b256

bash experiments/libero_bc/run.sh train --profile unified --gpu 1 \
  --output /raid/yoon/vla_finetune/outputs/libero-n15-unified-b256
```

Add `--execute` to each command **only when its GPU is available**. There is no
automatic queue, polling, or background monitoring. W&B uses the current account's
`libero-gr00t-n15` project. Models are not uploaded to W&B.

## Saves and resume

- Single cosine LR schedule over 5,000 optimizer steps; LR `1e-4`, warmup 5%.
- Explicit snapshots at 500/1,000/2,000/3,000/5,000, plus regular 2,000-step saves,
  an early recovery save at 100, and a 30-minute wall-clock recovery save.
- Save full state and stop after 8 hours, leaving time before the 10-hour container limit.
- Retain prior model weights. Only after a new complete checkpoint commits, remove
  the older recipe-owned Adam/scheduler/RNG state. Resume uses the latest full state.
- Re-run the identical command with `--resume --execute`. GPU index may change, but
  batch, dataset, schedule, and world size must stay the same. W&B identity is retained.
- Do not use `--resume` to turn a completed 5,000-step run into additional training;
  that requires an explicit new experiment and LR schedule choice.

N1.5 checkpoints and optimizers are not interchangeable with N1.7 checkpoints.

The GPU-container smoke test exposed a GUI/headless OpenCV installation conflict;
the bootstrap now installs headless only. The PyAV reader uses one codec thread,
selects the nearest actual PTS rather than returning the preceding keyframe, and
caches at most 8 GiB of decoded RGB frames per DataLoader worker (two workers/job).
This cache is RAM-only; the downloaded videos are not modified or transcoded.

## Optional post-BC temporary run

Add `--after-steps 100000 --execute` to the BC command. After BC exits successfully
at its scheduled final step and its full checkpoint is verified, the same launcher
starts a separate `temporary-nosave-100000` run on that GPU. This is event-driven;
there is no polling or automatic resume after container shutdown.

The temporary phase loads final BC weights but starts a fresh optimizer/schedule.
It uses micro-batch 32 with accumulation 1 by default, **not BC's effective 256**.
It writes only small recipe/metadata/log files: periodic/final weights, Adam,
scheduler/RNG state, and W&B reporting are disabled. A hard save-method guard also
blocks accidental checkpoint calls. The final research BC checkpoint is untouched.
The temporary run lasts up to 100,000 steps or container termination; the server's
hard 10-hour lifetime cannot be bypassed. Existing GPU keepalive jobs stay running.

If BC stops early (8-hour budget or explicit stop), the temporary run does not start;
resume BC first. An unsaved-only short smoke test uses `--steps 2 --no-checkpoints`.

## 2026-09-17 launch

Both runs were launched in container port 23011 at 23:47:10 KST, with effective
batch 256 (32 × accumulation 8), 5,000 BC steps, and `--after-steps 100000`:

| GPU | tmux session / RAID output directory name |
| --- | --- |
| 0 | `libero-n15-long-b256-20260917T144710` |
| 1 | `libero-n15-unified-b256-20260917T144710` |

Outputs are under `/raid/yoon/vla_finetune/outputs/`. Each output has `train.log`,
`launch_recipe.json`, and W&B identity. After BC completion, `after_started.json`
records the separate unsaved run. Launch logs are under `/raid/yoon/vla_finetune/logs/`.
Short GPU tests verified finite training loss, full checkpoint commit, and automatic
transition to an unsaved run; no weight/optimizer files appeared in the temporary outputs.
13 LIBERO CPU tests and 19 existing checkpoint/recipe tests passed. No periodic monitor is running.
