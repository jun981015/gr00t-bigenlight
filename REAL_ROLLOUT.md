# Real-robot rollout handoff (2026-09-21)

This repository contains current N1.5/N1.7 BC and N1.7 offline-RL code.
It does not yet provide an end-to-end, hardware-validated UR7e controller.
No robot actuation or inference parity test was performed for this handoff.

## Checkpoint compatibility

| Artifact | Existing BC inference server | Required extra work |
| --- | --- | --- |
| Complete N1.7 BC checkpoint directory | Native `Gr00tPolicy` input | Observation replay and robot adapter validation |
| N1.5 BC checkpoint | Version-specific policy | Register Bigenlight data config and validate transforms |
| N1.7 SVF `step-*.pt` / `model-step-*.pt` | **Not directly supported** | Restore trained actor/LoRA and matching sampling policy in a dedicated inference adapter |
| Cached IQL/DEAS critic checkpoint | Not an action policy | Only needed if the selected inference algorithm uses the critic |

Do not rename an SVF `.pt` file to a BC checkpoint or silently serve the original
BC actor instead. SVF deployment must reconstruct the same actor architecture,
LoRA rank/alpha, normalization, action indices and inference-time sampling used
by the trained policy. Inspect `n17/gr00t/rl/projected_actor.py` and `frozen_q.py`.
There is currently no packaged SVF-to-BC merge/export loader. Whether the inner
critic is required depends on the chosen sampling path; actor-only rollout must
be explicitly checked against the training implementation, not assumed equivalent.

## N1.7 BC server, no robot commands

Use the N1.7 environment only (Python 3.12 and compatible CUDA dependencies).
On another machine install from `n17/pyproject.toml` / `uv.lock`; the provided
`environments/activate_n17.sh` selects this server's pre-existing RAID environment
and is not a portable installer. From the checkout, after activating the environment:

```bash
cd n17
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m gr00t.eval.run_gr00t_server \
  --model-path /absolute/path/to/full-bc-checkpoint \
  --embodiment-tag new_embodiment --device cuda \
  --host 127.0.0.1 --port 5555
```

Keep checkpoint processor configs, embodiment mapping and normalization statistics
alongside model weights. Use the matching 50/task or all-data BC artifact; do not
swap their statistics. Checkpoint weights are external artifacts, not Git content.

From the laptop, tunnel through the allocated container's SSH endpoint:

```bash
ssh -N -L 5555:127.0.0.1:5555 -p 23011 yoon@166.104.28.73
```

The endpoint/port can change with allocation. Use the N1.7 client protocol in
`n17/gr00t/policy/server_client.py` against localhost:5555; it is not a generic
HTTP endpoint. Do not expose this unauthenticated service publicly. Starting the
server alone does not send commands to a robot. Do not launch it on an occupied
training GPU without a separate resource decision.

## Robot-side contract to validate

- Scene/wrist RGB order, tensor shapes and task text must match the checkpoint's
  processor/modalities. Do not reuse an N1.5 client payload blindly for N1.7.
- Six joint coordinates in radians plus normalized gripper; actions are absolute
  joint targets, not Cartesian deltas. Confirm hardware joint ordering and
  gripper conversion against recorded data before actuation.
- Supervised action horizon is 16; N1.7 internal padding is 40 x 132, not 40
  executable robot actions. Use the processor-decoded real action fields.
- Data rate is 30 Hz; this is not a measured network/robot control rate. Select
  receding-horizon execution cadence using measured latency; do not blindly
  execute a full stale chunk or speed up actions to catch up.
- Require timestamp/sequence checks, finite outputs, joint/velocity limits,
  operator enable, hardware emergency stop and safe timeout/disconnect handling.
- First replay observations without actuation; then supervised low-speed trials.
  Offline losses and checkpoint availability are not evidence of safe rollout.

## Current research code map

- Cached IQL and held-out evaluation: `n17/gr00t/rl/train_cached.py`,
  `eval_cached_iql.py`, `prepare_critic_holdout.py`.
- DEAS-style cached critic: `n17/gr00t/rl/deas_cached.py`.
- Frozen IQL/DEAS Q, inner-only and joint actor training: `frozen_q.py`, `train.py`.
- Frozen projection, DiT LoRA and optional full-token cache: `projected_actor.py`,
  `actor_cache.py`, `cache_actor_features.py`, `train_cached_svf.py`.
- N1.5 episode-sharded extraction: `n15/experiments/bigenlight_multitask/cache_*.py`.

Pooled critic caches cannot replace actor token inputs. Training token caches also
cannot supply features for unseen live robot observations: live rollout still
needs VLM forward for new images. Resume/sweep scripts with dated RAID paths are
experiment provenance, not portable deployment entry points.
