# Offline RL with GR00T datasets (local extension)

For the four DEAS RoboCasa tasks with **N1.7 + SVF**, see the local
[RoboCasa SVF recipe](../examples/robocasa_svf/README.md). The CLI now accepts
multiple paths after `--dataset-path` and an explicit `--annotation-format deas-robocasa`.
That preset distinguishes successful terminals from failed time limits; its recipe
explicitly disables timeout bootstrapping because post-timeout observations are unavailable.

This extension adds a separate offline RL path without changing the existing
supervised `DatasetFactory`, `ShardedSingleStepDataset`, or GR00T model forward.
It reuses `LeRobotEpisodeLoader`, `extract_step_data`, `VLAStepData`, and the
GR00T state/action and multimodal processors. It does not require JAX.

Reference: local `fmrl` **origin/dh at
`7304b615e6f1bbb07d462ea24bf008e8ac019a17`**, specifically
`docs/SVF_IMPLEMENTATION.md` and `qflow_svf/agents/qflow_rc.py`.
That branch is read with `git show`; the fmrl working tree/branch is not changed.
This is a PyTorch implementation of the reference's **default SVF equations**,
not a drop-in replacement for its JAX agents or a claim of benchmark parity.

추가 참조: **DEAS (ICLR 2026, Kimin Lee 공저)**의 GR00T N1.5 구현도 검토했습니다.
아래 DEAS 절에 공통 설계에 반영한 부분과 미구현 부분을 구분했습니다.
현재 학습 알고리즘은 BC/SVF이며, DEAS 재현 구현이라고 부르지 않습니다.

## Architecture and extension points

```text
LeRobotEpisodeLoader + explicit RL annotation provider
    -> LeRobotOfflineRLDataset -> RLTransition(VLAStepData, next VLAStepData, labels)
    -> state-only collator OR GR00T processor/collator + FrozenGR00TEncoder
    -> OfflineRLBatch
    -> OfflineTrainer -> any OfflineAlgorithm.update(batch)
                         |- FlowBC (baseline)
                         `- SoftValueFlow (SVF)
```

| Module | Responsibility |
|---|---|
| `gr00t/rl/types.py` | Algorithm-independent transition/batch contracts and learner protocol |
| `gr00t/rl/dataset.py` | Explicit annotations, episode-local next observations, fixed action chunks, bounded episode cache |
| `gr00t/rl/adapters.py` | Existing GR00T normalization/collation, frozen VLM encoding, differentiable head velocity |
| `gr00t/rl/networks.py` | Replaceable feature MLP actor and ensemble outer/inner critics |
| `gr00t/rl/algorithms.py` | BC baseline and SVF, independent of LeRobot and model internals |
| `gr00t/rl/selection.py` | Optional best-of-N / softmax selection, separate from training |
| `gr00t/rl/trainer.py` | Updates, JSONL metrics, full learner/optimizer/RNG checkpoint and resume |
| `gr00t/rl/train.py` | Single-device CLI |

For another offline algorithm, implement `update`, `state_dict`, and
`load_state_dict`, then pass it to `OfflineTrainer`. No dataset changes are
needed. A different actor, token-aware critic, or frozen visual representation
can replace the supplied modules. Use `torch.utils.data.ConcatDataset` and a
custom sampler for multiple datasets; match normalization/action schemas or
provide a collator that handles their differences explicitly.

`StateActionTransitionCollator` deliberately ignores images and language. Its
MLP runs on CPU and is useful for testing and low-dimensional baselines; it is
**not** evidence that pretrained GR00T was trained. The `gr00t` backend uses the
actual pretrained action head and a **frozen vision tower + LLM**.

## Required data: do not invent rewards from demonstrations

Standard GR00T demonstrations do not necessarily contain RL labels. The earlier
synthetic SO100 VRAM dataset does not become an RL dataset automatically.

Each row is interpreted as:

```text
observation[t], action[t], reward_after_action[t], terminated[t], truncated[t]
```

Supply the actual parquet columns explicitly:

```python
labels = ColumnRLAnnotations(
    reward_column="next.reward",
    terminated_column="next.terminated",
    truncated_column="next.truncated",
    last_row_is_observation=False,
)
```

Column names above are examples, **not assumed LeRobot defaults**. A flag column
may be `None` only if that flag is known to be absent/false. A combined `done`
column is not automatically a true terminal: split timeouts from task terminals
before mapping it. Missing rewards, nonfinite values, inconsistent lengths,
overlapping terminal/timeout labels, or transitions after an episode boundary
fail explicitly. Never modify the source dataset to silently append fake zeros.

For relabeling, success-based rewards, human annotations, or sidecars, provide
a callable `(episode_loader, episode_index) -> EpisodeRLAnnotations` instead:

```python
def my_annotations(loader, index):
    episode_id = loader.episodes_metadata[index]["episode_index"]
    labels = read_my_label_store(episode_id)  # User-supplied implementation.
    return EpisodeRLAnnotations(
        rewards=labels.rewards,
        terminated=labels.terminated,
        truncated=labels.truncated,
        final_observation=labels.final_observation,  # VLAStepData or None.
    )
```

`final_observation` must carry all robot observation modalities/history expected
by the processor, in the same embodiment and normalization input coordinates.
Its actions are ignored. The callback API supports sidecars; the CLI exposes
the simpler column-mapping provider.

### Final observations and time limits

- If the final row is **observation only**, use `last_row_is_observation=True`.
  N rows then describe N-1 transitions; the final row's action/reward/flags are ignored.
- A true terminal has bootstrap multiplier zero. If its post-action observation
  is missing, the last recorded observation is used only as a shape placeholder;
  `RLTransition.next_observation_valid=False` records this explicitly.
- A truncation bootstraps **by default** and therefore needs a real final next
  observation. Supply an observation-only row or the annotation callback's
  `final_observation`. Do not use the next episode's reset observation.
- `--no-bootstrap-on-truncation` is an explicit alternate convention, not an
  automatic fallback. It permits a missing final observation because its
  bootstrap multiplier is zero.
- Continuing dataset fragments may bootstrap from a genuine recorded successor.
  No episode is considered successful merely because its recording ended.

### Action chunks and discount contract

For `horizon=H`, one RL action is a **complete H-step action chunk**:

```text
actions             = [a_t, ..., a_(t+H-1)]
rewards              = sum_i gamma**i * r_(t+i)
next_observations    = o_(t+H)
discounts            = gamma**H * bootstrap_mask
TD target            = rewards + discounts * Q_target(next_observations, next_actions)
```

By default, reward and bootstrap use the same gamma. For algorithms requiring
separate intra-/inter-chunk discounts, set `bootstrap_gamma` on the dataset (CLI
`--bootstrap-gamma`): rewards still use `gamma`, but `discounts` becomes
`bootstrap_gamma**H * bootstrap_mask`. Each `RLTransition.reward_gamma` records
the within-chunk setting, checked against the collator even for terminal samples.
Default SVF is unchanged; separate gammas define a different objective.

Do not multiply `discounts` by gamma or a mask again. `H=1` recovers the
one-step transition used in `qflow_svf`. Its original expression
`gamma * batch['masks']` corresponds to our complete `batch.discounts`.
Chunk rewards are not un-discounted sums or repeated copies of the last reward.

Incomplete action chunks are dropped rather than filled with repeated terminal
actions. Episodes shorter than H have no samples. Negative observation-history
offsets repeat only the episode's first frame; positive/future observation
offsets are rejected. Current and next image/state histories come from the same
episode; next observations contain **no next-action labels**.

The full GR00T processor pads actions to the checkpoint's `[H_max, A_max]`.
`action_mask` excludes inactive dimensions and timesteps from actor losses,
critics, reference SDE, and policy sampling. Padded coordinates stay zero,
including noise. This differs from vanilla GR00T's unconstrained noise on padded
coordinates, so downstream sampling must use the same RL mask convention.

`H` here is the executed/critic horizon, not the model's padded output length.
If a controller replans after 8 of 16 predicted actions, use transitions ending
at t+8 and train/score those 8 actions. Do not use t+16 or gamma**16 for an 8-step
execution. This adapter masks the unused suffix; it does not claim equivalence
to sampling a full unmasked 16-step chunk and truncating it afterward.

## SVF implementation mapping

For policy improvement against the Q learned by our cached IQL runs, see
[frozen-IQL-Q SVF](../examples/bigenlight_multitask/FIXED_Q_SVF.md). This opt-in mode
freezes env Q and the BC reference, trains only inner V and the actor, and keeps
IQL's frozen BC conditioning separate from raw actor tokens. Default SVF below
is unchanged.

The table follows the function locations in the user's SVF document:

| Reference `qflow_rc.py` | Torch implementation |
|---|---|
| `_base_sde_endpoints`, around line 78 | `SoftValueFlow.base_sde_endpoints` |
| `_soft_value_target`, around line 159 | `soft_value` and the inner loss in `losses` |
| `critic_loss`, around line 204 | Outer TD + inner soft-value regression in `losses` |
| `actor_loss`, around line 504 | Data CFM target + detached inner-value gradient in `losses` |
| `total_loss`, around line 800 | Independent temperature estimate and sum of four objectives |
| `update`, around line 885 | One Adam step in `update` |
| `sample_actions`, around line 899 | Masked Euler actor ODE in `sample_actions` |

Implemented defaults:

- Outer Q ensemble 2, mean/min aggregation, TD bootstrapping from the actor ODE.
- Inner V ensemble 2, Fourier time embedding 16, **aggregate heads before MSE**.
- Separate reference BC flow; K=8 Euler–Maruyama candidates, kappa constant,
  `dt=min(1-t, 1/flow_steps)`, flow_steps=10, t_min=0.1.
- `lambda=lambda_multiplier*max(mean_state(population_std_candidate(Q)), 1e-3)`.
  The temperature and inner target use **independent random draws**. An explicit
  positive `SVFConfig.soft_lambda` is also supported.
- Actor target is **data velocity** `(action-noise) + beta(t)*stopgrad(grad_x V)`;
  beta=kappa²(1-t)/(t*lambda) above t_min and zero below it. Guidance differentiates
  the mean inner-critic heads, including when Q uses min, matching the reference.
- Actor times cover [0,1); inner/SDE anchors cover [t_min,1).
- Reference BC uses pure CFM and is optionally frozen with `--freeze-reference`.
  This flag assumes the reference is already trained; it does **not** run the
  reference repository's 250-epoch BC-pretraining/cache workflow.
- Target critics are frozen. EMA uses **pre-optimizer online parameters**, matching
  the dh code's `self.network` lookup (which lags the freshly updated network by
  one step). Both target networks, all four modules, and Adam state are checkpointed.

Not ported: FAC density penalty, flow-time TD inner targets, per-state temperature,
split/each/aggout distillation, adaptive kappa, guided deployment, or online replay.
Unsupported variants are not silently approximated by the CLI. The small MLPs
use GELU/LayerNorm and are reference scaffolding, not a numerical JAX weight port.

### GR00T-specific model integration

`FrozenGR00TEncoder` runs the VLM **once for current and once for next** observations
per batch. All actor/reference ODE/SDE calls reuse those raw, detached token features.
The trainable VL projection/self-attention, state encoder, action encoder, DiT,
and action decoder remain inside each independent actor/reference head.

`Gr00tFlowActor` exposes `v(obs, noisy_action, time)`. It does not call the original
`get_action()` for training because that path is `no_grad`, and the original
training `forward()` chooses its own noise/time and reduces to a CFM loss.
The adapter mirrors the deterministic velocity computation without state dropout.

The supplied critics receive masked mean raw VLM features, normalized state,
and a one-hot embodiment ID. A token-aware critic is an independent extension.
Trainable GR00T head weights and Adam are FP32 in the CLI; the VLM stays frozen
BF16. No claim is made that the previous BC VRAM numbers apply to SVF: it has a
second action head and many reference rollouts. Start with a very small batch.

## DEAS reference and reusable extension points

[DEAS (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/68b7f71467c83d5d32275c5acbf6b588-Abstract-Conference.html)
is by Changyeon Kim, Haeone Lee, Younggyo Seo, Kimin Lee, and Yuke Zhu.
Its [method and VLA appendix](https://arxiv.org/html/2510.07730v1#S4)
describe actor-independent value learning over action sequences: a state value
V(s), an action-chunk Q(s,a), asymmetric value learning, distributional losses,
and separate within-chunk and bootstrap discounts. This is **not SVF's**
time-conditioned V(s,x,t), soft-value target, or guided CFM objective.

The [author's GR00T N1.5 implementation](https://github.com/csmile-1006/DEAS-Isaac-GR00T)
was inspected at commit `1bfc8464c4d28574cbd2e699342791c7e4d80028`:

- [deas_critic.py](https://github.com/csmile-1006/DEAS-Isaac-GR00T/blob/1bfc8464c4d28574cbd2e699342791c7e4d80028/gr00t/model/action_head/deas_critic.py)
  separates critic/value modules and uses VLM representations plus robot state.
- [deas_action_head_bon.py](https://github.com/csmile-1006/DEAS-Isaac-GR00T/blob/1bfc8464c4d28574cbd2e699342791c7e4d80028/gr00t/model/action_head/deas_action_head_bon.py)
  samples flow-action candidates, scores their critic-horizon prefixes, and
  selects via argmax or temperature-controlled categorical sampling.

Reflected in this N1.7 extension:

1. `OfflineTrainer` accepts **critic-only learners**, not just actor-critic pairs.
   A future DEAS learner can consume the same `OfflineRLBatch` without an actor
   in its TD target. No dependency on SVF's inner critic is required.
2. `--gamma` / `--bootstrap-gamma` allow independent discounts; the default remains
   one gamma for SVF. For example, `--gamma 0.9 --bootstrap-gamma 0.99` changes only
   transition construction, not the learner class into DEAS.
3. `select_action_candidates` is a separate policy-extraction utility:

```python
from gr00t.rl.selection import select_action_candidates

choice = select_action_candidates(
    learner.sample_actions, trained_critic, batch.observations, batch.action_mask,
    num_candidates=10, aggregation="min", temperature=0.0,
)
normalized_actions = choice.actions
```

`temperature=0` is greedy; positive temperature uses softmax sampling. The critic
must output expected scalar Q values `[ensemble,B]`, not distributional logits.
Candidates reuse already encoded observations, with sequential evaluation to bound
VRAM. This utility neither retrains the VLA nor changes SVF's default sampling.
It is not wired to the robot server and does not guarantee a performance gain.

**Not implemented:** the DEAS distributional Q/V learner, HL-Gaussian targets,
expectile-weighted classification, or N1.5 checkpoint conversion. No DEAS
experiment or performance reproduction is claimed. No old dependencies or
datasets were installed into the active GR00T N1.7 environment.

## Running

From this checkout, activate the existing environment:

```bash
source /home/yoon/vla_finetune/activate_gr00t.sh
cd /home/yoon/vla_finetune/gr00t-bigenlight/n17
python -m gr00t.rl.train --help
```

CPU/state baseline on **your RL-labeled dataset** (replace the example paths and
column names; use the observation-only flag only if your data has that layout):

```bash
python -m gr00t.rl.train \
  --backend state --algorithm svf --device cpu \
  --dataset-path /raid/yoon/vla_finetune/datasets/robot_rl \
  --modality-config-path examples/SO100/so100_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --reward-column next.reward \
  --terminated-column next.terminated --truncated-column next.truncated \
  --horizon 1 --batch-size 8 --steps 10 \
  --output-dir /raid/yoon/vla_finetune/outputs/rl-state-smoke
```

Actual GR00T (inside an allocated GPU container):

```bash
python -m gr00t.rl.train \
  --backend gr00t --algorithm svf --device cuda:0 \
  --model-path /raid/yoon/vla_finetune/models/robot_bc_checkpoint \
  --dataset-path /raid/yoon/vla_finetune/datasets/robot_rl \
  --embodiment-tag NEW_EMBODIMENT \
  --reward-column next.reward \
  --terminated-column next.terminated --truncated-column next.truncated \
  --horizon 8 --batch-size 1 --steps 10 \
  --output-dir /raid/yoon/vla_finetune/outputs/rl-gr00t-smoke
```

Use a checkpoint whose embodiment/modality configs and normalization already
match the robot. `NEW_EMBODIMENT` normally requires a robot-specific BC checkpoint;
the previous fake-data memory tests did not save one. Base checkpoints can be used
for a matching pretrained embodiment. This CLI does not silently register a new
robot against base-model statistics. GR00T RL H must not exceed the robot's
trained horizon. State-only mode uses dataset statistics and offers an explicit
`--no-relative-actions`; full GR00T always uses its checkpoint normalization.

Change `--algorithm svf` to `--algorithm bc` for the same pipeline without RL
guidance. The common dataset still requires labels, keeping the batch contract
identical; ordinary unlabeled BC remains available through the original launcher.

## Sampling, outputs, and resume

- The default sampler selects episodes proportional to valid transition count,
  then samples a batch within one episode with replacement. This bounds video
  decoding/cache churn but introduces within-batch correlation. It is a starting
  sampler, not an assertion of optimal large-scale mixing. Replace it independently.
- The loader caches **one full decoded episode** by default, not the whole dataset.
  Long/high-resolution videos can still be large. GR00T's existing per-episode
  decoder is reused; a streaming/sharded RL loader is future work.
- CLI is **single process, single device**, num_workers=0. DDP/ZeRO and large-GPU
  memory/throughput testing are not implemented or claimed in this extension.
- `metrics.jsonl` contains every update; `run.json` records the configuration,
  normalization, annotation hash, and reference commit. `processor/` is saved for
  GR00T runs. Checkpoints go to `checkpoints/step-N.pt` and include all optimizer
  and RNG state. By default only the final checkpoint is written; use `--save-every`
  explicitly for large runs. Existing checkpoints are never overwritten/deleted.
- Resume with the same arguments plus `--resume .../checkpoints/step-N.pt` and a
  larger **total** `--steps`. Changed data labels, gamma, normalization, or semantic
  training arguments fail validation. Existing output metadata must also match;
  if its metrics extend past the restored step, use a new output directory to
  fork the run without duplicate steps. Use only trusted local `.pt` files.
- These are **RL learner checkpoints, not vanilla `Gr00tPolicy` checkpoints**.
  Serving/export integration must preserve actor sampling steps, clipping, action
  mask, execution horizon and normalization. Do not hand them directly to the
  standard policy server or actuate a robot without separate deployment validation.

## Validation

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NO_ALBUMENTATIONS_UPDATE=1 \
python -m pytest tests/gr00t/rl -q --timeout=60
```

CPU tests cover actual LeRobot parquet loading, final observations/time limits,
chunk returns/dual discounts, reward errors, bounded caching and deterministic sampling,
SVF formula limits, independent temperature samples, gradient isolation, frozen
reference behavior, padding invariance, checkpoint continuation, and actual tiny
GR00T DiT/AlternateVLDiT forward parity and SVF updates with a stub frozen VLM.
The CLI is exercised against a generated labeled LeRobot fixture, including resume.
Additional tests exercise the real N1.7 processor's short-chunk normalization and
padding (only the external tokenizer is stubbed), critic-only learner injection,
and greedy/softmax candidate selection with masks and gradient isolation.
These are correctness tests, **not** pretrained 3B GPU convergence/VRAM results.
