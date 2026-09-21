# Action-reinjected IQL critic (q-vgm-critic branch)

This implements the requested design, not a claim of exact reproduction of a
Q-VGM paper. Legacy IQL, DEAS, SVF, BC training and robot actors are unchanged.

## Representation and actions

Reuse `FrozenBCGR00TEncoder` in `gr00t/rl/adapters.py`:

1. Frozen image/language backbone gives `[B,L,2048]` tokens.
2. Frozen BC action-head `vlln` + `vl_self_attention` conditions those tokens.
3. Attention-mask-weighted mean over tokens gives `[B,2048]`.
4. Concatenate normalized padded proprio `[B,132]` (7 physical values) and
   embodiment one-hot `[B,32]`, giving **`z_s: [B,2212]`**.

This is state-only, compact and already cached; it never runs the action DiT.
`FrozenGR00TStateExtractor` wraps the existing encoder for optional live use;
`FrozenStateFeatures` accepts a tensor or `{"features": tensor}` in cached use.
No second LN/self-attention pass is applied to cached features. The complete BC,
including its observation projection, is frozen. Proprio is concatenated raw
normalized/padded, NOT the BC's learned `state_encoder` output. Token-level
spatial detail is lost by pooling; this is a baseline tradeoff, not proof that
pooling is optimal. BC pooling follows the existing encoder exactly (the token
mask is applied at pooling); changing extraction requires a new cache.

Replay actions are **clean dataset commands**, in the BC processor's normalized
coordinates, not denoising samples. Here these are absolute joint+gripper
commands. Physical action `[B,16,7]` is processor-padded to `[B,40,132]`.
The Q module uses the cache's `action_indices` to gather **112** real coordinates,
preserving gradients and dropping padding. A new compact-only deployment can
construct Q with `(16,7)` and no indices, but to load an existing trained model
use its stored padded layout and mask. Input/action normalization must match BC.
`dQ/da` is in NORMALIZED coordinates; physical-unit guidance requires the chain
rule through the same normalization. Do not normalize again if inputs already are.

## Models and losses

- Ten independent Q heads by default, output **`[B,10]`**.
- Each head: `[z,a] -> 512 GELU -> [h,a] -> 512 GELU -> [h,a] -> 256 GELU
  -> [h,a] -> scalar`. No shared learned action/state encoder, no LayerNorm.
- Separate V: `2212 -> 512 GELU -> 512 GELU -> 256 GELU -> 1`.
- All Q/V operations FP32 even inside an outer autocast region; do not cast
  these modules to BF16. State detaches are intentional, action detaches absent.
- Q objective: `mean_B,E (Q_i(z,A) - [r_H + d*V_target(z_next)])**2`.
- V objective: expectile regression against **detached online mean Q**.
- One EMA **V target**, tau=0.005. No Q target is needed for this variant.
- Adam parameter groups: Q LR=1e-4, V LR=1e-4; each network gradient norm clipped
  to 10 separately. Default expectile=0.8, batch=64, gamma=0.99.

The cache samples a start t and clean H-command chunk, and returns s[t+H], not
s[t+1]. Thus `r_H=sum_i gamma**i*r[t+i]` and
`d=gamma**H*(1-terminated)`; **H=16 here**. The learner consumes the complete
`OfflineRLBatch.discounts` and does not discount again. Other replay adapters
can supply primitive H=1 transitions or already chunk-discounted transitions
under that same contract. There is intentionally no switch that changes only
gamma**H to gamma on this cache: doing so without changing reward/transition
semantics is inconsistent. The CLI gamma is per primitive action.

The current cache loader only supports explicit all-success reward presets;
`--assume-all-success` is required. Step cost is -1 per primitive command and 0
at the assumed last successful command. Success timing is not measured.
Full chunks only: the last H-1 possible starts are excluded; terminal bootstrap
is zero. Mixed success/failure data needs genuine reward/terminal annotations.

## Run

From repository root, using an unused output directory:

```bash
source environments/activate_n17.sh
python -m gr00t.rl.train_action_iql \
  --cache /raid/yoon/vla_finetune/features/bigenlight-n17-50per-task-bc10000-h16-v1 \
  --output-dir /raid/yoon/vla_finetune/outputs/action-iql-50per-e08-10heads \
  --assume-all-success --reward step-cost --gamma 0.99 \
  --num-q-heads 10 --hidden-dims 512 512 256 --batch-size 64 \
  --critic-lr 1e-4 --value-lr 1e-4 --expectile-tau 0.8 \
  --steps 10000 --save-every 5000 --log-every 50 \
  --wandb-project bigenlight-multitask-gr00t --device cuda:0
```

This command starts training; it is not launched automatically by this change.
Use `--debug-fixed-batch --steps 200 --device cpu` for a local overfit diagnostic.
That diagnostic is not a validation score. Without `--wandb-project`, logging is
local JSONL only; with it W&B is online. No actor or backbone is loaded at all.
Resume via `--resume .../checkpoints/step-N.pt` with identical semantic arguments.
Periodic model-only archives exclude optimizer; only the latest full recovery
state is retained. SIGTERM/SIGINT save through the existing OfflineTrainer.

## Metrics and inference

Every update logs Q/V loss, Q mean/min/max, ensemble std, across-state Q std,
V min/mean/max, TD target mean, reward mean, terminal fraction and parameter
gradient norms. `data/success_ratio=1` is explicitly marked ASSUMED for this
dataset; terminal fraction is not success ratio. At step 1 and every log interval,
diagnostics include mean per-example `||d mean_i Q_i / da||`, max absolute action
gradient, Q on dataset/perturbed actions, and mean absolute score difference.
Perturbation sigma defaults to 0.05 in normalized units; no clipping, no label
claim that perturbed actions are worse. A local RNG leaves training RNG untouched.
`q/mean_minus_beta_std` is metric-only and never affects losses or targets.
`q_statistics` uses population std (correction=0), so E=1 remains finite.

```python
import torch
from gr00t.rl.action_conditioned_iql import load_action_iql_models, q_statistics
q, v, metadata = load_action_iql_models("model-step-5000.pt", device="cuda:0")
# z_s and action must come from the SAME BC processor/features as metadata.
action = action.detach().clone().requires_grad_(True)
q_values = q(z_s, action)  # [B,E]; Q weights frozen, action graph alive
grad = torch.autograd.grad(q_values.mean(-1).sum(), action)[0]
stats = q_statistics(q_values)
```

Using a sum over B gives per-sample guidance without a 1/B scale. The requested
`q_values.mean()` also works, but scales gradients by batch size. Do not wrap
guidance computation in `torch.inference_mode()` or `no_grad()`.

Checkpoint backend is `action-conditioned-iql-cache-v1`. For evaluation use
`eval_action_iql`, not the legacy twin-Q evaluator. Fixed-Q SVF now supports
these model-only archives via `load_frozen_iql_q` and an explicit
`FrozenActionIQLQ` adapter: `[B,10] -> ensemble mean -> [1,B]`. This preserves the
mean-Q semantics used to train/validate this critic, independently of the SVF
`q_aggregation` setting for the inner loss (min by default). All ten Q heads are
retained and frozen; no V, EMA V or IQL optimizer is imported. Provenance records
the source head count, aggregation and proprio selection. Action gradients stay
enabled. Existing legacy IQL/DEAS loading and aggregation are unchanged.

The proprio-included 150k experiment uses the fixed BC50 pooled/projected caches
without image/state/action augmentation. SVF CFM/SDE noise remains part of the
algorithm, not data augmentation. Actor LoRA and two inner heads are trained;
actor guidance uses the mean inner-value gradient, as before.

## Implementation verification

On the actual N1.7 50/task cache, the default architecture (10 heads, 512/512/256,
batch 64, Q/V LR=1e-4, expectile=0.8) was tested on one fixed batch for 200 CPU
updates. Results are in
`/raid/yoon/vla_finetune/outputs/action-iql-qvgm-50per-debug200/`.

| Metric | Step 1 | Step 50 | Step 200 |
|---|---:|---:|---:|
| Q MSE | 220.5911 | 9.7620 | 0.01963 |
| V expectile loss | 0.000489 | 0.4360 | 0.00675 |
| Mean per-example action gradient norm | 0.1093 | 0.7452 | 0.7362 |
| Perturbed-action absolute Q change | 0.00445 | 0.02990 | 0.02849 |

V loss is not monotonic: its target changes as Q learns. This fixed batch had
no terminal chunks, so this run only proves optimization/action sensitivity,
not long-horizon reward propagation or validation quality. A separate unit
test overfits terminal samples to known state/action-dependent rewards. Tests
also cover finite-difference agreement of action gradients, frozen Q with live
action gradients, padding invariance, independent heads, backbone immutability,
exact target/expectile formulas, EMA V, model-only loading, full resume, and
primitive-reward chunk discount semantics. All 125 tests in `tests/gr00t/rl/`
passed at implementation time. Q has 16,707,690 parameters; V has 1,527,297.
The debug checkpoint is not a trained policy or a production critic.

## 5k/10k heldout comparison

`train_action_iql_50per_with_validation.sh FRESH_OUTPUT_DIR` runs the 10k
training configuration above, then evaluates both saved 5k and 10k model-only
archives on the existing 56-episode holdout cache (BC50 conditioning).
Training and validation use W&B online. Per-checkpoint reports, predictions,
and `validation-comparison.json` are saved inside the run directory.

The evaluator is `python -m gr00t.rl.eval_action_iql`; it validates the same BC,
cache provenance and disjoint episode split. Unlike the legacy evaluator it
uses **ensemble mean**, online Q for the V residual, and **EMA V** for TD targets.
`transition_weighted.q_std` is the across-state std of the mean Q;
`ensemble_std` is the mean within-state dispersion across heads. Terminal and
nonterminal diagnostics are reported separately. Action perturbations use the
same seeded samples at both checkpoints. Comparing to legacy min-Q evaluation
requires explicitly accounting for the different aggregation/target rules.

## Explicit proprio ablation

`--exclude-proprio` removes state coordinates `[2048:2180]` from the existing
2212-wide N1.7 cache inside BOTH Q and V (including the EMA V). The external
feature input stays `[B,2212]` for cache/rollout compatibility; the MLP sees
`[B,2080]` = pooled VLM 2048 + embodiment 32. No recaching, no actor modification.
The model loader restores this choice from checkpoint metadata; validation uses
the same choice automatically. Absent/false preserves old checkpoints exactly.
Use a fresh run, not a resume from a with-proprio model (first-layer shapes differ).
The action still contains absolute joint commands and images still show robot
pose: this removes explicit proprio, not every correlated source of information.
First-layer parameter counts and random initialization change slightly as input
size changes; this is not an identical-initialization paired experiment.
