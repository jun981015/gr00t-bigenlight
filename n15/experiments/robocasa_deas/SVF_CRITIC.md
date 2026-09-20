# GR00T N1.5 + SVF critics, optional IQL Q-learning

Implemented against `fmrl/dh` at `710cde1` (2026-09-20), specifically
`docs/SVF_IMPLEMENTATION.md` and `qflow_svf/agents/qflow_rc.py`.
This is an opt-in model composition and critic-loss API, **not a complete SVF
training recipe**. Existing BC, DEAS distributional critic and IQL entry points
are unchanged. No running BC process is restarted or modified.

## Architecture

```text
frozen BC VLM -> raw token features
    |-> existing BC action head / DiT (preserved, frozen in this wrapper)
    `-> copied, frozen BC LayerNorm + self-attention -> mean pool
            |-> Q-specific embodiment MLP + tanh -> concat(state, action chunk)
            |                                      -> outer Q ensemble [2,B]
            `-> V-specific embodiment MLP + tanh -> concat(state, x_t, Fourier(t))
                                                   -> inner V ensemble [2,B]
```

The attention module's **output hidden states**, not its internal Q/K/V tensors,
feed the critics. No second cross-attention over actions is introduced. As in
DEAS, actions enter after pooling, concatenated with state and projected features.
Both branches see one computation of the same frozen observation features.
The raw `backbone_features` dictionary is never overwritten; never pass features
already processed/mutated by the legacy actor's `process_backbone_output`.
Pooling matches DEAS's unmasked token mean; the copied attention has no mask API.

By default there are two critic *families*, each with two scalar ensemble members, not two
scalar outputs total. Each family also has its own frozen EMA copy, including its
trainable feature projection. The projections are independent across Q and V,
unlike the single shared bottleneck in the original DEAS Q/V critic. Defaults:

With `outer_update='iql'`, a third, independent `iql_value(s)` branch is
added after the same pooling. It is **not** the SVF `tc_critic(s,x_t,t)`. With
`q_loss_type='hl_gauss'`, each Q member outputs categorical logits instead of a
scalar; public `q_values()` still returns the decoded scalar expectation `[E,B]`.
IQL V is scalar by default, or categorical with `v_loss_type='deas'`; its public
`state_values()` always returns scalar values `[B]` (expectations when categorical).

- Embodiment projection: DEAS `CategorySpecificMLP`, hidden 1024 x 3, feature 64,
  SiLU between hidden layers and tanh on the feature output.
- Each scalar member: MLP 512 x 4, GELU then LayerNorm after each hidden layer.
- Time: 16D fixed Fourier features, frequencies exp(linspace(0, log(256), 8)),
  concatenate sin then cos, no extra 2*pi. SVF convention: t=0 noise, t=1 action.
- Outer Q, inner distillation and guidance aggregation: independently configurable
  `mean`/`min`, all default `mean`. For reference `q_agg=min`, set Q and inner to
  `min`; reference default actor guidance remains `mean`.
- EMA tau 0.005, applied **after** each optimizer step. Never after every
  accumulation microbatch. Both target networks are checkpointed.
- N1.5 checkpoint supplies feature/state/action/embodiment dimensions and horizon;
  no hard-coded assumption that the padded action dimension equals robot DoF.

## Interfaces

`gr00t/model/action_head/svf_critic.py` contains the reusable feature-input head.
`gr00t/model/gr00t_n1_svf.py` connects it to a real N1.5 BC model and its input
preparation. N1.7 is **not wired up**: it needs its own backbone/feature adapter,
not simply loading a N1.7 checkpoint through this wrapper.

```python
from gr00t.model.gr00t_n1_svf import GR00TN15SVF
from gr00t.model.action_head.svf_critic import (
    aggregate, outer_td_target, soft_value_target,
)
import torch

model = GR00TN15SVF.from_bc_checkpoint(
    BC_CHECKPOINT_PATH, time_embed_dim=16,  # choose 16 (default) or 64
).to("cuda")
model.train()  # only online critic modules enter training mode
head = model.critic_head
optimizer = torch.optim.Adam(
    (p for p in head.parameters() if p.requires_grad), lr=3e-4,
)
```

Alternatively `GR00TN15SVF(existing_bc_model, critic_config)` takes ownership of
that model and freezes it; it does not clone the full VLM. New critics remain
float32 by default; frozen copies preserve BC dtype. The trainer may use BF16
autocast. Do not call `model.bfloat16()` if float32 critic master weights are wanted.

Fourier width can also be set with `SVFCriticConfig(time_embed_dim=64)` or
`GR00TN15SVF(existing_bc_model, time_embed_dim=64)`. An explicit keyword overrides
the config without mutating it; omitting it preserves the config (default 16).
16D uses 8 frequencies; 64D uses 32, both spanning [1,256], with sin/cos pairs.
Only the inner critic input width changes, not the outer Q or DiT time embedding.
The width is saved in critic checkpoints; a 16D critic cannot resume as 64D
(strict loading rejects this). Choose the width before starting critic training.

Inputs below are **already transformed GR00T training inputs**, with normalized
state/actions, correct embodiment IDs, and original BC normalization statistics.
`actions` and `noisy_actions` are `[B,H,A_padded]`, time is `[B]` or `[B,1]`.
Pass the binary `[B,H,A_padded]` action mask for padded dimensions/invalid chunk
positions to **all** Q, V, target and gradient evaluations; masked gradients are
zero. The wrapper defaults to `inputs['action_mask']` when supplied. A missing
mask means every action coordinate is real/valid, not automatic mask inference.

The following illustrates a critic update with externally prepared transitions
and conditional endpoints; it is not a runnable dataset/sampler implementation:

```python
obs = model.encode_observation(current_inputs)       # VLM + attention once
next_obs = model.encode_observation(next_inputs)     # VLM + attention once
with torch.no_grad():
    next_q = aggregate(head.q_values(next_obs, next_actions, target=True,
                                    action_mask=next_action_mask),
                       head.config.q_aggregation)
    q_target = outer_td_target(chunk_returns, bootstrap_discounts, next_q)
    # endpoints: [K,B,H,A], conditional BC-SDE continuations from (obs,x_t,t).
    # Reuse cached obs: no K-fold VLM/self-attention recomputation.
    endpoint_q = torch.stack([
        aggregate(head.q_values(obs, endpoint, target=True, action_mask=action_mask),
                  head.config.q_aggregation)
        for endpoint in endpoints
    ])
    inner_target = soft_value_target(endpoint_q, temperature)

optimizer.zero_grad(set_to_none=True)
predictions = head(obs, actions, noisy_actions, times, action_mask=action_mask)
losses = head.loss(predictions, q_target, inner_target)
losses['loss'].backward()
optimizer.step()
head.update_targets()
gradient = head.guidance_gradient(obs, noisy_actions, times, action_mask=action_mask)
```

In default MSE mode, `loss()` implements mean_{ensemble,batch}(Q-y)^2 and
mean_batch(aggregate(V)-soft_target)^2 (`sv_head_distill='agg'`). Targets detach,
loss arithmetic is float32. `guidance_gradient()` returns detached dV/dx_t without
accumulating parameter gradients, even inside `torch.no_grad()` (not inference
mode). `soft_values()` itself preserves action gradients, including through a
frozen target, for alternate actor objectives/higher-order derivatives.

`outer_td_target` deliberately takes `[B]` discounted chunk return and `[B]`
bootstrap discount (gamma**H times continuation), not raw reward sequences.
fmrl `sample_sequence` rewards are already cumulative; select the appropriate
final valid return, do not sum them again. Align next state, executed horizon,
terminal and time-limit truncation handling in the dataset adapter. No automatic
reward shift, clipping, reward creation, or BC success labeling is performed.

## IQL target builder and selectable Q loss

These are independent choices, including `td + hl_gauss` and `iql + mse`:

| Setting | Options | Default |
| --- | --- | --- |
| `outer_update` | `td` (caller-provided target), `iql` (transition-based target) | `td` |
| `q_loss_type` | `mse`, `hl_gauss` | `mse` |
| `v_loss_type` | `mse` (scalar expectile), `deas` (distributional weighted CE) | `mse` |
| `iql_expectile` | scalar in (0,1), through `SVFCriticConfig` | 0.7 |
| `hl_gauss_min`, `hl_gauss_max` | explicit finite Q support bounds | required for HL-Gauss |
| `hl_gauss_num_bins` | through config, >=2 | 101 |
| `hl_gauss_sigma_ratio` | sigma / bin width, through config | 0.1 (DEAS convention) |

```python
model = GR00TN15SVF.from_bc_checkpoint(
    BC_CHECKPOINT_PATH,
    time_embed_dim=64,
    outer_update="iql",
    q_loss_type="mse",  # or "hl_gauss" plus explicit bounds below
).to("cuda")

# Example ONLY for a reward convention whose Q range is [-100,0]:
model_hl = GR00TN15SVF.from_bc_checkpoint(
    BC_CHECKPOINT_PATH, outer_update="iql", q_loss_type="hl_gauss",
    hl_gauss_min=-100.0, hl_gauss_max=0.0,
).to("cuda")
```

Choose one model; the two examples are alternatives, not an instruction to load
two VLMs. Set bounds from the actual reward scale, discount and horizon; positive
rewards must not be silently mapped to a negative support. No reward shift occurs.

IQL with default `v_loss_type='mse'` uses:

```text
q_bar       = stopgrad(min_e Q_target,e(s,a))
L_iql_value = mean(expectile_weight(q_bar - V_iql(s)) * (q_bar - V_iql(s))^2)
y           = stopgrad(chunk_return + bootstrap_discount * V_iql(s'))
L_Q         = MSE(Q, y) or HL-Gauss(logits_Q, y)
```

`min` for IQL's dataset-action Q target is fixed independently of `q_aggregation`
(which still controls caller-built SVF endpoint/TD aggregation). V_iql is a
single state-only MLP with its own trainable feature projection; its next
value is the **online** V under no-grad, not the SVF inner critic or an EMA V.
All losses use one pre-optimizer parameter snapshot. Q, IQL V, and SVF inner
losses have independent parameter gradients. The existing Q/inner EMA updates
remain; no unnecessary target V is created.

```python
model.train()
optimizer = torch.optim.Adam((p for p in model.critic_head.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
result = model(
    current_inputs,
    actions=actions,
    next_inputs=next_inputs,
    chunk_returns=chunk_returns,             # [B], already discounted and terminal-masked
    bootstrap_discounts=bootstrap_discounts, # [B], gamma**H * continuation
    # Optional: all three together also train the SVF inner critic:
    # noisy_actions=x_t, times=t, soft_targets=conditional_soft_targets,
)
result['loss'].backward()
optimizer.step()
model.critic_head.update_targets()
```

Omitting the optional trio trains only Q + IQL V (critic warmup). Providing it
adds the original **MSE** SVF inner distillation. IQL V defaults to scalar
expectile regression; Q's HL-Gauss switch alone does not change the V loss.
Select `v_loss_type='deas'` explicitly for distributional IQL V (below).
No IQL advantage-weighted actor
update is implemented; the intended future actor update remains SVF guidance.
`q_targets` is rejected by IQL wrapper forward so externally supplied targets
cannot accidentally bypass IQL. Cached-context callers can use
`critic_head.iql_loss(obs, next_obs, ...)` directly; no feature transformation is
performed again. Wrapper forward processes current and next observations once
each, regardless of how many critic branches are trained.

HL-Gauss reuses DEAS `gr00t/model/critic/hlg.py`: Gaussian-bin probabilities and
cross-entropy averaged over members and batch; logits `[E,B,N]`, decoded Q is the
softmax expectation of bin centers. Probabilities/log-softmax/expectations use
float32, including under BF16 autocast. Unlike the legacy implementation, finite
targets outside the support are explicitly clipped to its endpoints to prevent
zero Gaussian normalization / NaNs; `critic/target_clipped_fraction` reports the
fraction clipped. Widen the support or revise reward scaling if this is frequent.
This metric is returned to the caller; no W&B run is launched by the module.
Nonfinite targets or invalid probability parameters fail explicitly.

### Optional DEAS distributional V

```python
model = GR00TN15SVF.from_bc_checkpoint(
    BC_CHECKPOINT_PATH,
    outer_update="iql", q_loss_type="hl_gauss", v_loss_type="deas",
    hl_gauss_min=Q_MIN, hl_gauss_max=Q_MAX,  # same support for Q and V
    time_embed_dim=64,
).to("cuda")
```

This is the DEAS `compute_value_loss` distributional loss, **not** HL-Gauss
Gaussian re-encoding of a scalar V target. It is only valid when IQL is enabled
and Q uses HL-Gauss; invalid combinations fail at configuration validation.

1. Compute scalar expectations for all target Q members; select the minimum
   member **per batch sample**, keeping that member's entire categorical distribution.
2. V emits `[B,N]` logits on the same bins/support as Q. Decode V's expectation.
3. Use `w=tau` when `Q >= V`, else `w=1-tau` (including DEAS's equality rule).
4. Minimize `mean(w * CE(stopgrad(p_Q_selected), p_V))` with stopped weights.

The next-state scalar expectation `E[p_V(s')]` still supplies Q's TD target.
Both probability decoding and cross-entropy run in float32 under BF16 autocast.
V-only loss cannot backpropagate to Q or the VLM; all three critic branches use
the once-processed frozen observation context. No binwise minimum or average of
member logits is used. IQL retains its fixed minimum-member rule rather than
copying the legacy DEAS `q_agg='mean'` branch.

`iql_value_logits` is included in IQL forward outputs in this mode. The SVF inner
critic remains scalar/MSE and time-conditioned. To return to scalar expectile V,
choose `v_loss_type='mse'`. Changing V mode changes its output-layer shape and
requires a new critic run; strict checkpoint loading rejects incompatible modes.
Older scalar V checkpoints lacking this field use its `mse` default.

Q/V loss types, IQL on/off, support, expectile and time width are serialized in the
critic config. Changing them is not a strict resume of the same critic: MSE to
HL-Gauss changes the Q output layer, and enabling IQL adds a network. Existing
default TD/MSE checkpoints still load with new configuration defaults. Use a
new run for architecture changes; original BC weights can be reused.

## Save/resume

Save into the selected RAID output directory through the future training loop:

```python
torch.save({
    'bc_checkpoint': BC_CHECKPOINT_PATH,
    'critics': model.critic_checkpoint(),
    'optimizer': optimizer.state_dict(),
    'step': step,
    # Also save scheduler, RNG, sampler progress and adaptive lambda state.
}, OUTPUT_CHECKPOINT_PATH)
```

Restore the same BC checkpoint and construct `SVFCriticConfig` from the saved
critic config, then `model.load_critic_checkpoint(saved['critics'])` and restore
the optimizer/scheduler/RNG. Critics use strict weight/config checks; this is not
HF `AutoModel`/`save_pretrained` registration and not a drop-in checkpoint for the
legacy DEAS BoN loader. The original BC checkpoint is not duplicated in the
critic-only payload. `state_dict()` tensors are live references: serialize at a
paused optimizer boundary, or clone them for asynchronous saving.

## Remaining integration

- Offline transitions/rewards/terminal masks and matching action-chunk horizon.
  Bigenlight BC data alone has no fabricated RL labels; do not train Q on it
  without defining the real reward/transition contract.
- Conditional reference BC-SDE sampler starting at each (x_t,t), frozen reference
  flow and a separate trainable guided DiT actor, adaptive temperature calibration.
- Guidance coefficient kappa^2*(1-t)/(t*lambda) and t_min gate, guided actor loss,
  full optimizer/Trainer, W&B, resumable checkpoints and evaluation.
- Nondefault fmrl options such as inner TD, split/each/aggout distillation and
  soft/min outer bootstrap variants are not automatically selected by this API.
  `target_tc_critic` is available for future inner TD but is not used in the
  default endpoint distillation target.

CPU test (does not touch running GPUs):

```bash
source experiments/robocasa_deas/activate.sh
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m pytest \
  tests/test_svf_critic.py tests/test_iql_critic.py -q
```

Tests cover real small attention modules, a lightweight BC wrapper, BF16 frozen
features with float32 critics, target/gradient isolation, padding, equations,
complete EMA and checkpoint round-trip. Full-size BC checkpoint loading and
end-to-end SVF training have not been exercised by these CPU tests.
