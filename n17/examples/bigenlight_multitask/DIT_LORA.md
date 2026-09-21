# DiT LoRA and reuse of the existing IQL cache

## Kappa/g controls

Both live `gr00t.rl.train` and cached `gr00t.rl.train_cached_svf` accept
`--kappa K --g G`. The canonical checkpoint config still stores
`lambda_multiplier = K**2/G`, preserving old checkpoints. The legacy
`--lambda-multiplier C` remains supported, but cannot be combined with `--g`.
Likewise `--g` is incompatible with fixed `--soft-lambda` (cached trainer).
Positive K/G are required for this parameterization; kappa=0 is still available
through the legacy c interface. No flags retains historical kappa=c=g=1.

Lambda is c * max(candidate-Q spread, 1e-3). Metrics expose sv/kappa, sv/c and
sv/g (g is omitted for absolute-temperature mode). This is not an extra
post-gradient multiplier. No bounded-guidance clipping is introduced here.
The newer fmrl FAC sweep includes clipping; matching K/G alone does not
reproduce that entire protocol.

The 50/task launchers accept `SVF_KAPPA` and `SVF_G`. For a new TD-critic run:

```bash
SVF_KAPPA=0.4 SVF_G=0.25 bash n17/examples/bigenlight_multitask/train_cached_td_svf_50per.sh
```

This command starts training. Check GPU availability first. Use fresh runs
from the same BC, Q/inner initialization seed and budgets for comparisons,
not different points of one changing critic. A compact initial grid is
K in {0.2,0.4,0.6}, G in {0.25,0.5}; the implied c values are
{0.16,0.08}, {0.64,0.32}, {1.44,0.72}, respectively. Compare Q action sensitivity,
terminal TD residuals, guidance/BC velocity norms and heldout/rollout behavior;
training loss alone cannot select a policy. Existing running jobs are unchanged.

## Shared SVF endpoint draws (2026-09-21)

Temperature estimation and the inner soft-value target now share the same
sampled time, noisy action anchor, SDE endpoints and endpoint Q scores. This
intentionally replaces the previous independent Monte Carlo estimates.
`lambda_batch_size` still limits the prefix of Q-score columns used for lambda;
fixed `soft_lambda` remains unchanged. Both joint and inner-only modes use this.
With 8 candidates and 10 flow steps, the second 80-call reference rollout is
removed. Integration excludes completed rows from reference forward and stops
when all rows reach t=1. Actor training and BC-loss logging still cost one DiT
call each in joint fixed-Q mode (at most 82 total, often fewer).
Existing checkpoints remain loadable, but resumed random draws and the lambda/
target correlation differ from old runs. Already-running processes are not
modified; this applies after a new launch. No GPU throughput claim is made.

The existing `bigenlight-n17-*-bc10000-h16-v1` caches hold **pooled critic vectors**,
normalized action chunks and transition metadata. They do not contain all VLM
tokens. Freezing a projection does not restore information removed by pooling.
Keep these caches: they remain the authoritative feature input to IQL Q and the
new inner critic. No existing cache or queue is changed by this implementation.

## Use existing caches now

```bash
cd ~/vla_finetune/gr00t-bigenlight
source environments/activate_n17.sh
SVF_DIT_LORA_RANK=16 bash n17/examples/bigenlight_multitask/train_fixed_q_svf.sh all
```

This is a dry run. `--execute` requires the completed IQL checkpoint and an idle
allocated GPU. Rank 16 / alpha 32 are defaults for the new cached trainer;
`SVF_DIT_LORA_RANK` and `SVF_DIT_LORA_ALPHA` control the live launcher option.
Without `SVF_DIT_LORA_RANK`, the previous full-action-head SVF mode remains available.

With LoRA enabled:

- DiT Q/K/V/output attention linear layers receive zero-initialized LoRA
  residuals. Only the LoRA A/B weights are trainable inside the actor.
- VLM, VL LN/self-attention, state encoder, action encoder/decoder, position
  embeddings and BC reference are frozen. Base weights stay BF16; LoRA weights
  and optimizer are FP32. Inner critic is trained normally; env Q is fixed IQL
  or learned by SVF TD according to the selected mode.
- Q/inner feature vectors are read from the **existing pooled cache**, matched
  by episode/frame indices. Action normalization, mask, reward, horizon and
  discount must agree. For fixed-Q LoRA this is automatic from `--fixed-iql-cache`.
- VLM and observation projection are still computed once per live observation
  batch for the actor/reference, then shared across every ODE/SDE step using
  distinct `projected_vl_features` and `projected_state_features` keys. The actor
  never sends these projected tokens through `_encode_features` again.
- Frozen action encoder/decoder still run: their inputs change with x_t, t and
  DiT output. Freezing decoder weights does not disable backpropagation through
  it to the DiT LoRA weights.

For joint TD-Q + inner + LoRA, use the general `gr00t.rl.train` CLI with
`--algorithm svf --backend gr00t --dit-lora-rank 16 --critic-feature-cache CACHE`
and the matching single dataset, BC checkpoint, reward and horizon. Omitting the
fixed-IQL flags selects ordinary SVF TD. IQL V(s) is not reused as inner V.

## Optional future token cache

To remove the VLM and observation projection from the training process completely,
**additional projected token/state data must be extracted once**. The current
pooled cache stays unchanged and is reused alongside it. No such extraction is
started automatically. Token caches may be far larger than pooled caches; each
episode stores its full BF16 token tensor, state features and masks.

The optional tools are:

```bash
python -m gr00t.rl.cache_actor_features --pooled-cache POOLED_CACHE \
  --output-dir NEW_RAID_TOKEN_CACHE --batch-size 32
python -m gr00t.rl.train_cached_svf --pooled-cache POOLED_CACHE \
  --actor-cache NEW_RAID_TOKEN_CACHE --output-dir NEW_RAID_RUN \
  --env-q fixed-iql --iql-checkpoint IQL_CHECKPOINT \
  --dit-lora-rank 16 --reward step-cost --wandb-project bigenlight-multitask-gr00t
```

These Python commands execute work when invoked; use them only deliberately.
For TD-Q mode use `--env-q td` and omit `--iql-checkpoint`. Extraction is
episode-resumable with exclusive atomic safetensor files. Training refuses
incomplete/mismatched caches, reads sampled token rows from disk, and constructs
only the action head (never a VLM). It preserves the exact original critic
vectors. Reward remains selectable independently of both caches.

Checkpoints contain full frozen and trainable learner state, not just LoRA
adapters. Resume requires the same cache identities and LoRA configuration.
No standalone LoRA merge/export or robot deployment is claimed. Tests cover
velocity parity, projection bypass, frozen-weight invariance, both Q modes,
existing critic-cache reuse, token disk reads and cached trainer resume with
real small DiT heads. Full N1.7 GPU memory/throughput remains to be measured.
