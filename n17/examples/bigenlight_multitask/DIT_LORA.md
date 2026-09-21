# DiT LoRA and reuse of the existing IQL cache

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
