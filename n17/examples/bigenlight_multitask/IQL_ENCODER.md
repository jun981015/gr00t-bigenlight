# Scalar IQL: with and without a pooled-feature encoder

Both variants are supported by `python -m gr00t.rl.train_cached --algorithm iql`:

| Option | Observation path |
| --- | --- |
| `--iql-encoder none` (default) | Frozen cached feature directly into Q/V MLPs |
| `--iql-encoder deas-mlp` | Pooled VLM feature through shared 2048→1024→1024→1024→64 encoder, then Q/V MLPs |

The added encoder uses SiLU and final Tanh, shared by online Q and V. The
optimizer includes each shared parameter once. Target Q has an independent EMA
copy including its encoder. State, embodiment and padded action layout remain
the same as baseline scalar IQL. This is NOT the full DEAS architecture: V
remains a scalar MLP, not the DEAS residual distributional V; Q uses MSE and V
uses expectile-weighted MSE. No HL-Gauss or dual discount is enabled.

BC/VLM/observation projection remain frozen cached inputs in both variants.
The post-pooling encoder is trainable. Match each critic with its 50/task or
all-data BC cache and normalization. Full checkpoints from either variant can
be imported by `load_frozen_iql_q`, including the encoder when present.
Weights between the two architectures are not interchangeable.

Completed encoder experiments: N1.7 50/task (200 episodes) and all (256 episodes),
each separate 10k and 30k runs, expectile 0.7, lr 3e-4, batch 32, gamma 0.99,
step-cost rewards (-1 per action, terminal successful action 0). Run names:
`bigenlight-n17-iql-encoder-{50per-task,all}-expectile0.7-{10k,30k}-20260921-encoder-iql`.
These used newly initialized encoder/Q/V, not a migration of old critic weights.

The completed cached SVF run
`bigenlight-n17-cached-svf-fixed-iql-50per-e07-q30k-lora16-b32-flow4-10k-20260921`
uses the OLD no-encoder 50/task IQL expectile-0.7 30k Q, frozen. Actor LoRA
rank16/alpha32 and inner critic are trained, batch32, lr3e-4, candidates8.
It continued from update357 with reference SDE flow_steps changed from10 to4,
preserving Adam/RNG/update count, and finished at update10000. It is not an
encoder-IQL SVF run. Flow-step transitions are recorded in checkpoint metadata.

Cached SVF permits an explicit `--resume CHECKPOINT --allow-flow-steps-change`
with the new `--flow-steps` and a fresh output directory. Other semantic
changes remain rejected. Keep the flag for subsequent resumes of this lineage.
Normal BC inference servers cannot directly load SVF training snapshots;
see the repository's REAL_ROLLOUT.md. Full-token caches are for offline training,
not a substitute for VLM inference on new live images.
