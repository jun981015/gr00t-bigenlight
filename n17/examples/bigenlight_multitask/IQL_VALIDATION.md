# 50/task IQL holdout validation

The BC and IQL training split has 200 episodes (50/task). The evaluation-only
view contains the complement of the original source episode identities: 56
episodes / 18,268 frames (carrot 15, bowl 21, triple bowl 10, cube 10).
Videos/parquet are symlinked, never copied or reindexed. Do not use this
non-contiguous-ID view for BC training or publish it as a standalone dataset.

Feature extraction uses the **50/task BC checkpoint**, its saved processor and
normalization, not the full-data BC. The separate holdout cache is pooled
critic conditioning, not a DiT token cache. Original training caches are untouched.

```bash
source environments/activate_n17.sh
bash n17/examples/bigenlight_multitask/validate_iql_50per.sh
```

The server-local queue waits for the registered 30k and 10k sweeps, checks that
the GPU is free, extracts holdout features (batch 32), and evaluates each 50/task
30k critic (kappa 0.7/0.8/0.9). It records status in
`/raid/yoon/vla_finetune/outputs/iql-50per-30k-holdout-20260921.json`.
No LLM polling is needed. Failed dependencies stop evaluation. Extraction is
resumable by episode; partial evaluation output requires inspection and a fresh
output directory before retry. No training parameters or original W&B runs change.

Reports and per-transition predictions are stored under
`/raid/yoon/vla_finetune/outputs/bigenlight-n17-iql-50per-holdout56-kappa*-30k-20260921`.
W&B uses separate online `critic-holdout-eval` runs in
`junhyeong/bigenlight-multitask-gr00t`.

Metrics:

- Q=min(Q1,Q2) versus discounted demonstration return: MAE, RMSE, signed bias,
  overestimate fraction and correlation.
- Twin-Q TD MAE/MSE with `r_chunk + gamma**H * V(s_next)`, terminal bootstrap off.
- IQL expectile residual using EMA target Q (same target definition as training).
- Twin-Q disagreement, Q/V/return ranges and progress-decile summaries.
- Transition-weighted and episode-weighted aggregate errors, plus per-task
  episode-weighted results. Full chunks only: final H-1 start positions excluded.

The reward is -1 per action, 0 on the last successful action, gamma=0.99. Returns
are computed backwards over the *entire* trajectory, not just the action chunk.
Demonstration returns are not ground truth for the improved IQL policy. Do not
select kappa by V-to-MC MSE, and do not interpret this success-only heldout test as
validation of failure discrimination, out-of-distribution actions, or robot
success rate. Once used for tuning, these episodes are validation, not a final test.
