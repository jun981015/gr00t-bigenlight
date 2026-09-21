# DEAS critic on frozen BC caches

Run from the N1.7 environment inside the allocated GPU container:

```bash
python examples/bigenlight_multitask/run_deas_cached_sweep.py --sweep-id 20260921-deas-dualdiscount --execute
```

Eight independently initialized runs: 50/task and all episodes, learning rates
1e-4 and 3e-4, 10k and 30k steps. Batch 32, seed 0, constant Adam learning rate,
expectile 0.7, target Q EMA tau 0.005. W&B online logs every 50 steps. Save every
5k for 10k runs, every 10k for 30k runs. Milestone weights are retained; only the
latest optimizer/recovery state is retained. Re-run the same sweep ID to resume.

For horizon 16, the target is sum(0.9**j * reward[j]) + 0.99**16 * V(next),
with zero bootstrap on terminal chunks. Reward is -1 per action, 0 on the final
successful action. Incomplete trailing chunks are excluded. Action padding is
masked and only real action coordinates are passed to the Q heads.

DEAS reference: n15/gr00t/model/action_head/deas_critic.py and
n15/gr00t/model/critic/{networks,hlg}.py. Q uses 101-bin HL-Gauss on [-100, 0],
sigma 0.1 bin-width. V uses expectile-weighted cross-entropy against the full
distribution of the lower-mean target Q head, not Gaussian labels around its mean.
Q heads are 4x512 LN/GELU MLPs. V is a width-512, depth-4 residual BRO network.
The pooled VLM projection is 2048->1024->1024->1024->64 with SiLU then tanh.
The projection receives both Q and V gradients, as in the reference forward.

Adaptations: N1.7's own frozen BC LN/self-attention is already included in its
cache. N1.5 token attention is not transplanted into N1.7. The single Bigenlight
embodiment uses one trainable projection rather than allocating unused category
slices. Pooled cache features are immutable: no repeated LN/attention and no
in-place reward subtraction. HL-Gauss targets outside support are clipped to
avoid zero-normalization NaNs. These are DEAS critic/loss adaptations to N1.7,
not a bit-for-bit reproduction of its entire N1.5 training frontend or schedule.

N1.5 caches are generated separately in its environment using
`bash experiments/bigenlight_multitask/cache_both.sh`. They retain the same common
cache format, with explicit `model_version=n15` and feature-layout metadata, their
own BC weight identity, and frozen N1.5 BC LN/self-attention applied exactly once.
