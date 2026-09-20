# IQL critic for the fixed GR00T N1.5 BC actor

Select `manage.py train critic --critic-algorithm iql`. Without that option,
the legacy DEAS critic is unchanged. This implements scalar Q/V learning and
BC best-of-N inference, not IQL's advantage-weighted actor update.

```bash
cd ~/vla_finetune/gr00t-bigenlight/n15
# Inspect configuration; add --execute to start after the BC checkpoint is ready.
bash experiments/robocasa_deas/run_iql_critic_1gpu.sh \
  /raid/yoon/vla_finetune/outputs/deas-n15-filtered-bc-direct-1gpu-b32/checkpoint-5000
# Resume with the same actor checkpoint and output: add --resume --execute.
```

Defaults: GPU 1 / batch 16 (current and next images are both encoded), 30,000
steps, Adam LR 1e-4 / constant schedule, expectile 0.7, gamma 0.99, EMA tau
0.005, action chunk H=16. W&B online / logging 50 steps / regular save 5,000
steps; existing recovery saves and latest-optimizer pruning also apply.
GPU memory for this critic launch has not yet been benchmarked.

Data: four-task `demos + rollouts`, including failed rollouts. Normalization
statistics come from the explicit BC actor checkpoint via the recipe worker.
The VLM, LayerNorm and self-attention feature transform remain frozen. A shared
trainable feature bottleneck feeds scalar Q1/Q2/V. The target Q has an EMA copy
of both its bottleneck and Q networks. Checkpoints contain the target encoder.

Given a chunk of H transitions:

* V: expectile squared loss against stop-gradient min(target Q1, target Q2).
* Q: sum of the two scalar MSE losses to discounted chunk return + gamma^H V(s[t+H]).
* The first terminal transition's reward is included; later rewards and
  bootstrap are masked. H=1 reduces to the ordinary one-step target.
* Q/V losses use one common pre-update parameter snapshot with a joint Adam
  step. This differs from the reference implementation's sequential V then Q
  optimizer steps. No actor loss is optimized.

There is one discount, unlike DEAS's two-discount target. The recipe disables
the DEAS -1 reward shift. It still uses the existing dataset loader's reward
preprocessing (RoboCasa successful episodes get last-15-frame rewards).
`next.done` is treated as terminal; these exports do not separately identify
time-limit truncations. `nstep != 1` and `expand_batch > 1` are rejected because
the data configuration supplies next observations at t+H.

Feature processing is non-mutating and performed once for current and once
for next observations. Repeated forward calls never shift rewards in-place.
The same processed feature representation is used by V, Q and inference.

Evaluation: use the existing `manage.py eval --actor <same BC> --critic <IQL checkpoint>`.
The checkpoint's `rl_config.algorithm=iql` chooses the scalar head and scalar
best-of-N scoring automatically. Do not substitute a different actor checkpoint:
the frozen features and normalization must match the critic's training inputs.

Reference equations: https://github.com/ikostrikov/implicit_q_learning/blob/master/critic.py
