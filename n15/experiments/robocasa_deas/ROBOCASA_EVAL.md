# DEAS GR00T N1.5 RoboCasa evaluation

The repository is `/home/yoon/vla_finetune/gr00t-bigenlight/n15`. The evaluation
script is the DEAS N1.5 path in `scripts/eval_policy_robocasa.py`; it uses
`embodiment_tag=new_embodiment`, `single_panda_gripper_rl_inference`, and an
action horizon of 16.

## One-time simulator setup

Run this inside the allocated GPU container, not on the login host:

```bash
cd ~/vla_finetune/gr00t-bigenlight/n15
bash experiments/robocasa_deas/bootstrap.sh
bash experiments/robocasa_deas/setup_robocasa.sh
```

The setup keeps the existing N1.5 torch environment and installs simulator
packages into `/raid/yoon/vla_finetune/envs/deas-gr00t-n1.5`. RoboCasa kitchen
assets are downloaded to the checked-out RoboCasa tree. To only test imports
when assets are already provisioned, use `SKIP_DOWNLOAD_ASSETS=1`; full kitchen
evaluation still needs the assets.

## Evaluation

First use a RoboCasa-finetuned N1.5 actor checkpoint, not the untouched base
model. A plain BC actor can be evaluated with:

```bash
source experiments/robocasa_deas/activate.sh
python experiments/robocasa_deas/manage.py eval \
  --task CoffeeSetupMug \
  --actor /raid/yoon/vla_finetune/outputs/<actor>/checkpoint-<step> \
  --output /raid/yoon/vla_finetune/evaluations/deas-n15-coffee \
  --execute
```

For a DEAS actor+critic checkpoint, add `--critic`:

```bash
python experiments/robocasa_deas/manage.py eval \
  --task CoffeeSetupMug \
  --actor /raid/yoon/vla_finetune/outputs/<actor>/checkpoint-<step> \
  --critic /raid/yoon/vla_finetune/outputs/<critic>/checkpoint-<step> \
  --output /raid/yoon/vla_finetune/evaluations/deas-n15-coffee-deas \
  --execute
```

The supported DEAS tasks are `CoffeeSetupMug`, `PnPMicrowaveToCounter`,
`TurnOffStove`, and `PnPCounterToMicrowave`. Results are written to
`eval.csv` and `success.txt` in the requested output directory. The current
launcher defaults to 50 episodes, 5 parallel environments, 10 critic samples,
and zero sampling temperature, matching the DEAS recipe.

If only the server is available, the same evaluator can use the client/server
path by omitting `--actor` in the underlying script; the local-model path above
is simpler and avoids an additional port dependency.
