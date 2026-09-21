# GR00T Bigenlight — canonical local workspace

For real-hardware handoff, start with [REAL_ROLLOUT.md](REAL_ROLLOUT.md).
It distinguishes native BC inference from SVF checkpoints (not yet directly
loadable by the BC server), documents the private connection and robot contract,
and maps the current critic/cache/LoRA implementation.

Maintain N1.5 and N1.7 development and new launches here. Consolidated locally on
2026-09-20, including the latest cached IQL and frozen-Q SVF implementation.
Legacy working directories remain available for already-running jobs and queued
commands. Data, weights, feature caches and environments remain on RAID.
The consolidated code is maintained in this private Git repository.

```bash
cd ~/vla_finetune/gr00t-bigenlight
source environments/activate_n17.sh
IQL_REWARD=step-cost bash n17/examples/bigenlight_multitask/train_cached_iql.sh all # dry run
bash n17/examples/bigenlight_multitask/train_fixed_q_svf.sh all # dry run
```

For N1.5, use a separate shell, `source environments/activate_n15.sh`, then
`cd n15` before running its Python modules. Both versions expose `gr00t`; the
helpers select the matching RAID environment and this checkout's PYTHONPATH.
See [local migration notes](LOCAL_WORKSPACE.md) for existing cache compatibility,
legacy jobs, simulator submodules and the boundary with the separate fmrl repo.

## Contents

- `n15/`: modified `csmile-1006/DEAS-Isaac-GR00T` (GR00T N1.5).
- `n17/`: modified `NVIDIA/Isaac-GR00T` (GR00T N1.7).
- `environments/`: installed package-version inventories for the two source
  environments, plus original server activation/cache helpers for reference.
- `SOURCE_SNAPSHOT.json`: exact upstream commits, source remotes, per-file hashes,
  exclusions and pinned simulator submodules.
- `CONSOLIDATION_SOURCE_SNAPSHOT.json`: source hashes immediately before this
  consolidation's path changes; `SOURCE_SNAPSHOT.json` is the original export.
- `tools/server/`: shared inference checks, VRAM/synthetic-data and GPU utilities.
- `tools/snapshot_worktrees.py`: export procedure; refuses to overwrite version directories.

Original LICENSE, attribution and source headers are retained within each tree.
There is no new blanket license overriding the original components. Model weights
and datasets have their own upstream terms and are not bundled here.

The trees are working-tree snapshots on top of the recorded upstream commits,
not full upstream Git histories. This is deliberate: current custom code is saved
without importing old datasets, model binaries, or repository credentials.
N1.7 simulator submodules remain pinned Git links under `n17/external_dependencies/`.

## Clone and environments

```bash
git clone --recurse-submodules https://github.com/jun981015/gr00t-bigenlight.git
```

For BC inference alone the optional simulator submodules are not needed.
Use **separate environments and processes**: both source trees provide a package
named `gr00t`, and installing both into the same environment is unsafe.

| Version | Source environment at export | Dependency references |
| --- | --- | --- |
| N1.5 | Python 3.10 / PyTorch 2.5.1+cu124 | `n15/pyproject.toml`, `n15/experiments/robocasa_deas/bootstrap.sh`, `environments/n15-packages.txt` |
| N1.7 | Python 3.12 / PyTorch 2.9 | `n17/pyproject.toml`, `n17/uv.lock`, `environments/n17-packages.txt` |

Package inventories record versions; they are not portable lockfiles for CUDA,
system/FFmpeg libraries or local packages. Use the respective upstream install
instructions and matching GPU runtime, not a combined pip install.

Maintained BC/IQL/SVF and simulator launchers resolve code relative to this checkout.
Environment helpers use the existing server RAID environments; they are not an
automatic installer. Override storage settings for another machine as needed.
Do not reinstall the shared environments while legacy jobs are active; use the
provided activation helpers to select this source tree for new processes.

## Bigenlight BC recipes

Four tasks: carrot in pot, two-bowl stack, triple-bowl stack, cube stack.
Views: scene and wrist RGB. State/action: six arm joints in radians and one
normalized gripper coordinate; **absolute joint targets**, not EEF deltas.
Supervised action horizon is 16. Internal padded dimensions differ by version.

- N1.7 dataset/config/run docs: `n17/examples/bigenlight_multitask/README.md`.
- N1.5 UR7e config/training: `n15/experiments/bigenlight_multitask/`.
- Latest requested experiments: separate 50/task (200 episodes) and all-data
  (256 episodes), each batch 32 and 10,000 optimizer updates.
- N1.5 completed checkpoints exist on the original server, not inside Git.
- N1.7 corresponding jobs were started on the original server on 2026-09-20.
  This README is an export-time record, not a live status dashboard.

Dataset repositories (download outside this Git tree):

- https://huggingface.co/datasets/RLobot-jun/bigenlight_multitask_gr00t_30per_task
- https://huggingface.co/datasets/RLobot-jun/bigenlight_multitask_gr00t_50per_task
- https://huggingface.co/datasets/RLobot-jun/bigenlight_multitask_gr00t_all

Keep weights on Hugging Face or other model storage. Preserve original base-model
ID/revision, action-head weights, processor/modality configuration and training
normalization statistics together. An action-head-only export/load workflow has
been discussed but is **not yet packaged here as a completed loader**.

## Research additions

N1.5 contains opt-in IQL and SVF critic heads and their tests. See
`n15/experiments/robocasa_deas/SVF_CRITIC.md`: two SVF critics, optional state-only
IQL V, scalar or distributional Q/V losses, Fourier time embedding, frozen VLM
context and independent gradients/targets. Full SVF sampling/actor training is
not yet an end-to-end recipe. N1.7 offline-RL adapters are under `n17/gr00t/rl/`.
N1.7 includes cached-feature IQL and SVF with frozen imported IQL Q1/Q2; see
`n17/examples/bigenlight_multitask/IQL.md`, `FEATURE_CACHE.md`, and `FIXED_Q_SVF.md`.
The frozen-Q mode trains inner value and actor while preserving the original BC
conditioning for Q. Full pretrained GPU SVF convergence is not yet measured.
The separate `fmrl` repository is not included; its referenced commit is documented
in the critic implementation. Research code remains private in this repository.

## Real-robot deployment work remaining

Native inference entry points are present (`n15/scripts/inference_service.py`,
`n17/gr00t/eval/run_gr00t_server.py`), but a unified Bigenlight/UR7e deployment
adapter is not yet complete. Before real actuation, implement and validate:

1. N1.5 custom data-config registration and version-specific model/processor loading.
2. Correct scene/wrist ordering, RGB preprocessing, joint order and gripper scaling.
3. Action-chunk execution cadence, stale observation rejection and network timeouts.
4. Joint/speed limits, operator enable, emergency stop and safe stop on disconnect.
5. Observation replay and no-actuation tests before enabling robot commands.

Do not expose an unauthenticated inference server directly to the public internet.
Use an authenticated private connection or SSH tunnel and run each model version
in its own environment. Merely cloning this repository does not validate a robot.

## Excluded from Git

Credentials/tokens, model weights, optimizer states, datasets/demo payloads,
training logs, caches/environments, generated artifacts and upstream media files.
Some upstream tutorial image/demo links will therefore need the upstream assets.
Preserved training scripts refer to external data/model paths rather than bundles.
