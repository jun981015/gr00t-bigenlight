# Bigenlight UR7e four-task BC corpus

Prepared from pinned Hugging Face revisions (see `prepare.py` / `SOURCE.json`):

| Source under `Bigenlight/` | Episodes | Frames | Merged episode IDs |
| --- | ---: | ---: | --- |
| carrot_in_pot_lighting_lerobot_v3 | 65 | 20,472 | 0–64 |
| bowl_stack_lighting_lerobot_v3 | 71 | 17,945 | 65–135 |
| bowl_stack_triple_lerobot_v3 | 60 | 36,030 | 136–195 |
| cube_stack_lerobot_v3 | 60 | 20,177 | 196–255 |
| Total | 256 | 94,624 | 0–255 |

## Storage and semantics

- Original snapshots: `~/raid/vla_finetune/datasets/<source name>/`.
- Unified corpus: `~/raid/vla_finetune/datasets/bigenlight_multitask_gr00t/`.
- N1.7 dataset path: `.../bigenlight_multitask_gr00t/n17`.
- N1.5 dataset path: `.../bigenlight_multitask_gr00t/n15`.
- These are **the same 256 episodes**, not separate datasets or train/val splits.
  N1.5's `data/` and `videos/` link to the N1.7 view, with version-specific stats metadata.
- All episodes are included for BC; no held-out validation split is claimed. CPU
  validation below checks file integrity and preprocessing, not policy performance.
- Four original task descriptions are preserved exactly. IDs are remapped without
  collisions. `source_mapping.json` maps every merged episode back to its source,
  revision, episode, parquet, and video frame offset. Recovery/collection annotations
  remain available in each original `meta/source_takes.json`; no recovery episodes
  are filtered and no reward/success/done labels are invented.
- Both recipes use **absolute joint targets**, six radians-valued joints plus
  gripper (`0=open`, `1=closed`), not EEF deltas. State/action values remain bit-exact.
  This deliberately differs from the old single-carrot N1.7 relative-action recipe.
- Both use action horizon **16**, scene + wrist RGB at 30 Hz. N1.7 internally pads
  to 40×132 and N1.5 to 16×32; only the 16×7 real action entries are supervised.
- V3 aggregate AV1 videos are converted to frame-accurate per-episode H.264 at
  original 1280×720, CRF18 (lossy re-encoding). Every clip's frame count is verified.
- Statistics are recomputed over the entire merged training corpus. N1.7 uses its
  standard percentile normalization; N1.5 uses the standard min/max transform.
  Image augmentation also follows each version's native pipeline; recipes are
  compatible, not a claim of perfectly controlled cross-version benchmarking.
- No task balancing is forced: longer tasks contain more frames.

## Prepare / validate (CPU; does not touch running GPU jobs)

```bash
source ~/vla_finetune/activate_gr00t.sh
cd ~/vla_finetune/gr00t-bigenlight/n17
python -m examples.bigenlight_multitask.prepare
python -m examples.bigenlight_multitask.validate

cd ~/vla_finetune/gr00t-bigenlight/n15
source experiments/robocasa_deas/activate.sh
python -m experiments.bigenlight_multitask.validate
```

Preparation is pinned and restartable. `READY.json` means conversion finished;
each view's `VALIDATION.json` records successful real-loader CPU preprocessing.
GPU model forward/backward is not tested by these commands. Do not modify or move
the N1.7 payload without updating the relative links in the N1.5 view.

## Train later, inside the allocated GPU container

Nothing trains automatically. Both entry points print a dry-run config unless
`--execute` is supplied. Defaults: one GPU, batch 32, 10,000 optimizer steps,
LLM/vision frozen, projector/DiT trainable, W&B **online** in project
`bigenlight-multitask-gr00t`, checkpoint cadence 5,000 steps on RAID.
Choose batch/steps at launch; defaults are not a VRAM capacity benchmark.

N1.7 dry run (add `--execute` as the first argument when ready):

```bash
bash ~/vla_finetune/gr00t-bigenlight/n17/examples/bigenlight_multitask/train.sh
```

N1.5 dry run (append `--execute` when ready):

```bash
cd ~/vla_finetune/gr00t-bigenlight/n15
source experiments/robocasa_deas/activate.sh
python -m experiments.bigenlight_multitask.train \
  --output ~/raid/vla_finetune/outputs/bigenlight-n15-bc
```

N1.5 reuses the existing recovery callback: extra first-100-step / 30-minute
checkpoints, save-and-stop after 8 hours, and only the latest committed checkpoint
retains optimizer/scheduler/RNG state. Earlier model weights stay on RAID.
Resume with the same output and training settings plus `--resume --execute`.
N1.7 uses the existing latest-state checkpoint policy and 8-hour limit.
Never launch on an occupied GPU without separately deciding what to stop.

## Prepared N1.7 runs: 50/task and all (2026-09-20)

These are **new N1.7 runs from the original GR00T-N1.7-3B weights**, not resumes
of the existing N1.5 Bigenlight checkpoints. Preparation does not stop those jobs
or launch anything. Both datasets already have N1.7 `VALIDATION.json` records;
no download, conversion, or full-size GPU forward/backward is part of preparation.

| Variant | GPU | Episodes | Dataset under RAID datasets/ |
| --- | --- | ---: | --- |
| `50per-task` | 0 | 200 (indices 0–49 per task) | `bigenlight_multitask_gr00t_50per_task/n17` |
| `all` | 1 | 256 | `bigenlight_multitask_gr00t/n17` |

Each run is independent: batch 32, accumulation 1, 10,000 optimizer steps,
BF16, learning rate 1e-4, frozen Cosmos/VLM and vision, trainable projector/DiT.
The 16-step absolute-joint action recipe above is unchanged. W&B is online in
`bigenlight-multitask-gr00t`, with training metrics every **50 optimizer steps**.
Run names/output folders include `bigenlight-n17-{variant}-b32-10k-{timestamp}`
under RAID `outputs/`, with a separate `train.log` and fresh W&B identity each.

Checkpoint policy: every 5,000 steps, plus early step 100 and 30-minute recovery
saves; full save-and-stop after 8 hours. Older weights remain, but only the latest
committed checkpoint retains optimizer/scheduler/RNG state. These extra saves do
not reset the 10,000-step learning-rate schedule. Config/normalization/processor
artifacts are saved by the standard N1.7 pipeline.

**Preview only** (safe while N1.5 is still training; no GPU queries or W&B calls):

```bash
bash ~/vla_finetune/gr00t-bigenlight/n17/examples/bigenlight_multitask/run_n17.sh 50per-task
bash ~/vla_finetune/gr00t-bigenlight/n17/examples/bigenlight_multitask/run_n17.sh all
```

Later, after the selected GPU is free and inside the allocated container, append
`--execute` to the corresponding command. Use separate persistent terminals for
the two jobs. The launcher refuses an occupied GPU and never kills existing
processes or creates keepalives. An existing user-owned keepalive may only be
exempted with the GPU guard's explicit `CARROT_KEEPALIVE_PIDS` setting. There is
no auto-start queue or completion watcher. Execute mode is fresh-run only; ask
for a verified resume command if the container expires mid-run.

The generic `train.sh` also supports `BIGENLIGHT_DATASET_PATH` and
`BIGENLIGHT_OUTPUT_PATH`; for this comparison prefer the named wrapper so model
version, subset, GPU and W&B project cannot be confused with N1.5.

## First 30 episodes per task (120 total)

The separate `~/raid/vla_finetune/datasets/bigenlight_multitask_gr00t_30per_task/`
corpus selects **original episode indices 0–29 in each of the four tasks** (not
random sampling, not 30 frames). It contains 120 episodes and 46,660 frames.
Subset episode IDs are remapped to 0–119 in task order; `source_mapping.json`
also retains the parent merged ID and original source ID. Both `n17/` and `n15/`
views are available, with statistics fitted only on these 120 episodes. Videos
link to the full corpus, which must remain in place. The 256-episode corpus is
unchanged; the unselected episodes are not automatically designated a test set.

Reproduce into a new, nonexistent output directory:

```bash
source ~/vla_finetune/activate_gr00t.sh
cd ~/vla_finetune/gr00t-bigenlight/n17
python -m examples.bigenlight_multitask.subset --per-task 30 --output /path/to/new/output
```

Select it for N1.7 training (dry run; add `--execute` when ready):

```bash
BIGENLIGHT_DATASET_PATH=~/raid/vla_finetune/datasets/bigenlight_multitask_gr00t_30per_task/n17 \
  bash ~/vla_finetune/gr00t-bigenlight/n17/examples/bigenlight_multitask/train.sh
```

For the N1.5 command above, add
`--dataset-path ~/raid/vla_finetune/datasets/bigenlight_multitask_gr00t_30per_task/n15`
and choose a **new** output directory (do not resume a full-corpus run with different data).
The CPU validators accept `--dataset-root <subset root>` (N1.7) and
`--dataset-path <subset root>/n15` (N1.5).

## Frozen N1.7 BC + environment critic-only IQL

See [IQL.md](IQL.md) for the opt-in all-success terminal reward assumption, fully
frozen BC feature encoder, independent scalar Q/V heads, and single-GPU launch /
recovery commands. This does not train the actor or the SVF inner critic.
