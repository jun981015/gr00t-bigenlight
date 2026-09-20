# Local consolidation — 2026-09-20

`~/vla_finetune/gr00t-bigenlight` is now the source of truth for GR00T work.

| Component | Canonical location |
| --- | --- |
| N1.5 / DEAS / critic heads | `n15/` |
| N1.7 / BC / IQL / caches / fixed-Q SVF | `n17/` |
| Environment and RAID configuration | `environments/` |
| Shared server utilities | `tools/server/` |
| Datasets, BC/IQL/SVF weights, caches, logs | `/raid/yoon/vla_finetune/` (external) |

Use separate shells/processes for each version:

```bash
cd ~/vla_finetune/gr00t-bigenlight
source environments/activate_n17.sh
cd n17
python -c 'import gr00t; print(gr00t.__file__)'
# .../gr00t-bigenlight/n17/gr00t/__init__.py
```

For N1.5 substitute `activate_n15.sh` and `cd n15`. For a running shell in the
legacy source directory, change directories too: Python gives its working
directory priority over PYTHONPATH. Shared editable installations are retained
for existing jobs; new launches through these helpers select canonical sources.

## Existing jobs and caches

The old `Isaac-GR00T` and `DEAS-Isaac-GR00T` directories have not been moved,
deleted, made into symlinks, or patched by this consolidation. In-flight extraction
and IQL queue commands continue using their original source paths. Future
development and newly submitted commands belong in this repository.

Cache manifests include absolute encoder source paths. The canonical N1.7 code
accepts relocation only when the exact encoder file content hashes match and
all BC/data paths, configuration hashes, weight stats and horizon still match.
It preserves the old manifest bytes, so existing cached IQL checkpoint hashes
remain valid. A changed encoder or mismatched dataset/BC is still rejected.
This does not make caches portable across arbitrary model or dataset moves.

The old snapshot is preserved in Git's existing commit; the latest pre-migration
source inventory is `CONSOLIDATION_SOURCE_SNAPSHOT.json`. That inventory describes
the imported source, not the subsequent canonical path adaptations. Do not rerun
the original snapshot exporter over this maintained checkout.

## Simulator and algorithm dependencies

Simulator submodule URLs and commit pins are unchanged. If needed, initialize
them from this root with `git submodule update --init --recursive`; this can
download external code and is not performed by environment activation.
Existing RAID simulator environments/assets remain external dependencies.

`../fmrl` remains a separate repository on its own `dh` branch. It is the SVF
algorithm reference, not a second copy of our GR00T training implementation.
LeRobot/hil-serl repositories also remain independent. No data, model weights,
credentials, runtime logs or virtual environments were imported into Git.

The initial consolidation was completed locally; committing and publishing are
performed separately when requested by the user.

Validation: both version environments imported canonical source paths; N1.7
RL/launcher tests passed (91), N1.5 critic/data tests passed (68). BC, cached-IQL
and fixed-Q-SVF launchers passed dry runs and shell syntax checks. The existing
all-data RAID feature cache matched the relocated encoder identity without any
manifest rewrite or extraction. No GPU learner was launched during consolidation.
