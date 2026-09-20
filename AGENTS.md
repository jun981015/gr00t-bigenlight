# Canonical GR00T workspace

Maintain N1.5 in `n15/` and N1.7 in `n17/`. Both provide `gr00t`; use separate
environments and `source environments/activate_n15.sh` or `activate_n17.sh`.
Read `n17/AGENTS.md` when changing N1.7. Keep model weights, datasets, feature
caches, logs and environments in external RAID storage, not this repository.

The sibling `Isaac-GR00T` and `DEAS-Isaac-GR00T` directories are legacy working
trees retained for in-flight jobs and provenance. Do not develop there or
automatically resync them over this canonical workspace. Do not move or replace
them while old jobs, checkpoint identities or queued commands reference them.

Commit and push only when requested by the user. Keep the separate fmrl
repository separate; record references.
