"""Download and validate the public few-shot demos for both Q-VGM-style BC budgets."""

import argparse
import json

from .data import (
    HERE,
    RECIPE,
    atomic_json,
    dataset_path,
    download_selected,
    prepare,
    storage,
    validate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    presets = json.loads((HERE / "qvgm_presets.json").read_text())
    completed = {}
    # Long first: it has the fully specified BC step count and can run independently.
    for profile in ("qvgm-long", "qvgm-unified"):
        cfg = presets["profiles"][profile]
        summaries = []
        for suite in cfg["suites"]:
            count = cfg["demos_per_task"]
            source = storage() / "datasets/libero_public" / suite
            target = dataset_path(suite, count, args.seed)
            spec = RECIPE["datasets"][suite]
            download_selected(spec, source, count, args.seed)
            summary = prepare(
                source, target, suite=suite, spec=spec, demos_per_task=count, seed=args.seed
            )
            validate(target)
            summaries.append(summary)
        episodes = sum(row["episodes"] for row in summaries)
        if episodes != cfg["episodes"]:
            raise ValueError(f"{profile}: expected {cfg['episodes']} episodes, got {episodes}")
        completed[profile] = {"episodes": episodes, "datasets": summaries, "recipe": cfg}
        print(f"READY: {profile}, {episodes} episodes; no training launched.", flush=True)
    destination = storage() / "datasets/libero_n17_bc" / f"QVGM_READY_seed{args.seed}.json"
    atomic_json(destination, {"seed": args.seed, "reference": presets, "profiles": completed})
    print(f"Both BC data budgets ready: {destination}", flush=True)


if __name__ == "__main__":
    main()
