"""Bounded, ordered episode-shard prefetch for frozen feature extraction.

Each spawned CPU worker decodes/transforms complete episodes. Batches never mix
episodes or language prompts. NumPy payloads avoid large /dev/shm tensor queues.
"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import torch


_dataset = None
_collator = None


def initialize_worker(model_path, dataset_path):
    global _dataset, _collator
    from experiments.bigenlight_multitask.cache_features import make_dataset
    from gr00t.model.transforms import DefaultDataCollator

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    _dataset = make_dataset(model_path, dataset_path)
    _collator = DefaultDataCollator()


def prepare_episode(job):
    episode, trajectory, length, batch_size = job
    batches = []
    for start in range(0, length, batch_size):
        inputs = _collator(
            [
                _dataset.transforms(_dataset.get_step_data(trajectory, i))
                for i in range(start, min(start + batch_size, length))
            ]
        )
        batches.append(
            (
                start,
                {
                    key: value.numpy() if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                },
            )
        )
    return episode, trajectory, length, batches


def bounded_map(executor, function, jobs, prefetch):
    """At most prefetch submitted shards, with deterministic output ordering."""
    if prefetch < 1:
        raise ValueError("prefetch must be positive")
    jobs = iter(jobs)
    pending = deque()
    for _ in range(prefetch):
        job = next(jobs, None)
        if job is None:
            break
        pending.append(executor.submit(function, job))
    try:
        while pending:
            value = pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(function, job))
            yield value
    finally:
        for future in pending:
            future.cancel()


def episode_shards(model_path, dataset_path, jobs, workers=4, prefetch=4):
    if workers < 1:
        raise ValueError("workers must be positive")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(model_path, dataset_path),
    ) as executor:
        for episode, trajectory, length, batches in bounded_map(executor, prepare_episode, jobs, prefetch):
            yield (
                episode,
                trajectory,
                length,
                [
                    (
                        start,
                        {
                            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                            for key, value in inputs.items()
                        },
                    )
                    for start, inputs in batches
                ],
            )
