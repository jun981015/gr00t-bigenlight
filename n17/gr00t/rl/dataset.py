"""Episode-safe offline RL over the existing GR00T LeRobotEpisodeLoader.

Rows follow (o_t, a_t, r_t, terminated_t, truncated_t), where flags describe
the outcome of a_t. Labels are explicit, including any user-selected assumptions.
"""

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, Dataset, Sampler

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, VLAStepData

from .types import RLTransition


@dataclass(frozen=True)
class EpisodeRLAnnotations:
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    # Needed when the last row contains an action but its nonterminal successor
    # is not another row (especially a time-limit truncation).
    final_observation: VLAStepData | None = None

    def validated(self, num_rows: int) -> "EpisodeRLAnnotations":
        rewards = np.asarray(self.rewards, dtype=np.float32)
        flags = [np.asarray(self.terminated), np.asarray(self.truncated)]
        if rewards.ndim != 1 or len(rewards) not in (num_rows, num_rows - 1):
            raise ValueError(
                "Annotations must describe N or N-1 transitions for N observation rows"
            )
        if len(rewards) == 0 or not np.isfinite(rewards).all():
            raise ValueError("Rewards must be nonempty and finite")
        for value in flags:
            if value.shape != rewards.shape or not np.isin(value, [0, 1]).all():
                raise ValueError("terminated/truncated must be aligned binary arrays")
        terminated, truncated = [value.astype(bool) for value in flags]
        if (terminated & truncated).any():
            raise ValueError("A transition cannot be both terminated and truncated")
        if (terminated[:-1] | truncated[:-1]).any():
            raise ValueError("Found transitions after an episode boundary; split the episode first")
        return replace(self, rewards=rewards.copy(), terminated=terminated, truncated=truncated)


class AllSuccessTerminalAnnotations:
    """Opt-in assumption: every episode succeeds on its final recorded action.

    Each of N rows is an action transition. No source files are modified; the
    missing final observation is allowed only because terminal bootstrap is zero.
    This is NOT a measured success timestamp or suitable for mixed-success data.
    """

    def __call__(self, loader, index):
        length = loader.get_episode_length(index)
        if length < 1:
            raise ValueError("Cannot label an empty episode")
        rewards = np.zeros(length, dtype=np.float32)
        terminated = np.zeros(length, dtype=bool)
        rewards[-1], terminated[-1] = 1.0, True
        return EpisodeRLAnnotations(rewards, terminated, np.zeros(length, dtype=bool))


class AllSuccessStepCostAnnotations(AllSuccessTerminalAnnotations):
    """Opt-in per-action -1 cost, with 0 on the final successful action.

    Success is still assumed at the final recorded action, not detected from video.
    This is per environment action, NOT one -1 reward per action chunk.
    """

    def __call__(self, loader, index):
        annotation = super().__call__(loader, index)
        return replace(annotation, rewards=annotation.rewards - 1.0)


@dataclass(frozen=True)
class ColumnRLAnnotations:
    """Read explicitly selected raw parquet columns (GR00T drops these modalities).

    Set a flag column to None only when that flag is known to be absent/false.
    A final observation-only row is never treated as an additional transition.
    For computed rewards, pass a custom callable instead of this class.
    """

    reward_column: str
    terminated_column: str | None = None
    truncated_column: str | None = None
    last_row_is_observation: bool = False

    def __call__(self, loader: LeRobotEpisodeLoader, index: int) -> EpisodeRLAnnotations:
        meta = loader.episodes_metadata[index]
        episode_id = meta["episode_index"]
        relative = loader.data_path_pattern.format(
            episode_chunk=episode_id // loader.chunk_size, episode_index=episode_id
        )
        columns = list(
            dict.fromkeys(
                key
                for key in (self.reward_column, self.terminated_column, self.truncated_column)
                if key is not None
            )
        )
        try:
            data = pd.read_parquet(loader.dataset_path / relative, columns=columns)
        except Exception as exc:
            raise ValueError(
                f"Cannot read explicit RL columns {columns} for episode {episode_id}"
            ) from exc
        if len(data) != loader.get_episode_length(index):
            raise ValueError(f"Episode {episode_id}: metadata/parquet row count mismatch")
        if self.last_row_is_observation:
            data = data.iloc[:-1]

        def flag(column):
            return np.zeros(len(data), dtype=bool) if column is None else data[column].to_numpy()

        return EpisodeRLAnnotations(
            data[self.reward_column].to_numpy(),
            flag(self.terminated_column),
            flag(self.truncated_column),
        )


@dataclass(frozen=True)
class DEASRoboCasaAnnotations:
    """Published DEAS collection convention: success terminal vs failed time limit.

    This is an explicit dataset-specific interpretation, NOT a generic done mapping.
    Preserve sparse recorded rewards; do not apply N1.5's last-15-frame shaping.
    Failed rollouts lack final post-action observations, so callers must explicitly
    disable timeout bootstrapping (or supply a different annotation provider).
    """

    def __call__(self, loader, index):
        labels = ColumnRLAnnotations("next.reward", "next.done")(loader, index)
        labels = labels.validated(loader.get_episode_length(index))
        if not np.isin(labels.rewards, [0, 1]).all():
            raise ValueError("DEAS RoboCasa expects recorded binary success rewards")
        done = labels.terminated
        if not done[-1]:
            raise ValueError("DEAS RoboCasa episode must have a recorded final next.done")
        success = bool(labels.rewards[-1] > 0)
        if not success and np.any(labels.rewards > 0):
            raise ValueError(
                "Success reward before a failed boundary; inspect collection semantics"
            )
        return EpisodeRLAnnotations(labels.rewards, done & success, done & (not success))


class LeRobotOfflineRLDataset(Dataset):
    """Map-style transitions with a fixed, fully observed action chunk.

    H action steps form one semi-Markov action: R=sum(gamma**i*r_i),
    next_observation=o[t+H], discount=bootstrap_gamma**H * bootstrap_mask.
    bootstrap_gamma defaults to gamma (ordinary SMDP); dual discounts are opt-in.
    Incomplete action chunks are dropped, never fabricated by repeating actions.
    History observations may repeat the episode's initial frame, not another episode.
    """

    def __init__(
        self,
        episode_loader: LeRobotEpisodeLoader,
        embodiment_tag: EmbodimentTag,
        annotations: Callable[[LeRobotEpisodeLoader, int], EpisodeRLAnnotations],
        *,
        horizon: int = 1,
        gamma: float = 0.99,
        bootstrap_gamma: float | None = None,
        bootstrap_on_truncation: bool = True,
        cache_episodes: int = 1,
        cache_bytes: int = 0,
    ):
        if horizon < 1 or not 0 <= gamma <= 1 or cache_episodes < 1:
            raise ValueError("Require horizon>=1, gamma in [0,1], cache_episodes>=1")
        bootstrap_gamma = gamma if bootstrap_gamma is None else bootstrap_gamma
        if not 0 <= bootstrap_gamma <= 1:
            raise ValueError("bootstrap_gamma must be in [0,1]")
        self.episode_loader = episode_loader
        self.embodiment_tag = embodiment_tag
        self.horizon, self.gamma = horizon, gamma
        self.bootstrap_gamma = bootstrap_gamma
        self.bootstrap_on_truncation = bootstrap_on_truncation
        self.cache_episodes = cache_episodes
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self.cache_bytes = cache_bytes
        self._cache_sizes = {}
        self._cache_nbytes = 0
        self._cache = OrderedDict()
        self.modality_configs = dict(episode_loader.modality_configs)
        for key, config in self.modality_configs.items():
            if key != "action" and any(delta > 0 for delta in config.delta_indices):
                raise ValueError(
                    f"Future {key} observations would leak information into the policy"
                )
        if "action" not in self.modality_configs or "language" not in self.modality_configs:
            raise ValueError("GR00T RL requires action and language modality configs")
        original_action_indices = self.modality_configs["action"].delta_indices
        if (
            original_action_indices != list(range(len(original_action_indices)))
            or not original_action_indices
        ):
            raise ValueError(
                "RL requires contiguous action steps starting at zero; explicitly resample strided data first"
            )
        # Sampling H is explicit; normalization still uses the existing processor.
        self.modality_configs["action"] = replace(
            self.modality_configs["action"], delta_indices=list(range(horizon))
        )
        self.observation_configs = {
            key: value for key, value in self.modality_configs.items() if key != "action"
        }
        self.annotations = []
        self.episode_sizes = []
        for index in range(len(episode_loader)):
            num_rows = episode_loader.get_episode_length(index)
            labels = annotations(episode_loader, index).validated(num_rows)
            blocks_bootstrap = labels.terminated[-1] or (
                labels.truncated[-1] and not bootstrap_on_truncation
            )
            if (
                len(labels.rewards) == num_rows
                and labels.final_observation is None
                and not blocks_bootstrap
            ):
                raise ValueError(
                    f"Episode {index}: missing final next observation for a bootstrapping transition. "
                    "Supply final_observation, store a final observation-only row, or explicitly "
                    "disable bootstrap_on_truncation for a labeled time limit."
                )
            self.annotations.append(labels)
            self.episode_sizes.append(max(0, len(labels.rewards) - horizon + 1))
        self.offsets = np.concatenate(([0], np.cumsum(self.episode_sizes)))
        if not len(self):
            raise ValueError(
                "No complete RL action chunks; reduce horizon or supply longer episodes"
            )

    def __len__(self):
        return int(self.offsets[-1])

    def _episode(self, index):
        active = getattr(self, "_active_batch_episode", None)
        if active is not None and active[0] == index:
            return active[1]
        if index not in self._cache:
            frame = self.episode_loader[index]
            if len(frame) != self.episode_loader.get_episode_length(index):
                raise ValueError(f"Episode {index}: decoded length differs from indexed length")
            # Include ndarray payloads: pandas object-column accounting alone
            # excludes the decoded RGB buffers. Conservative double counting is OK.
            size = int(frame.memory_usage(deep=True).sum()) + sum(
                value.nbytes
                for column in frame.columns
                for value in frame[column]
                if isinstance(value, np.ndarray)
            )
            if self.cache_bytes and size > self.cache_bytes:
                return frame  # Oversized episodes are usable but not retained.
            while self._cache and (
                len(self._cache) >= self.cache_episodes
                or (self.cache_bytes and self._cache_nbytes + size > self.cache_bytes)
            ):
                old, _ = self._cache.popitem(last=False)
                self._cache_nbytes -= self._cache_sizes.pop(old)
            self._cache[index] = frame
            self._cache_sizes[index] = size
            self._cache_nbytes += size
        self._cache.move_to_end(index)
        return self._cache[index]

    def __getitems__(self, indices):
        # Reuse even an oversized, uncached episode within the current batch.
        # This avoids decoding it once for every sample when it exceeds the cap.
        samples = []
        try:
            for index in indices:
                episode = int(np.searchsorted(self.offsets, index, side="right") - 1)
                active = getattr(self, "_active_batch_episode", None)
                if active is None or active[0] != episode:
                    self._active_batch_episode = None
                    self._active_batch_episode = (episode, self._episode(episode))
                samples.append(self[int(index)])
            return samples
        finally:
            self._active_batch_episode = None

    def __getstate__(self):
        # Spawn CPU workers without copying the main process's decoded example.
        state = self.__dict__.copy()
        state.update(_cache=OrderedDict(), _cache_sizes={}, _cache_nbytes=0)
        state["_active_batch_episode"] = None
        return state

    def __getitem__(self, index: int) -> RLTransition:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.offsets, index, side="right") - 1)
        start = int(index - self.offsets[episode])
        end = start + self.horizon
        frame, labels = self._episode(episode), self.annotations[episode]
        terminated, truncated = bool(labels.terminated[end - 1]), bool(labels.truncated[end - 1])
        mask = float(not terminated and (not truncated or self.bootstrap_on_truncation))
        current = extract_step_data(
            frame, start, self.modality_configs, self.embodiment_tag, allow_padding=True
        )
        valid_next = True
        if end < len(frame):
            following = extract_step_data(
                frame, end, self.observation_configs, self.embodiment_tag, allow_padding=True
            )
        elif labels.final_observation is not None:
            following = replace(labels.final_observation, actions={})
            if following.embodiment != self.embodiment_tag:
                raise ValueError("final_observation embodiment must match the dataset")
        else:
            if mask != 0:
                raise ValueError("Cannot bootstrap from a fabricated next observation")
            valid_next = False
            following = extract_step_data(
                frame,
                len(frame) - 1,
                self.observation_configs,
                self.embodiment_tag,
                allow_padding=True,
            )
        return RLTransition(
            observation=current,
            next_observation=following,
            rewards=labels.rewards[start:end].copy(),
            terminated=terminated,
            truncated=truncated,
            bootstrap_mask=mask,
            discount=self.bootstrap_gamma**self.horizon * mask,
            episode_index=episode,
            step_index=start,
            reward_gamma=self.gamma,
            next_observation_valid=valid_next,
        )


class OfflineRLConcatDataset(ConcatDataset):
    """Same-schema datasets with global episode IDs and transition-weighted sampling."""

    def __init__(self, datasets):
        datasets = list(datasets)
        if not datasets:
            raise ValueError("At least one offline dataset is required")
        first = datasets[0]
        for dataset in datasets[1:]:
            for key in (
                "embodiment_tag",
                "modality_configs",
                "horizon",
                "gamma",
                "bootstrap_gamma",
                "bootstrap_on_truncation",
            ):
                if getattr(dataset, key) != getattr(first, key):
                    raise ValueError(f"Offline dataset mixture must share {key}")
        super().__init__(datasets)
        self.episode_sizes = [size for ds in datasets for size in ds.episode_sizes]
        self.offsets = np.concatenate(([0], np.cumsum(self.episode_sizes)))
        self.annotations = [item for ds in datasets for item in ds.annotations]
        self.episode_offsets = np.cumsum([0, *[len(ds.episode_sizes) for ds in datasets]])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        dataset_index = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        sample = super().__getitem__(index)
        return replace(
            sample, episode_index=sample.episode_index + int(self.episode_offsets[dataset_index])
        )


class EpisodeBatchSampler(Sampler):
    """Episode-local random batches, weighted by valid transition count.

    Avoids repeatedly decoding different videos for every element of a batch.
    Samples with replacement; samples inside a batch are correlated. A regular
    PyTorch sampler or ConcatDataset can be substituted when resources permit.
    Each batch seed depends on its absolute step, making resume deterministic.
    """

    def __init__(self, dataset, batch_size, num_batches, seed=0, start_step=0):
        if batch_size < 1 or num_batches < 0 or start_step < 0:
            raise ValueError("Invalid sampler dimensions")
        self.dataset, self.batch_size = dataset, batch_size
        self.num_batches, self.seed, self.start_step = num_batches, seed, start_step

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        probabilities = np.asarray(self.dataset.episode_sizes, dtype=float) / len(self.dataset)
        for step in range(self.start_step, self.start_step + self.num_batches):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            episode = rng.choice(len(probabilities), p=probabilities)
            yield rng.integers(
                self.dataset.offsets[episode],
                self.dataset.offsets[episode + 1],
                size=self.batch_size,
            ).tolist()
