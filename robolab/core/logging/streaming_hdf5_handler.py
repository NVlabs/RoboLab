# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Streaming HDF5 dataset file handler that supports incremental writing with multiple concurrent episodes."""

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch
from isaaclab.utils.datasets import EpisodeData
from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase


@dataclass
class _OpenEpisode:
    """State for a single open (in-progress) episode in the HDF5 file."""
    group: h5py.Group
    datasets: dict = field(default_factory=dict)   # cache_key -> h5py.Dataset
    seed: int | None = None
    success: bool | None = None
    index: int = 0


class StreamingHDF5DatasetFileHandler(DatasetFileHandlerBase):
    """HDF5 dataset file handler that supports streaming/incremental writes.

    Supports multiple concurrent open episodes, allowing multi-env recording
    where each env flushes its buffer independently without interfering with
    other envs' open episodes.
    """

    def __init__(self):
        self._hdf5_file_stream = None
        self._hdf5_data_group = None
        self._demo_count = 0
        self._env_args = {}

        # Multiple concurrent open episodes, keyed by episode index
        self._open_episodes: dict[int, _OpenEpisode] = {}

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._hdf5_file_stream = h5py.File(file_path, mode)
        self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._demo_count = len(self._hdf5_data_group)

    def create(self, file_path: str, env_name: str = None):
        """Create a new dataset file, or open existing file to resume."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        if os.path.exists(file_path):
            print(f"[StreamingHDF5] Found existing file: {file_path}. Resuming...")
            try:
                self._hdf5_file_stream = h5py.File(file_path, "a")
            except OSError as e:
                print(f"\033[93m[StreamingHDF5] WARNING: Corrupt HDF5 file, cannot resume: {e}\033[0m")
                print(f"\033[93m[StreamingHDF5] WARNING: Removing corrupt file and creating a new one: {file_path}\033[0m")
                os.remove(file_path)
                self._hdf5_file_stream = h5py.File(file_path, "w")
                self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
                self._hdf5_data_group.attrs["total"] = 0
                self._demo_count = 0
                env_name = env_name if env_name is not None else ""
                self.add_env_args({"env_name": env_name, "type": 2})
                return

            if "data" in self._hdf5_file_stream:
                self._hdf5_data_group = self._hdf5_file_stream["data"]
                self._demo_count = len(self._hdf5_data_group)
                print(f"[StreamingHDF5] Loaded {self._demo_count} existing demos")

                if "env_args" in self._hdf5_data_group.attrs:
                    self._env_args = json.loads(self._hdf5_data_group.attrs["env_args"])
            else:
                self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
                self._hdf5_data_group.attrs["total"] = 0
                self._demo_count = 0
                env_name = env_name if env_name is not None else ""
                self.add_env_args({"env_name": env_name, "type": 2})
        else:
            self._hdf5_file_stream = h5py.File(file_path, "w")
            self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
            self._hdf5_data_group.attrs["total"] = 0
            self._demo_count = 0
            env_name = env_name if env_name is not None else ""
            self.add_env_args({"env_name": env_name, "type": 2})

    def __del__(self):
        self.close()

    # ========================================================================
    # Properties
    # ========================================================================

    def add_env_args(self, env_args: dict):
        self._raise_if_not_initialized()
        self._env_args.update(env_args)
        self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def set_env_name(self, env_name: str):
        self._raise_if_not_initialized()
        self.add_env_args({"env_name": env_name})

    def get_env_name(self) -> str | None:
        self._raise_if_not_initialized()
        env_args = json.loads(self._hdf5_data_group.attrs["env_args"])
        if "env_name" in env_args:
            return env_args["env_name"]
        return None

    def get_episode_names(self) -> Iterable[str]:
        self._raise_if_not_initialized()
        return self._hdf5_data_group.keys()

    def get_num_episodes(self) -> int:
        return self._demo_count

    @property
    def demo_count(self) -> int:
        return self._demo_count

    @property
    def has_active_episode(self) -> bool:
        """Check if there's any active episode being written."""
        return len(self._open_episodes) > 0

    def get_episode_stats(self) -> dict:
        """Get statistics about the current state of the handler."""
        stats = {
            "demo_count": self._demo_count,
            "current_samples": 0,
            "dataset_keys": [],
            "file_path": self._hdf5_file_stream.filename if self._hdf5_file_stream else None,
            "has_active_episode": self.has_active_episode,
            "open_episodes": list(self._open_episodes.keys()),
        }

        # Report stats for the first open episode (backward compat)
        if self._open_episodes:
            ep = next(iter(self._open_episodes.values()))
            stats["current_samples"] = ep.group.attrs.get("num_samples", 0)

            def get_keys(group, prefix=""):
                keys = []
                for key in group.keys():
                    full_key = f"{prefix}/{key}" if prefix else key
                    if hasattr(group[key], 'keys'):
                        keys.extend(get_keys(group[key], full_key))
                    else:
                        keys.append(full_key)
                return keys

            stats["dataset_keys"] = get_keys(ep.group)

        return stats

    # ========================================================================
    # Streaming Operations (multi-episode)
    # ========================================================================

    def begin_episode(self, seed: int = None, episode_index: int = None):
        """Begin a new episode for streaming writes.

        Multiple episodes can be open concurrently (one per episode_index).

        Args:
            seed: Optional seed for the episode.
            episode_index: Specific episode index. If None, uses demo_count.
        """
        self._raise_if_not_initialized()

        if episode_index is None:
            episode_index = self._demo_count

        # Already open — just return (idempotent)
        if episode_index in self._open_episodes:
            return

        demo_name = f"demo_{episode_index}"

        # If episode already exists on disk (from a previous run), delete it
        if demo_name in self._hdf5_data_group:
            print(f"[StreamingHDF5] Episode '{demo_name}' already exists on disk. Deleting and recreating...")
            del self._hdf5_data_group[demo_name]
            self._hdf5_file_stream.flush()

        group = self._hdf5_data_group.create_group(demo_name)
        group.attrs["num_samples"] = 0

        ep = _OpenEpisode(group=group, seed=seed, index=episode_index)
        if seed is not None:
            group.attrs["seed"] = seed

        self._open_episodes[episode_index] = ep

    def append_data(self, episode: EpisodeData, episode_index: int = None):
        """Append episode data to an open episode.

        If the episode is not yet open, it will be opened automatically.
        Multiple episodes can be appended to concurrently.

        Args:
            episode: The episode data buffer to append.
            episode_index: Which episode to append to. If None, uses demo_count.
        """
        self._raise_if_not_initialized()

        if episode.is_empty():
            return

        if episode_index is None:
            episode_index = self._demo_count

        # Auto-open if not already open
        if episode_index not in self._open_episodes:
            self.begin_episode(episode.seed, episode_index=episode_index)

        ep = self._open_episodes[episode_index]

        # Update seed/success if provided
        if episode.seed is not None:
            ep.seed = episode.seed
            ep.group.attrs["seed"] = episode.seed
        if episode.success is not None:
            ep.success = episode.success

        # Append all data
        for key, value in episode.data.items():
            self._append_to_dataset(ep.group, key, value, ep.datasets)

        # Update num_samples from actions shape
        if "actions" in episode.data:
            actions_key = f"{ep.group.name}/actions"
            if actions_key in ep.datasets:
                ep.group.attrs["num_samples"] = ep.datasets[actions_key].shape[0]

        self._hdf5_file_stream.flush()

    def end_episode(self, success: bool = None, episode_index: int = None):
        """Finalize an open episode.

        Args:
            success: Whether the episode was successful.
            episode_index: Which episode to finalize. If None, finalizes the
                first (or only) open episode for backward compatibility.
        """
        self._raise_if_not_initialized()

        if episode_index is None:
            # Backward compat: end the first open episode
            if not self._open_episodes:
                return
            episode_index = next(iter(self._open_episodes))

        ep = self._open_episodes.pop(episode_index, None)
        if ep is None:
            return

        # Set success attribute
        final_success = success if success is not None else ep.success
        if final_success is not None:
            ep.group.attrs["success"] = final_success

        # Update total samples count
        num_samples = ep.group.attrs.get("num_samples", 0)
        self._hdf5_data_group.attrs["total"] += num_samples

        # Update demo_count to track the highest index seen
        if episode_index >= self._demo_count:
            self._demo_count = episode_index + 1

        self._hdf5_file_stream.flush()

    # ========================================================================
    # Standard Operations (for compatibility)
    # ========================================================================

    def load_episode(self, episode_name: str, device: str) -> EpisodeData | None:
        """Load episode data from the file."""
        self._raise_if_not_initialized()
        if episode_name not in self._hdf5_data_group:
            return None
        episode = EpisodeData()
        h5_episode_group = self._hdf5_data_group[episode_name]

        def load_dataset_helper(group):
            data = {}
            for key in group:
                if isinstance(group[key], h5py.Group):
                    data[key] = load_dataset_helper(group[key])
                else:
                    data[key] = torch.tensor(np.array(group[key]), device=device)
            return data

        episode.data = load_dataset_helper(h5_episode_group)

        if "seed" in h5_episode_group.attrs:
            episode.seed = h5_episode_group.attrs["seed"]
        if "success" in h5_episode_group.attrs:
            episode.success = h5_episode_group.attrs["success"]

        episode.env_id = self.get_env_name()
        return episode

    def write_episode(self, episode: EpisodeData):
        """Write a complete episode (standard non-streaming mode)."""
        self._raise_if_not_initialized()
        if episode.is_empty():
            return
        self.begin_episode(episode.seed)
        idx = next(iter(self._open_episodes))  # get the just-opened index
        self.append_data(episode, episode_index=idx)
        self.end_episode(episode.success, episode_index=idx)

    def flush(self):
        self._raise_if_not_initialized()
        self._hdf5_file_stream.flush()

    def close(self):
        """Close the dataset file handler and reset internal state."""
        # Finalize any open episodes before closing
        for idx in list(self._open_episodes.keys()):
            self.end_episode(episode_index=idx)

        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.close()
            self._hdf5_file_stream = None
            self._hdf5_data_group = None
            self._open_episodes.clear()
            self._demo_count = 0

    # ========================================================================
    # Internal helpers
    # ========================================================================

    @staticmethod
    def _append_to_dataset(group, key, value, datasets_cache):
        """Append data to a resizable HDF5 dataset, creating it if needed."""
        if isinstance(value, dict):
            if key not in group:
                key_group = group.create_group(key)
            else:
                key_group = group[key]
            for sub_key, sub_value in value.items():
                StreamingHDF5DatasetFileHandler._append_to_dataset(
                    key_group, sub_key, sub_value, datasets_cache
                )
        else:
            np_data = value.cpu().numpy()
            cache_key = f"{group.name}/{key}"

            if cache_key in datasets_cache:
                dataset = datasets_cache[cache_key]
                old_size = dataset.shape[0]
                new_size = old_size + np_data.shape[0]
                dataset.resize(new_size, axis=0)
                dataset[old_size:new_size] = np_data
            else:
                maxshape = (None,) + np_data.shape[1:]
                dataset = group.create_dataset(
                    key,
                    data=np_data,
                    maxshape=maxshape,
                    chunks=True,
                    compression="gzip"
                )
                datasets_cache[cache_key] = dataset

    def _raise_if_not_initialized(self):
        if self._hdf5_file_stream is None:
            raise RuntimeError("HDF5 dataset file stream is not initialized")
