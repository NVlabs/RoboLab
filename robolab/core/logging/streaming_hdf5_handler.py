# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming HDF5 dataset file handler that supports incremental writing with multiple concurrent episodes."""

import atexit
import importlib.metadata
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

import h5py
import numpy as np
import torch
from isaaclab.utils.datasets import EpisodeData
from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase

from robolab.core.utils.hardware_utils import GPU_INFO_KEYS, get_gpu_info


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
        self._recording_info: dict[str, str] = {}

        # Multiple concurrent open episodes, keyed by episode index
        self._open_episodes: dict[int, _OpenEpisode] = {}

        # Best-effort: finalize open episodes and close the file on interpreter
        # exit (covers KeyboardInterrupt and most unhandled exceptions). __del__
        # is non-deterministic and can miss this on crash.
        atexit.register(self._cleanup_at_exit)

    def _cleanup_at_exit(self):
        """Finalize any still-open episodes and close the file on exit."""
        if self._hdf5_file_stream is None:
            return
        for idx in list(self._open_episodes.keys()):
            try:
                self.end_episode(episode_index=idx)
            except Exception:
                pass  # interpreter is already exiting; nothing useful to do
        try:
            self._hdf5_file_stream.close()
        except Exception:
            pass

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._hdf5_file_stream = h5py.File(file_path, mode)
        self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._demo_count = len(self._hdf5_data_group)

    def create(self, file_path: str, env_name: str = None, gpu_info: Mapping[str, str] | None = None):
        """Create a new dataset file, or open an existing file to resume.

        Args:
            file_path: Output HDF5 path.
            env_name: Optional environment name stored in ``env_args``.
            gpu_info: Information about the simulation GPU. When omitted, the
                current CUDA device is queried on a best-effort basis. Pass an
                empty mapping for a CPU simulation.
        """
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        queried_info = get_gpu_info() if gpu_info is None else gpu_info
        self._recording_info = {}
        for key in GPU_INFO_KEYS:
            value = queried_info.get(key)
            if value is not None and str(value).strip():
                self._recording_info[key] = str(value).strip()
        for package in ("isaaclab", "isaacsim"):
            try:
                self._recording_info[f"{package}_version"] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        self._recording_info["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
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
                self._init_data_group(env_name)
                return

            if "data" in self._hdf5_file_stream:
                self._hdf5_data_group = self._hdf5_file_stream["data"]
                self._demo_count = len(self._hdf5_data_group)
                print(f"[StreamingHDF5] Loaded {self._demo_count} existing demos")

                if "env_args" in self._hdf5_data_group.attrs:
                    self._env_args = json.loads(self._hdf5_data_group.attrs["env_args"])
            else:
                self._init_data_group(env_name)
        else:
            self._hdf5_file_stream = h5py.File(file_path, "w")
            self._init_data_group(env_name)

    def _init_data_group(self, env_name: str | None):
        """Create the ``data`` group in a fresh file and stamp recording provenance.

        The recording information (simulator stack versions, GPU details, and
        recording date) lets replay tooling and reviewers warn when a recording
        is replayed on a different stack or GPU, whose contact mechanics differ.
        """
        self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
        self._hdf5_data_group.attrs["total"] = 0
        self._demo_count = 0
        self.add_env_args({"env_name": env_name if env_name is not None else "", "type": 2})
        for key, value in self._recording_info.items():
            self._hdf5_data_group.attrs[key] = value

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

    def begin_episode(self, seed: int = None, episode_index: int = None) -> int:
        """Begin a new episode for streaming writes.

        Multiple episodes can be open concurrently (one per episode_index).

        Args:
            seed: Optional seed for the episode.
            episode_index: Specific episode index. If None, uses demo_count.

        Returns:
            The episode index that was opened (caller-supplied or derived from demo_count).
        """
        self._raise_if_not_initialized()

        if episode_index is None:
            episode_index = self._demo_count

        # Already open — just return (idempotent)
        if episode_index in self._open_episodes:
            return episode_index

        demo_name = f"demo_{episode_index}"

        # If episode already exists on disk (from a previous run), delete it
        if demo_name in self._hdf5_data_group:
            print(f"[StreamingHDF5] Episode '{demo_name}' already exists on disk. Deleting and recreating...")
            del self._hdf5_data_group[demo_name]
            self._hdf5_file_stream.flush()

        group = self._hdf5_data_group.create_group(demo_name)
        group.attrs["num_samples"] = 0
        # A resumed file can receive episodes from a different simulator stack,
        # machine, or GPU. Per-episode attrs describe newly written episodes;
        # readers fall back to file-level attrs for older recordings.
        for key, value in self._recording_info.items():
            group.attrs[key] = value

        ep = _OpenEpisode(group=group, seed=seed, index=episode_index)
        if seed is not None:
            group.attrs["seed"] = seed

        self._open_episodes[episode_index] = ep
        return episode_index

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
                single open episode; raises if multiple are open.
        """
        self._raise_if_not_initialized()

        if episode_index is None:
            if not self._open_episodes:
                return
            if len(self._open_episodes) > 1:
                raise RuntimeError(
                    f"end_episode() called without episode_index but "
                    f"{len(self._open_episodes)} episodes are open: "
                    f"{list(self._open_episodes)}. Specify episode_index explicitly."
                )
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
        idx = self.begin_episode(episode.seed)
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

    # Keys already warned about to avoid per-step log spam.
    _warned_nontensor_keys: set = set()

    @staticmethod
    def _leaf_to_numpy(group, key, value):
        """Convert a recorder leaf value to a numpy array for HDF5 storage.

        Leaves are normally CUDA/CPU torch tensors, but on the IsaacSim 5.1 /
        IsaacLab 2.3 stack some ``initial_state`` leaves arrive as Python lists
        (or lists of tensors) rather than a single stacked tensor. Coerce those
        so export doesn't crash with ``'list' object has no attribute 'cpu'``.
        """
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()

        # Dedup the warning across demos: group.name carries a per-episode
        # ".../demo_N/..." segment, so key on the field path with that stripped.
        field_path = "/".join(p for p in f"{group.name}/{key}".split("/") if not p.startswith("demo_"))
        if field_path not in StreamingHDF5DatasetFileHandler._warned_nontensor_keys:
            StreamingHDF5DatasetFileHandler._warned_nontensor_keys.add(field_path)
            print(f"[StreamingHDF5] non-tensor recorder leaf at {field_path}: "
                  f"type={type(value).__name__}; coercing to numpy.")

        if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            return torch.stack(list(value)).detach().cpu().numpy()
        return np.asarray(value)

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
            np_data = StreamingHDF5DatasetFileHandler._leaf_to_numpy(group, key, value)
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
