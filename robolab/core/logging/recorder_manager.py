# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import gc
import os
from collections.abc import Sequence

import isaaclab.envs.manager_based_env as manager_based_env
import psutil
import torch
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg
from isaaclab.utils.datasets import EpisodeData

from robolab.constants import DEBUG
from robolab.core.logging.streaming_hdf5_handler import StreamingHDF5DatasetFileHandler


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_episode_data_size(episode: EpisodeData) -> dict:
    """Get size info about episode data buffer."""
    if episode.is_empty():
        return {"num_keys": 0, "total_elements": 0, "size_mb": 0.0}

    def count_elements(data, prefix=""):
        """Recursively count elements in nested dict of tensors."""
        info = {}
        total = 0
        for key, value in data.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                sub_info = count_elements(value, full_key)
                info.update(sub_info["details"])
                total += sub_info["total"]
            elif isinstance(value, torch.Tensor):
                num_elements = value.numel()
                size_mb = value.element_size() * num_elements / (1024 * 1024)
                info[full_key] = {"shape": list(value.shape), "elements": num_elements, "size_mb": size_mb}
                total += num_elements
        return {"details": info, "total": total}

    result = count_elements(episode.data)
    total_size_mb = sum(v.get("size_mb", 0) for v in result["details"].values())

    return {
        "num_keys": len(result["details"]),
        "total_elements": result["total"],
        "size_mb": total_size_mb,
        "details": result["details"]
    }


def patch_recorder_manager():
    """Monkey patch the RecorderManager in the base environment module."""
    # Replace the RecorderManager class with our CustomRecorderManager
    manager_based_env.RecorderManager = RobolabRecorderManager
    if DEBUG:
        print("[INFO] Patched RecorderManager with RobolabRecorderManager (with streaming support)")


class RobolabRecorderManager(RecorderManager):
    """Custom recorder manager with streaming HDF5 support for memory-efficient long episodes.

    This manager allows:
    - Incremental flushing of data to disk during an episode (via flush_buffer())
    - Proper episode finalization without creating multiple demos
    - Memory-efficient recording for episodes with thousands of steps
    """

    initialized = False

    def __init__(self, cfg: RecorderManagerBaseCfg, env: ManagerBasedEnv):
        """Initialize the recorder manager with streaming HDF5 support."""
        self._term_names: list[str] = list()
        self._terms: dict = dict()

        # Track streaming state per environment
        self._streaming_active: dict[int, bool] = {}

        # Do nothing if cfg is None or an empty dict
        if not cfg:
            return

        # Call grandparent's __init__ to skip RecorderManager's file handler setup
        from isaaclab.managers.manager_base import ManagerBase
        ManagerBase.__init__(self, cfg, env)

        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if not isinstance(cfg, RecorderManagerBaseCfg):
            raise TypeError("Configuration for the recorder manager is not of type RecorderManagerBaseCfg.")

        # create episode data buffer indexed by environment id
        self._episodes: dict[int, EpisodeData] = dict()
        for env_id in range(env.num_envs):
            self._episodes[env_id] = EpisodeData()
            self._streaming_active[env_id] = False

        self._env_name = getattr(env.cfg, "env_name", None)

        # HDF5 file handlers — created lazily on first write or when set_hdf5_file() is called.
        # This avoids creating empty data.hdf5 files that are never written to.
        self._dataset_file_handler = None
        self._failed_episode_dataset_file_handler = None
        self._hdf5_initialized = False

        self._exported_successful_episode_count = {}
        self._exported_failed_episode_count = {}

        # Track current episode index per environment (for overwriting existing episodes)
        self._current_episode_index: dict[int, int | None] = {}
        for env_id in range(env.num_envs):
            self._current_episode_index[env_id] = None

        # Auto-flush configuration
        self._flush_interval: int = 500  # Flush every N steps (0 = disabled)
        self._step_count: dict[int, int] = {}
        for env_id in range(env.num_envs):
            self._step_count[env_id] = 0

        self._auto_flush_verbose: bool = False  # Print diagnostics on auto-flush

        self.initialized = True

    def set_flush_interval(self, interval: int, verbose: bool = False):
        """Set the automatic flush interval.

        Args:
            interval: Number of steps between automatic flushes. Set to 0 to disable auto-flush.
            verbose: If True, print diagnostics when auto-flushing.
        """
        self._flush_interval = interval
        self._auto_flush_verbose = verbose
        if interval > 0:
            print(f"[RobolabRecorderManager] Auto-flush enabled: every {interval} steps")
        else:
            print(f"[RobolabRecorderManager] Auto-flush disabled")

    def _ensure_hdf5_handler(self, filename: str = None):
        """Create or switch the HDF5 file handler.

        Called lazily on first write or explicitly by set_hdf5_file().
        """
        cfg: RecorderManagerBaseCfg = self.cfg
        if cfg.dataset_export_mode == DatasetExportMode.EXPORT_NONE:
            return

        if filename is None:
            filename = cfg.dataset_filename

        # Close existing handler if switching files
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.close()
        else:
            self._dataset_file_handler = StreamingHDF5DatasetFileHandler()

        filepath = os.path.join(cfg.dataset_export_dir_path, filename)
        self._dataset_file_handler.create(filepath, env_name=self._env_name)

        if cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
            if self._failed_episode_dataset_file_handler is not None:
                self._failed_episode_dataset_file_handler.close()
            else:
                self._failed_episode_dataset_file_handler = StreamingHDF5DatasetFileHandler()
            failed_path = os.path.join(cfg.dataset_export_dir_path, f"{filename}_failed")
            self._failed_episode_dataset_file_handler.create(failed_path, env_name=self._env_name)

        self._hdf5_initialized = True

    def set_hdf5_file(self, filename: str):
        """Switch to a new HDF5 file. Closes the current file and opens/creates the new one.

        Args:
            filename: New filename (e.g. "run_0.hdf5"). Written to the same export directory.
        """
        self._ensure_hdf5_handler(filename)

    def set_episode_index(self, episode_index: int, env_ids: Sequence[int] | None = None):
        """Set the episode index to use for the next flush/export.

        If the episode already exists in the HDF5 file, it will be deleted and recreated.

        Args:
            episode_index: The episode index (demo_N) to use.
            env_ids: The environment ids. Defaults to None (all environments).
        """
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        for env_id in env_ids:
            self._current_episode_index[env_id] = episode_index

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the recorder data, but does not clear the buffer.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """

        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return {}

        # resolve environment ids
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))

        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        info = {}
        if hasattr(self, "_terms"):
            for term in self._terms.values():
                data = term.reset(env_ids=env_ids)
                info[term.__class__.__name__] = data

        # Disable clearing buffer - we handle this manually via clear()
        # for env_id in env_ids:
        #     self._episodes[env_id] = EpisodeData()

        return info

    def record_post_step(self) -> None:
        """Trigger recorder terms for post-step functions with auto-flush support.

        This overrides the parent method to add automatic flushing every N steps
        to prevent OOM on long episodes.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        # Call parent to record the data
        for term in self._terms.values():
            key, value = term.record_post_step()
            self.add_to_episodes(key, value)

        # Increment step count and check if we need to auto-flush
        if self._flush_interval > 0:
            for env_id in range(self._env.num_envs):
                self._step_count[env_id] += 1

                if self._step_count[env_id] >= self._flush_interval:
                    # Time to flush this environment's buffer
                    self.flush_buffer(env_ids=[env_id], verbose=self._auto_flush_verbose)
                    self._step_count[env_id] = 0

    def flush_buffer(self, env_ids: Sequence[int] | None = None, verbose: bool = True):
        """Flush the current buffer to disk without finalizing the episode.

        This writes the accumulated data to the HDF5 file and clears the in-memory
        buffer, but keeps the episode "open" so subsequent data gets appended to
        the same demo_N entry.

        Use this periodically during long episodes to prevent OOM errors.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
            verbose: If True, print diagnostic information about memory and HDF5 state.
        """
        # Do nothing if no active recorder terms
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # Lazy-init HDF5 on first write
        if not self._hdf5_initialized:
            self._ensure_hdf5_handler()

        for env_id in env_ids:
            if env_id not in self._episodes or self._episodes[env_id].is_empty():
                continue

            # Get target file handler based on export mode
            # Note: During streaming, we don't know success yet, so we use the main handler
            target_handler = self._dataset_file_handler
            if target_handler is None:
                continue

            # === BEFORE FLUSH DIAGNOSTICS ===
            if verbose:
                ram_before = get_memory_usage_mb()
                buffer_info = get_episode_data_size(self._episodes[env_id])
                hdf5_before = target_handler.get_episode_stats() if hasattr(target_handler, 'get_episode_stats') else {}

                print("\n" + "="*60)
                print(f"[FLUSH_BUFFER] env_id={env_id} - BEFORE FLUSH")
                print("="*60)
                print(f"  RAM Usage: {ram_before:.1f} MB")
                print(f"  In-Memory Buffer: {buffer_info['num_keys']} keys, {buffer_info['total_elements']:,} elements, {buffer_info['size_mb']:.2f} MB")
                if hdf5_before:
                    print(f"  HDF5: demo_{hdf5_before.get('demo_count', 0)}, {hdf5_before.get('current_samples', 0):,} samples")

            # Get episode index for this environment (if set)
            episode_index = self._current_episode_index.get(env_id)

            # Append data to the current episode (creates one if not started)
            # If episode_index is set and episode exists, it will be deleted and recreated
            target_handler.append_data(self._episodes[env_id], episode_index=episode_index)
            self._streaming_active[env_id] = True

            # Clear the in-memory buffer (but episode stays open in HDF5)
            self._episodes[env_id] = EpisodeData()

            # Force garbage collection to actually free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # === AFTER FLUSH DIAGNOSTICS ===
            if verbose:
                ram_after = get_memory_usage_mb()
                hdf5_after = target_handler.get_episode_stats() if hasattr(target_handler, 'get_episode_stats') else {}
                samples_added = hdf5_after.get('current_samples', 0) - hdf5_before.get('current_samples', 0)

                print(f"[FLUSH_BUFFER] AFTER: RAM {ram_after:.1f} MB (freed {ram_before - ram_after:.1f} MB), "
                      f"HDF5 demo_{hdf5_after.get('demo_count', 0)} now has {hdf5_after.get('current_samples', 0):,} samples (+{samples_added})")
                print("="*60 + "\n")

        # Clear any term buffers if they support it
        if hasattr(self, "_terms"):
            for term in self._terms.values():
                if hasattr(term, "clear"):
                    term.clear()

    def export_episodes(self, env_ids: Sequence[int] | None = None) -> None:
        """Concludes and exports the episodes for the given environment ids.

        If streaming is active, this finalizes the current demo. Otherwise,
        it writes the complete episode as a new demo (standard behavior).

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # Lazy-init HDF5 on first write
        if not self._hdf5_initialized:
            self._ensure_hdf5_handler()

        # Record final status for incomplete tasks BEFORE exporting
        # This ensures error codes are registered for stalled conditions
        for term in self._terms.values():
            if hasattr(term, 'record_final_status'):
                key, value = term.record_final_status()
                if key is not None and value is not None:
                    self.add_to_episodes(key, value)

        # Get export mode from config
        cfg: RecorderManagerBaseCfg = self.cfg  # type: ignore

        for env_id in env_ids:
            episode = self._episodes.get(env_id)
            is_streaming = self._streaming_active.get(env_id, False)

            # Determine target handler based on success (for non-streaming final export)
            episode_succeeded = episode.success if episode else None
            target_handler = None

            if (cfg.dataset_export_mode == DatasetExportMode.EXPORT_ALL) or (
                cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_ONLY and episode_succeeded
            ):
                target_handler = self._dataset_file_handler
            elif cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
                if episode_succeeded:
                    target_handler = self._dataset_file_handler
                else:
                    target_handler = self._failed_episode_dataset_file_handler

            if target_handler is None:
                continue

            # Get episode index for this environment (if set)
            episode_index = self._current_episode_index.get(env_id)

            if is_streaming:
                # Streaming mode: append any remaining data and finalize
                if episode and not episode.is_empty():
                    target_handler.append_data(episode, episode_index=episode_index)
                target_handler.end_episode(success=episode_succeeded, episode_index=episode_index)
                self._streaming_active[env_id] = False
            else:
                # Standard mode: write complete episode (with optional episode_index)
                if episode and not episode.is_empty():
                    if episode_index is not None:
                        target_handler.begin_episode(episode.seed, episode_index=episode_index)
                        target_handler.append_data(episode, episode_index=episode_index)
                        target_handler.end_episode(success=episode_succeeded, episode_index=episode_index)
                    else:
                        target_handler.write_episode(episode)

            # Reset episode index after export
            self._current_episode_index[env_id] = None

            # Update counts
            if episode_succeeded:
                if env_id not in self._exported_successful_episode_count:
                    self._exported_successful_episode_count[env_id] = 0
                self._exported_successful_episode_count[env_id] += 1
            else:
                if env_id not in self._exported_failed_episode_count:
                    self._exported_failed_episode_count[env_id] = 0
                self._exported_failed_episode_count[env_id] += 1

    def clear(self, env_ids: Sequence[int] | None = None):
        """Clear the buffer of the recorder manager.

        Note: If streaming is active, calling clear() without export_episodes()
        will leave an incomplete episode in the HDF5 file.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))

        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        if hasattr(self, "_episodes"):
            for env_id in env_ids:
                self._episodes[env_id] = EpisodeData()
                # Reset streaming state since we're clearing
                self._streaming_active[env_id] = False
                # Reset episode index
                if hasattr(self, "_current_episode_index"):
                    self._current_episode_index[env_id] = None
                # Reset step count for auto-flush
                if hasattr(self, "_step_count"):
                    self._step_count[env_id] = 0

        if hasattr(self, "_terms"):
            # Call clear on terms, if provided
            for term in self._terms.values():
                if hasattr(term, "clear"):
                    term.clear()

        return {}
