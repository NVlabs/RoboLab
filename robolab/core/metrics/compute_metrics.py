# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Compute trajectory metrics from experiment data.

This module provides functions to compute trajectory quality metrics from HDF5 data files
and save them to episode_metrics.json. Designed to be called after experiment completion.

Metrics computed:
- ee_sparc: End-effector SPARC smoothness (more negative = smoother)
- joint_sparc_mean: Mean SPARC across arm joints
- ee_isj: End-effector Integrated Squared Jerk
- joint_isj: Joint space Integrated Squared Jerk
- ee_path_length: End-effector path length in meters
- joint_rmse_mean: Mean joint tracking error (action vs actual)
- ee_speed_max: Maximum end-effector speed
- ee_speed_mean: Mean end-effector speed
"""

import json
import os
from typing import Any

import h5py
import numpy as np

from robolab.core.metrics.trajectory_metrics import (
    compute_ee_isj_from_position,
    compute_ee_isj_from_velocity,
    compute_ee_path_length,
    compute_ee_sparc_from_position,
    compute_ee_sparc_from_velocity,
    compute_joint_isj_from_velocity,
    compute_sparc_per_joint,
)

# Default timestep (15 Hz)
DEFAULT_DT = 1.0 / 15.0

# Number of arm joints
NUM_ARM_JOINTS = 7


def load_demo_data(hdf5_path: str, demo_key: str = "demo_0") -> dict | None:
    """
    Load trajectory data from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file
        demo_key: Key for the demo to load (default: "demo_0")

    Returns:
        Dictionary containing trajectory data, or None if demo doesn't exist.
        Includes 'ee_linear_velocity' if available in the HDF5 file.
    """
    data = {}

    try:
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f or demo_key not in f["data"]:
                return None

            demo = f["data"][demo_key]

            # Load actions
            data["actions"] = demo["actions"][:]

            # Load robot state
            robot_state = demo["states"]["articulation"]["robot"]
            data["joint_position"] = robot_state["joint_position"][:]
            data["joint_velocity"] = robot_state["joint_velocity"][:]

            # Load end-effector pose
            data["ee_position"] = demo["ee_pose"]["position"][:]
            data["ee_orientation"] = demo["ee_pose"]["orientation"][:]

            # Load end-effector linear velocity if available
            if "linear_velocity" in demo["ee_pose"]:
                data["ee_linear_velocity"] = demo["ee_pose"]["linear_velocity"][:]

    except Exception as e:
        print(f"Error loading {hdf5_path}/{demo_key}: {e}")
        return None

    return data


def get_available_demos(hdf5_path: str) -> list:
    """Get list of available demo keys in the HDF5 file."""
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                return []
            return list(f["data"].keys())
    except Exception:
        return []


def compute_episode_metrics(
    data: dict,
    dt: float = DEFAULT_DT,
    compute_ee_sparc: bool = True,
    compute_joint_sparc: bool = True,
    compute_ee_isj: bool = True,
    compute_joint_isj: bool = True,
    compute_path_length: bool = True,
    compute_joint_rmse: bool = True,
    compute_speed_stats: bool = True,
) -> dict | None:
    """
    Compute trajectory metrics for a single episode.

    Args:
        data: Dictionary containing trajectory data from load_demo_data
        dt: Time step between samples (seconds)
        compute_*: Flags to enable/disable specific metrics

    Returns:
        Dictionary of computed metrics, or None if data is too short
    """
    metrics = {}

    ee_position = data["ee_position"]
    joint_velocity = data["joint_velocity"][:, :NUM_ARM_JOINTS]
    actions = data["actions"]
    joint_position = data["joint_position"][:, :NUM_ARM_JOINTS]

    # Check if ee_linear_velocity is available (preferred) or compute from position
    has_ee_velocity = "ee_linear_velocity" in data and data["ee_linear_velocity"] is not None
    if has_ee_velocity:
        ee_velocity = data["ee_linear_velocity"]
    else:
        ee_velocity = None  # Will compute from position when needed

    # Check minimum data length for gradient computation (need at least 3 points)
    min_length = 3
    if len(ee_position) < min_length:
        print(f"Warning: Episode too short ({len(ee_position)} samples, need >= {min_length}) - skipping metrics")
        return None

    # Compute velocity from position if not available from HDF5
    if ee_velocity is None:
        ee_velocity = np.gradient(ee_position, dt, axis=0)

    # Compute speed for EE
    ee_speed = np.linalg.norm(ee_velocity, axis=1)

    # EE SPARC (smoothness) - use velocity if available, otherwise from position
    if compute_ee_sparc:
        if has_ee_velocity:
            metrics["ee_sparc"] = float(compute_ee_sparc_from_velocity(ee_velocity, dt))
        else:
            metrics["ee_sparc"] = float(compute_ee_sparc_from_position(ee_position, dt))

    # Joint SPARC (mean across joints)
    if compute_joint_sparc:
        sparc_per_joint = compute_sparc_per_joint(joint_velocity, dt)
        metrics["joint_sparc_mean"] = float(np.mean(sparc_per_joint))

    # EE Integrated Squared Jerk - use velocity if available, otherwise from position
    if compute_ee_isj:
        if has_ee_velocity:
            metrics["ee_isj"] = float(compute_ee_isj_from_velocity(ee_velocity, dt))
        else:
            metrics["ee_isj"] = float(compute_ee_isj_from_position(ee_position, dt))

    # Joint ISJ
    if compute_joint_isj:
        metrics["joint_isj"] = float(compute_joint_isj_from_velocity(joint_velocity, dt))

    # EE Path Length
    if compute_path_length:
        metrics["ee_path_length"] = float(compute_ee_path_length(ee_position))

    # Joint tracking RMSE (mean across joints)
    if compute_joint_rmse:
        # Only compare first 7 dims of actions (joint positions) with actual joint positions
        action_joints = actions[:, :NUM_ARM_JOINTS]
        # Cast to float64 to avoid overflow when squaring errors
        errors = np.asarray(action_joints - joint_position, dtype=np.float64)
        rmse_per_joint = np.sqrt(np.mean(errors**2, axis=0))
        metrics["joint_rmse_mean"] = float(np.mean(rmse_per_joint))

    # Speed statistics
    if compute_speed_stats:
        metrics["ee_speed_max"] = float(np.max(ee_speed))
        metrics["ee_speed_mean"] = float(np.mean(ee_speed))

    return metrics


def _load_file(filepath: str) -> Any:
    """Load JSON file, returns None if file doesn't exist or is invalid."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _load_timestep_from_config(config_path: str) -> float | None:
    """Load timestep from env_cfg.json config file."""
    try:
        config = _load_file(config_path)
        if config:
            sim = config.get("sim", {})
            dt = sim.get("dt")
            decimation = config.get("decimation")
            if dt is not None and decimation is not None:
                return dt * decimation
    except Exception:
        pass
    return None


def process_experiment_folder(
    folder_path: str,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Process a single experiment folder and compute trajectory metrics.

    Reads episode_results.json, computes metrics from HDF5 data for each episode,
    and saves results to episode_metrics.json.

    Args:
        folder_path: Path to the experiment folder containing episode_results.json
        overwrite: If True, recompute metrics even if they exist
        verbose: If True, print progress information

    Returns:
        List of episode dictionaries with metrics added
    """
    episode_metrics_file = os.path.join(folder_path, "episode_metrics.json")

    # Load existing episode results (supports both .jsonl and legacy .json)
    from robolab.core.logging.results import load_episode_results
    episode_results = load_episode_results(folder_path)
    if not episode_results:
        if verbose:
            print(f"Warning: Could not load episode results from {folder_path}")
        return []

    # Load existing metrics if not overwriting
    existing_metrics = {}
    if not overwrite and os.path.exists(episode_metrics_file):
        existing_data = _load_file(episode_metrics_file)
        if existing_data:
            # Index by (env_name, episode) for quick lookup
            for ep in existing_data:
                key = (ep.get("env_name"), ep.get("episode"))
                existing_metrics[key] = ep

    # Process each episode
    processed_episodes = []
    skipped_count = 0
    computed_count = 0
    failed_count = 0

    for ep in episode_results:
        # Try multiple folder name candidates
        run_name = ep.get("run_name") or ep.get("run")
        env_name = ep.get("env_name")
        episode_num = ep.get("episode")

        # For display/grouping, prefer env_name
        display_name = env_name or run_name
        key = (display_name, episode_num)

        # Start with existing data or copy from episode_results
        if key in existing_metrics and not overwrite:
            # Check if metrics already computed
            existing = existing_metrics[key]
            has_metrics = any(k in existing for k in [
                "ee_sparc", "joint_sparc_mean", "ee_isj", "joint_isj",
                "ee_path_length", "joint_rmse_mean", "ee_speed_max", "ee_speed_mean"
            ])
            if has_metrics:
                processed_episodes.append(existing)
                skipped_count += 1
                continue

        # Copy base episode data
        ep_with_metrics = ep.copy()

        # Try to find HDF5 file - supports both run_*.hdf5 (multi-env) and data.hdf5 (legacy)
        folder_candidates = []
        if env_name:
            folder_candidates.append(env_name)
        if run_name and run_name != env_name:
            folder_candidates.append(run_name)

        run_idx = ep.get("run")
        env_id = ep.get("env_id")
        hdf5_path = None
        demo_key = None

        for candidate in folder_candidates:
            candidate_dir = os.path.join(folder_path, candidate)
            if not os.path.isdir(candidate_dir):
                continue
            # Multi-env: run_{run_idx}.hdf5 with demo_{env_id}
            if run_idx is not None:
                run_path = os.path.join(candidate_dir, f"run_{run_idx}.hdf5")
                if os.path.exists(run_path):
                    hdf5_path = run_path
                    demo_key = f"demo_{env_id}" if env_id is not None else f"demo_{episode_num}"
                    break
            # Legacy: data.hdf5 with demo_{episode_num}
            legacy_path = os.path.join(candidate_dir, "data.hdf5")
            if os.path.exists(legacy_path):
                hdf5_path = legacy_path
                demo_key = f"demo_{episode_num}"
                break

        if hdf5_path is None:
            if verbose:
                print(f"Warning: HDF5 file not found for {display_name} episode {episode_num}")
            processed_episodes.append(ep_with_metrics)
            failed_count += 1
            continue

        # Check if demo exists before trying to load
        available_demos = get_available_demos(hdf5_path)
        if demo_key not in available_demos:
            if verbose:
                print(f"Warning: {demo_key} not found in {hdf5_path} (available: {available_demos})")
            processed_episodes.append(ep_with_metrics)
            failed_count += 1
            continue

        data = load_demo_data(hdf5_path, demo_key)
        if data is None:
            if verbose:
                print(f"Warning: Could not load {demo_key} from {hdf5_path} (data error)")
            processed_episodes.append(ep_with_metrics)
            failed_count += 1
            continue

        # Get timestep from config
        config_path = os.path.join(folder_path, folder_name, "env_cfg.json")
        dt = _load_timestep_from_config(config_path)
        if dt is None:
            dt = DEFAULT_DT

        # Compute all metrics
        metrics = compute_episode_metrics(data, dt=dt)

        # Add metrics to episode data (if computed successfully)
        if metrics is not None and metrics:  # metrics is a non-empty dict
            ep_with_metrics.update(metrics)
            computed_count += 1
        elif metrics is not None:  # metrics is empty dict (episode too short)
            failed_count += 1
        ep_with_metrics["dt"] = dt

        processed_episodes.append(ep_with_metrics)

    # Save to episode_metrics.json
    if processed_episodes:
        with open(episode_metrics_file, "w") as f:
            json.dump(processed_episodes, f, indent=2)
        if verbose:
            print(f"Saved {len(processed_episodes)} episodes to: {episode_metrics_file}")
            print(f"  - Computed metrics: {computed_count}")
            print(f"  - Skipped (existing): {skipped_count}")
            print(f"  - Failed/skipped (data issues): {failed_count}")
    elif verbose:
        print("No episodes to save.")

    return processed_episodes


def compute_experiment_metrics(
    output_dir: str,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Compute trajectory metrics for an experiment and save to episode_metrics.json.

    This is the main entry point for computing metrics after an experiment completes.
    Call this after summarize_experiment_results() in run_eval*.py scripts.

    Args:
        output_dir: Path to the experiment output directory (contains episode_results.json)
        overwrite: If True, recompute metrics even if episode_metrics.json exists
        verbose: If True, print progress information

    Returns:
        List of episode dictionaries with computed metrics

    Example:
        >>> from robolab.core.metrics import compute_experiment_metrics
        >>> compute_experiment_metrics(output_dir)
    """
    if not os.path.exists(output_dir):
        if verbose:
            print(f"Warning: Output directory not found: {output_dir}")
        return []

    if verbose:
        print(f"\nComputing trajectory metrics for: {output_dir}")

    return process_experiment_folder(output_dir, overwrite=overwrite, verbose=verbose)
