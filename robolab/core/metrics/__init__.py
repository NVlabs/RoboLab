# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Metrics module for trajectory analysis and quality evaluation.
"""

from robolab.core.metrics.compute_metrics import (
    compute_episode_metrics,
    compute_experiment_metrics,
    get_available_demos,
    load_demo_data,
    process_experiment_folder,
)
from robolab.core.metrics.trajectory_metrics import (
    compute_ee_isj_from_position,
    compute_ee_isj_from_velocity,
    compute_ee_path_length,
    compute_ee_sparc_from_position,
    compute_ee_sparc_from_velocity,
    compute_joint_isj_from_position,
    compute_joint_isj_from_velocity,
    compute_joint_isj_per_joint_from_position,
    compute_joint_isj_per_joint_from_velocity,
    compute_sparc,
    compute_sparc_from_velocity,
    compute_sparc_per_joint,
)

__all__ = [
    # Trajectory metrics (low-level)
    "compute_ee_isj_from_position",
    "compute_ee_isj_from_velocity",
    "compute_ee_path_length",
    "compute_ee_sparc_from_position",
    "compute_ee_sparc_from_velocity",
    "compute_joint_isj_from_position",
    "compute_joint_isj_from_velocity",
    "compute_joint_isj_per_joint_from_position",
    "compute_joint_isj_per_joint_from_velocity",
    "compute_sparc",
    "compute_sparc_from_velocity",
    "compute_sparc_per_joint",
    # Experiment metrics (high-level)
    "compute_experiment_metrics",
    "compute_episode_metrics",
    "process_experiment_folder",
    "load_demo_data",
    "get_available_demos",
]
