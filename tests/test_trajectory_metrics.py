# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from robolab.core.metrics.trajectory_metrics import (
    compute_ee_isj_from_position,
    compute_ee_isj_from_velocity,
    compute_joint_isj_from_position,
    compute_joint_isj_from_velocity,
    compute_joint_isj_per_joint_from_position,
    compute_joint_isj_per_joint_from_velocity,
)


def test_isj_aggregation_uses_numpy_version_independent_integration() -> None:
    """Joint and end-effector ISJ agree with the per-axis decomposition."""
    time = np.linspace(0.0, 1.0, 16)
    trajectory = np.stack((time**3, np.sin(time), np.cos(time)), axis=1)
    dt = float(time[1] - time[0])

    position_per_axis = compute_joint_isj_per_joint_from_position(trajectory, dt)
    velocity = np.gradient(trajectory, dt, axis=0)
    velocity_per_axis = compute_joint_isj_per_joint_from_velocity(velocity, dt)

    np.testing.assert_allclose(
        compute_joint_isj_from_position(trajectory, dt),
        position_per_axis.sum(),
    )
    np.testing.assert_allclose(
        compute_joint_isj_from_velocity(velocity, dt),
        velocity_per_axis.sum(),
    )
    np.testing.assert_allclose(
        compute_ee_isj_from_position(trajectory, dt),
        position_per_axis.sum(),
    )
    np.testing.assert_allclose(
        compute_ee_isj_from_velocity(velocity, dt),
        velocity_per_axis.sum(),
    )
