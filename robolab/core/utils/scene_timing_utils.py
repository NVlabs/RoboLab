# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import math
from typing import Any, Protocol, Union

from isaaclab.envs.manager_based_env import ManagerBasedEnv


def video_fps(env: ManagerBasedEnv) -> float:
    return 1 / (env.cfg.sim.render_interval * env.cfg.sim.dt) # Hz

def wall_time_to_steps(
    env: ManagerBasedEnv,
    wall_time_seconds: float,
) -> int:
    """Convert wall time in seconds to the number of IsaacLab environment steps.

    IsaacLab environments have two key timing parameters:
    - physics_dt: The physics simulation timestep (e.g., 1/60 ≈ 0.0167s)
    - decimation: Number of physics steps per environment step (e.g., 4 or 8)
    - env_dt = physics_dt * decimation (the actual environment step duration)

    Args:
        wall_time_seconds: The desired wall time duration in seconds
        env: IsaacLab environment object with .physics_dt and .step_dt properties.

    Returns:
        int: Number of environment steps needed to achieve the specified wall time

    Examples:
        # Using environment object
        >>> steps = wall_time_to_isaaclab_steps(1.0, env=my_env)

        # Using environment dt directly
        >>> steps = wall_time_to_isaaclab_steps(1.0, env_dt=0.033)  # 30 steps

        # Using physics parameters
        >>> steps = wall_time_to_isaaclab_steps(1.0, physics_dt=1/60, decimation=4)  # ~15 steps

    Raises:
        ValueError: If insufficient parameters are provided or if values are invalid
    """
    if wall_time_seconds < 0:
        raise ValueError(f"wall_time_seconds must be non-negative, got {wall_time_seconds}")

    if env is None:
        raise ValueError("Environment object must be provided")

    effective_dt = env.physics_dt * env.cfg.decimation
    num_steps = math.ceil(wall_time_seconds / effective_dt)
    return num_steps


def steps_to_wall_time(
    env: ManagerBasedEnv,
    num_steps: int,
) -> float:
    """Convert number of IsaacLab environment steps to wall time in seconds.

    This is the inverse operation of wall_time_to_isaaclab_steps.

    Args:
        env: IsaacLab environment object with .physics_dt and .step_dt properties.
        num_steps: Number of environment steps

    Returns:
        float: Wall time duration in seconds for the specified number of steps

    Examples:
        # Using environment object
        >>> time = isaaclab_steps_to_wall_time(30, env=my_env)

        # Using environment dt directly
        >>> time = isaaclab_steps_to_wall_time(30, env_dt=0.033)  # 0.99 seconds

        # Using physics parameters
        >>> time = isaaclab_steps_to_wall_time(15, physics_dt=1/60, decimation=4)  # ~1.0 seconds

    Raises:
        ValueError: If insufficient parameters are provided or if values are invalid
    """
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}")

    if env is None:
        raise ValueError("Environment object must be provided")

    effective_dt = env.physics_dt * env.cfg.decimation
    wall_time = num_steps * effective_dt
    return wall_time
