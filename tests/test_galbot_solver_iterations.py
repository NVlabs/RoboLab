# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solver limits after task registration and runtime configuration parsing."""

import pytest

from robolab.core.environments.config import parse_env_cfg
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
from robolab.registrations.galbot.auto_env_registrations_abs_ik import auto_register_galbot_abs_ik_envs
from robolab.registrations.galbot.auto_env_registrations_jointpos import auto_register_galbot_envs


@pytest.mark.parametrize(
    "register, kwargs, expected",
    [
        (auto_register_galbot_envs, {"action": "whole_body"}, (128, 4)),
        (auto_register_galbot_envs, {"action": "arms"}, (128, 4)),
        (auto_register_galbot_abs_ik_envs, {}, (128, 4)),
        (auto_register_droid_envs, {}, (32, 1)),
    ],
)
def test_registered_task_solver_limits(register, kwargs, expected):
    postfix = f"SolverLimits{register.__name__}{kwargs.get('action', '')}"
    if register is auto_register_droid_envs:
        register(task="BananaInBowlTask")
        postfix = ""
    else:
        register(task="BananaInBowlTask", env_postfix=postfix, **kwargs)
    cfg = parse_env_cfg(f"BananaInBowlTask{postfix}", num_envs=1)
    if expected == (128, 4):
        articulation = cfg.scene.robot.spawn.articulation_props
        assert articulation.solver_position_iteration_count == 128
        assert articulation.solver_velocity_iteration_count == 4
    for axis, count in zip(("position", "velocity"), expected):
        physx = cfg.sim.physx
        field = f"max_{axis}_iteration_count"
        if not hasattr(physx, field):
            field = f"num_{axis}_iterations"
        assert getattr(physx, field) == count
