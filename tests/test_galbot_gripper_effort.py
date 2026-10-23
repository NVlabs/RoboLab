# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from robolab.core.environments.config import parse_env_cfg
from robolab.registrations.galbot.auto_env_registrations_jointpos import auto_register_galbot_envs
from robolab.robots.galbot_golf import make_galbot_golf_tabletop_cfg


def test_gripper_effort_override_does_not_mutate_default_profile():
    tuned = make_galbot_golf_tabletop_cfg(gripper_effort=0.5)().robot
    default = make_galbot_golf_tabletop_cfg()().robot
    assert tuned.actuators["grippers"].effort_limit_sim == 0.5
    assert default.actuators["grippers"].effort_limit_sim == 1.5
    for field in ["stiffness", "damping", "velocity_limit_sim", "joint_names_expr"]:
        assert getattr(tuned.actuators["grippers"], field) == getattr(default.actuators["grippers"], field)
    for name in tuned.actuators:
        if name != "grippers":
            assert tuned.actuators[name].to_dict() == default.actuators[name].to_dict()


@pytest.mark.parametrize("effort", [0, -0.1, float("nan"), float("inf")])
def test_gripper_effort_must_be_positive_and_finite(effort):
    with pytest.raises(ValueError, match="finite positive"):
        make_galbot_golf_tabletop_cfg(gripper_effort=effort)


def test_tuned_registration_preserves_timing_and_solver_limits():
    postfix = "TunedGripperEffortTest"
    auto_register_galbot_envs(
        task="BananaInBowlTask",
        env_postfix=postfix,
        robot_cfg=make_galbot_golf_tabletop_cfg(gripper_effort=0.5),
        solver_type=1,
        dt=1 / 120,
        decimation=12,
        render_interval=12,
    )
    cfg = parse_env_cfg("BananaInBowlTask" + postfix, num_envs=1)
    assert cfg.scene.robot.actuators["grippers"].effort_limit_sim == 0.5
    assert cfg.sim.physx.solver_type == 1
    assert cfg.sim.dt * cfg.decimation == pytest.approx(0.1)
    assert cfg.sim.render_interval == 12
    assert cfg.sim.physx.max_position_iteration_count == 128
    assert cfg.sim.physx.max_velocity_iteration_count == 4
