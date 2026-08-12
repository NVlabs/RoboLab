# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal runtime coverage for fixed-base Galbot registration and controls."""

import pytest
import torch
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.stage import get_current_stage

from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env
from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
from robolab.core.task.predicate_logic import gripper_detached
from robolab.core.world.world_state import get_world
from robolab.registrations.galbot.auto_env_registrations_abs_ik import auto_register_galbot_abs_ik_envs
from robolab.registrations.galbot.auto_env_registrations_jointpos import auto_register_galbot_envs
from robolab.robots.galbot_golf import (
    GALBOT_GOLF_ROOT_Z_ABOVE_GROUND,
    GALBOT_GOLF_TABLETOP_ROOT_POS,
    GalbotGolfTabletopCfg,
)
from robolab.robots.galbot_golf_definitions import ARM_JOINTS, HEAD_JOINTS, LEFT_EE_BODY, LEG_JOINTS, RIGHT_EE_BODY

# Tabletop Golf plus a contact sensor on the passive caster wheels — the
# robot's lowest colliders, rebased onto each scene's authored ground by the
# root_z_above_ground label. Built via type() so the sensor is added without
# re-decorating the label-carrying base class in a new class body.
_WHEEL_PROBE_CFG = configclass(
    type(
        "GalbotGolfWheelProbeCfg",
        (GalbotGolfTabletopCfg,),
        {"wheel_contact": ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot/wheel_.*")},
    )
)


def _registered_env(postfix: str) -> str:
    matches = [name for name in get_envs(task="BananaInBowlTask") if name.endswith(postfix)]
    assert len(matches) == 1
    return matches[0]


def _assert_pedestal_removed() -> None:
    pedestal = get_current_stage().GetPrimAtPath("/World/envs/env_0/scene/franka_table")
    assert pedestal.IsValid()
    assert not pedestal.IsActive()


def _assert_per_arm_ee_recorders(recorders) -> None:
    """Bimanual label: one EE-pose channel per arm, no single-arm channel."""
    assert not hasattr(recorders, "record_ee_pose")
    assert recorders.record_left_ee_pose.ee_body_name == LEFT_EE_BODY
    assert recorders.record_left_ee_pose.record_key == "left_ee_pose"
    assert recorders.record_right_ee_pose.ee_body_name == RIGHT_EE_BODY
    assert recorders.record_right_ee_pose.record_key == "right_ee_pose"


def test_arm_only_joint_control_holds_uncommanded_joints():
    postfix = "GalbotGolfFixedBaseArmsTest"
    auto_register_galbot_envs(
        task="BananaInBowlTask",
        action="arms",
        env_postfix=postfix,
        include_viewport=False,
        robot_cfg=_WHEEL_PROBE_CFG,
    )
    env, _ = create_env(_registered_env(postfix), num_envs=1, use_fabric=True)
    try:
        env.reset()
        robot = env.scene["robot"]
        _assert_pedestal_removed()
        _assert_per_arm_ee_recorders(env.cfg.recorders)
        assert float(robot.data.default_root_state[0, 2]) == pytest.approx(GALBOT_GOLF_TABLETOP_ROOT_POS[2])

        held_joint_ids = [robot.data.joint_names.index(name) for name in LEG_JOINTS + HEAD_JOINTS]
        torch.testing.assert_close(
            robot.data.joint_pos_target[:, held_joint_ids],
            robot.data.default_joint_pos[:, held_joint_ids],
        )

        arm_joint_ids = [robot.data.joint_names.index(name) for name in ARM_JOINTS]
        action = torch.cat(
            (
                robot.data.joint_pos[:, arm_joint_ids],
                torch.zeros((1, 2), device=env.device),
            ),
            dim=1,
        )
        assert action.shape == (1, 16)
        max_wheel_force = 0.0
        for _ in range(5):
            obs, *_ = env.step(action)
            wheel_forces = env.scene["wheel_contact"].data.net_forces_w
            max_wheel_force = max(max_wheel_force, float(wheel_forces.norm(dim=-1).max()))
        assert obs["proprio_obs"]["joint_pos"].shape == (1, 21)
        # Wheels rest on the -0.697 ground with no penetration: standing
        # contact forces must stay near zero (interpenetration shows up as
        # meganewton-scale depenetration forces against the welded root).
        assert max_wheel_force < 50.0, f"standing wheel contact force {max_wheel_force:.1f} N"
    finally:
        env.close()


def test_root_follows_legacy_scene_ground_without_wheel_contact():
    """In a legacy -0.65 scene the root rides the ground up 47 mm, wheels resting force-free.

    Regression test for the ground-height coupling: a fixed canonical root in a
    legacy scene embeds the wheels 47 mm in the floor (~2.4 MN standing force).
    """
    postfix = "GalbotGolfLegacyGroundTest"
    auto_register_galbot_envs(
        task="RubiksCubeAndBananaTask",
        action="arms",
        env_postfix=postfix,
        include_viewport=False,
        robot_cfg=_WHEEL_PROBE_CFG,
    )
    matches = [name for name in get_envs(task="RubiksCubeAndBananaTask") if name.endswith(postfix)]
    assert len(matches) == 1
    env, _ = create_env(matches[0], num_envs=1, use_fabric=True)
    try:
        env.reset()
        robot = env.scene["robot"]
        assert float(robot.data.default_root_state[0, 2]) == pytest.approx(-0.65 + GALBOT_GOLF_ROOT_Z_ABOVE_GROUND)

        arm_joint_ids = [robot.data.joint_names.index(name) for name in ARM_JOINTS]
        action = torch.cat(
            (robot.data.joint_pos[:, arm_joint_ids], torch.zeros((1, 2), device=env.device)),
            dim=1,
        )
        max_wheel_force = 0.0
        for _ in range(5):
            env.step(action)
            wheel_forces = env.scene["wheel_contact"].data.net_forces_w
            max_wheel_force = max(max_wheel_force, float(wheel_forces.norm(dim=-1).max()))
        assert max_wheel_force < 50.0, f"standing wheel contact force {max_wheel_force:.1f} N"
    finally:
        env.close()


def test_absolute_ik_accepts_observed_root_relative_tcp_pose():
    postfix = "GalbotGolfFixedBaseIKTest"
    auto_register_galbot_abs_ik_envs(
        task="BananaInBowlTask",
        env_postfix=postfix,
        include_viewport=False,
    )
    env, _ = create_env(_registered_env(postfix), num_envs=1, use_fabric=True)
    try:
        obs, _ = env.reset()
        _assert_pedestal_removed()
        _assert_per_arm_ee_recorders(env.cfg.recorders)
        proprio = obs["proprio_obs"]
        gripper_open = torch.zeros((1, 1), device=env.device)
        action = torch.cat(
            (
                proprio["left_ee_pos"],
                proprio["left_ee_quat"],
                gripper_open,
                proprio["right_ee_pos"],
                proprio["right_ee_quat"],
                gripper_open,
            ),
            dim=1,
        )
        assert action.shape == (1, 16)
        next_obs, *_ = env.step(action)
        assert next_obs["proprio_obs"]["joint_pos"].shape == (1, 21)
    finally:
        env.close()


def test_gripper_alias_group_resolves_to_either_hand():
    postfix = "GalbotGolfFixedBaseContactTest"
    auto_register_galbot_envs(
        task="BananaInBowlTask",
        action="arms",
        env_postfix=postfix,
        include_viewport=False,
    )
    env, _ = create_env(_registered_env(postfix), num_envs=1, use_fabric=True)
    try:
        env.reset()
        robot = env.scene["robot"]
        arm_joint_ids = [robot.data.joint_names.index(name) for name in ARM_JOINTS]
        action = torch.cat(
            (robot.data.joint_pos[:, arm_joint_ids], torch.zeros((1, 2), device=env.device)),
            dim=1,
        )
        env.step(action)

        # Sensors exist per concrete label; the "gripper" group gets none of its own.
        sensors = get_contact_sensors(env.scene)
        assert "left__banana" in sensors
        assert "right__banana" in sensors
        assert not any(name.startswith("gripper__") for name in sensors)

        # The group name queries both hands' sensors and reduces per env.
        world = get_world(env)
        contact = world.in_contact("banana", "gripper")
        assert contact.shape == (1,) and contact.dtype == torch.bool

        # Group-then-negate and negate-each-then-combine must agree.
        via_group = gripper_detached(world, "banana", "gripper")
        via_labels = gripper_detached(world, "banana", ["left", "right"])
        assert torch.equal(via_group, via_labels)
    finally:
        env.close()
