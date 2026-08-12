# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the fixed-base Galbot One Golf embodiment."""

import pytest

from robolab.constants import TASK_DIR
from robolab.core.environments.config import generate_scene_env_cfg
from robolab.core.environments.scene_fixture import (
    FRANKA_TABLE_FIXTURE,
    TableFixtureCfg,
    scene_ground_height,
    spawn_scene_without_table_fixture,
    table_fixture_asset,
)
from robolab.core.task.task_utils import load_task_from_file, resolve_task_path
from robolab.robots.droid import DroidCfg
from robolab.robots.galbot_golf import (
    GalbotGolfFixedBaseCfg,
    GalbotGolfFixedBaseDefaultPoseCfg,
    GalbotGolfLeftEgoCameraCfg,
    GalbotGolfLeftWristCameraCfg,
    GalbotGolfRightWristCameraCfg,
    GalbotGolfTabletopCfg,
)
from robolab.robots.galbot_golf_definitions import (
    BODY_JOINTS,
    GRIPPER_JOINTS,
    PROPRIO_JOINTS,
    WHOLE_BODY_JOINTS,
    GalbotGolfJointPositionAction,
    GalbotGolfJointPositionActionCfg,
    GalbotGolfWholeBodyJointPositionActionCfg,
)


def test_policy_joint_order_is_stable_and_complete():
    assert PROPRIO_JOINTS == BODY_JOINTS
    assert len(PROPRIO_JOINTS) == 21
    assert WHOLE_BODY_JOINTS == BODY_JOINTS + GRIPPER_JOINTS
    assert len(WHOLE_BODY_JOINTS) == 23
    assert len(set(WHOLE_BODY_JOINTS)) == len(WHOLE_BODY_JOINTS)


def test_joint_actions_seed_uncommanded_targets_on_reset():
    arms = GalbotGolfJointPositionActionCfg().arms
    whole_body = GalbotGolfWholeBodyJointPositionActionCfg().body

    assert arms.class_type is GalbotGolfJointPositionAction
    assert whole_body.class_type is GalbotGolfJointPositionAction
    assert arms.joint_names == BODY_JOINTS[7:]
    assert whole_body.joint_names == BODY_JOINTS


def test_left_ego_camera_uses_golf_sensor_calibration():
    camera = GalbotGolfLeftEgoCameraCfg().left_ego_cam
    assert camera.prim_path.endswith("/robot/head_link2/head_end_effector_mount_link/camera_front_head_left_rgb")
    assert camera.offset.pos == pytest.approx((0.082781626, -0.053015078, 0.031427263))
    assert camera.offset.rot == pytest.approx((0.698790116, -0.094725615, 0.702703943, 0.094480690))
    assert camera.offset.convention == "ros"
    assert (camera.width, camera.height) == (640, 480)
    assert camera.spawn.clipping_range == pytest.approx((0.1, 15.0))
    assert camera.width * camera.spawn.focal_length / camera.spawn.horizontal_aperture == pytest.approx(
        198.96716907989332
    )
    assert camera.height * camera.spawn.focal_length / camera.spawn.vertical_aperture == pytest.approx(
        198.96716907989332
    )


def test_wrist_cameras_use_golf_sensor_calibration():
    left = GalbotGolfLeftWristCameraCfg().left_wrist_cam
    right = GalbotGolfRightWristCameraCfg().right_wrist_cam
    assert (left.width, left.height) == (400, 224)
    assert (right.width, right.height) == (400, 224)
    for camera in (left, right):
        assert camera.width * camera.spawn.focal_length / camera.spawn.horizontal_aperture == pytest.approx(202)
        assert camera.height * camera.spawn.focal_length / camera.spawn.vertical_aperture == pytest.approx(202)
        assert camera.spawn.clipping_range == pytest.approx((0.03, 10.0))


def test_tabletop_and_replay_root_frames_remain_separate():
    tabletop_pos = tuple(GalbotGolfTabletopCfg().robot.init_state.pos)
    replay_pos = tuple(GalbotGolfFixedBaseDefaultPoseCfg().robot.init_state.pos)
    assert tabletop_pos[:2] == (0.0, 0.0)
    assert replay_pos == (0.0, 0.0, 0.0)
    assert tabletop_pos[2] != replay_pos[2]


def test_tabletop_root_clears_canonical_ground():
    """Wheel colliders (root - 26.4 mm) must rest on, not below, the -0.697 ground."""
    root_z = GalbotGolfTabletopCfg().robot.init_state.pos[2]
    wheel_bottom_z = root_z - 0.0264
    assert wheel_bottom_z == pytest.approx(-0.697, abs=1e-4)


def test_table_fixture_labels():
    assert DroidCfg.table_fixture is FRANKA_TABLE_FIXTURE
    assert GalbotGolfFixedBaseCfg.table_fixture is None
    # Labels must stay class attributes, not config fields.
    for cfg_cls in (DroidCfg, GalbotGolfFixedBaseCfg, GalbotGolfTabletopCfg):
        assert "table_fixture" not in cfg_cls.__dataclass_fields__


def test_table_fixture_follows_robot_label():
    task_path, _ = resolve_task_path("BananaInBowlTask", TASK_DIR)
    task_class = load_task_from_file(task_path)

    # The legacy baked-in fixture prim is deactivated for every robot; the
    # robot's declared fixture is spawned as its own scene entity.
    droid_scene = generate_scene_env_cfg(task_class, DroidCfg)(num_envs=1, env_spacing=10.0)
    assert droid_scene.scene.spawn.func is spawn_scene_without_table_fixture
    assert droid_scene.table_fixture.spawn.usd_path == FRANKA_TABLE_FIXTURE.usd_path
    assert tuple(droid_scene.table_fixture.init_state.pos) == pytest.approx((-0.087, 0.0, 0.0))

    galbot_scene = generate_scene_env_cfg(task_class, GalbotGolfTabletopCfg)(num_envs=1, env_spacing=10.0)
    assert galbot_scene.scene.spawn.func is spawn_scene_without_table_fixture
    assert galbot_scene.table_fixture is None
    # Fixed root height: authored in the robot cfg, independent of the scene.
    assert galbot_scene.robot.init_state.pos[2] == pytest.approx(-0.6706)
    assert scene_ground_height(task_class.scene().scene) == pytest.approx(-0.697)


def test_robot_frame_fixture_pose_composes_with_robot_root():
    fixture = TableFixtureCfg(usd_path="unused.usd", pos=(0.5, 0.0, 0.0), frame="robot")

    class _FakeState:
        pos = (1.0, 0.0, 0.0)
        rot = (0.0, 0.0, 0.0, 1.0)  # 180 degrees about +Z

    class _FakeRobot:
        init_state = _FakeState()

    class _FakeRobotCfg:
        def __init__(self):
            self.robot = _FakeRobot()

    asset = table_fixture_asset(fixture, _FakeRobotCfg)
    assert tuple(asset.init_state.pos) == pytest.approx((0.5, 0.0, 0.0))
    assert tuple(asset.init_state.rot) == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert table_fixture_asset(None, _FakeRobotCfg) is None
