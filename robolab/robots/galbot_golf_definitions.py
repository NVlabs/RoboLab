# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Galbot One Golf fixed-base joints, observations, and action configs."""

from collections.abc import Sequence

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

LEG_JOINTS = [f"leg_joint{i}" for i in range(1, 6)]
HEAD_JOINTS = [f"head_joint{i}" for i in range(1, 3)]
LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINTS = ["left_gripper_joint"]
RIGHT_GRIPPER_JOINTS = ["right_gripper_joint"]
WHEEL_JOINTS = [f"wheel{i}_joint" for i in range(1, 5)]

ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
GRIPPER_JOINTS = LEFT_GRIPPER_JOINTS + RIGHT_GRIPPER_JOINTS
BODY_JOINTS = LEG_JOINTS + HEAD_JOINTS + ARM_JOINTS
WHOLE_BODY_JOINTS = BODY_JOINTS + GRIPPER_JOINTS
PROPRIO_JOINTS = BODY_JOINTS
LEFT_EE_BODY = "left_arm_link7"
RIGHT_EE_BODY = "right_arm_link7"

# Fixed transforms and linkage values authored in the received Golf description.
TCP_OFFSET_POS = (-0.25572, 0.0, 0.0)
TCP_OFFSET_ROT = (0.0, 0.0, 0.0, 1.0)
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.703
GRIPPER_KNUCKLE_ANGLE = 1.2465
GRIPPER_INNER_PIVOT_HALF_GAP = 0.026
GRIPPER_FINGER_LINK_LENGTH = 0.045
GRIPPER_PAD_INSET = 0.0062

GALBOT_GOLF_DEFAULT_JOINT_POS = {
    "leg_joint1": 0.8,
    "leg_joint2": 2.3,
    "leg_joint3": 1.55,
    "leg_joint4": 0.0,
    "leg_joint5": 0.0,
    "head_joint1": 0.0,
    "head_joint2": 0.36,
    "left_arm_joint1": -0.1535,
    "left_arm_joint2": -1.0087,
    "left_arm_joint3": -0.0895,
    "left_arm_joint4": -1.5743,
    "left_arm_joint5": 0.2422,
    "left_arm_joint6": -0.0009,
    "left_arm_joint7": 0.9143,
    "left_gripper_joint": GRIPPER_OPEN,
    "right_arm_joint1": 0.1535,
    "right_arm_joint2": 1.0087,
    "right_arm_joint3": 0.0895,
    "right_arm_joint4": 1.5743,
    "right_arm_joint5": -0.2422,
    "right_arm_joint6": -0.0009,
    "right_arm_joint7": -0.9143,
    "right_gripper_joint": GRIPPER_OPEN,
}

contact_gripper = {
    "left": "{ENV_REGEX_NS}/robot/left_gripper_l_finger_link",
    "right": "{ENV_REGEX_NS}/robot/right_gripper_l_finger_link",
    # Benchmark tasks' generic "gripper" means either hand; bimanual tasks
    # target "left"/"right" directly (or ["left", "right"] for both-hands-on).
    "gripper": ["left", "right"],
}

# Contact label -> (actuated joint, open position, closed position).
gripper_closure_cfg = {
    "left": ("left_gripper_joint", GRIPPER_OPEN, GRIPPER_CLOSED),
    "right": ("right_gripper_joint", GRIPPER_OPEN, GRIPPER_CLOSED),
}


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = [robot.data.joint_names.index(name) for name in PROPRIO_JOINTS]
    return robot.data.joint_pos[:, joint_ids]


def _gripper_closed_fraction(robot: Articulation, parent_joint: str):
    idx = robot.data.joint_names.index(parent_joint)
    return torch.clamp(robot.data.joint_pos[:, idx] / GRIPPER_CLOSED, 0.0, 1.0).unsqueeze(1)


def left_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _gripper_closed_fraction(env.scene[asset_cfg.name], "left_gripper_joint")


def right_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _gripper_closed_fraction(env.scene[asset_cfg.name], "right_gripper_joint")


def _ee_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, body_name: str):
    robot: Articulation = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index(body_name)
    pos, quat = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, body_idx, :],
        robot.data.body_quat_w[:, body_idx, :],
    )
    offset_pos = torch.tensor(TCP_OFFSET_POS, device=pos.device, dtype=pos.dtype).expand_as(pos)
    offset_rot = torch.tensor(TCP_OFFSET_ROT, device=quat.device, dtype=quat.dtype).expand_as(quat)
    return math_utils.combine_frame_transforms(pos, quat, offset_pos, offset_rot)


def left_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, asset_cfg, LEFT_EE_BODY)[0]


def left_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, asset_cfg, LEFT_EE_BODY)[1]


def right_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, asset_cfg, RIGHT_EE_BODY)[0]


def right_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, asset_cfg, RIGHT_EE_BODY)[1]


@configclass
class ProprioceptionObservationCfg(ObsGroup):
    joint_pos = ObsTerm(func=joint_pos)
    left_gripper_pos = ObsTerm(func=left_gripper_pos)
    right_gripper_pos = ObsTerm(func=right_gripper_pos)
    left_ee_pos = ObsTerm(func=left_ee_pos)
    left_ee_quat = ObsTerm(func=left_ee_quat)
    right_ee_pos = ObsTerm(func=right_ee_pos)
    right_ee_quat = ObsTerm(func=right_ee_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


class GalbotGolfJointPositionAction(JointPositionAction):
    """Joint targets that keep all uncommanded joints at the reset posture."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        target_env_ids = slice(None) if env_ids is None else env_ids
        self._asset.set_joint_position_target(
            self._asset.data.default_joint_pos[target_env_ids], env_ids=target_env_ids
        )
        self._asset.set_joint_velocity_target(
            self._asset.data.default_joint_vel[target_env_ids], env_ids=target_env_ids
        )


def _joint_pos_action(joint_names: list[str]) -> mdp.JointPositionActionCfg:
    cfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        preserve_order=True,
        use_default_offset=False,
    )
    cfg.class_type = GalbotGolfJointPositionAction
    return cfg


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    """Binary joint target with RoboLab's policy convention: >0.5 closes."""

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        binary_mask = actions if actions.dtype == torch.bool else actions > 0.5
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    class_type = BinaryJointPositionZeroToOneAction


def _gripper_binary_action(gripper_joints: list[str]) -> BinaryJointPositionZeroToOneActionCfg:
    return BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=gripper_joints,
        open_command_expr={joint: GRIPPER_OPEN for joint in gripper_joints},
        close_command_expr={joint: GRIPPER_CLOSED for joint in gripper_joints},
    )


class GalbotGolfDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Differential IK with Golf's TCP-offset Jacobian expressed in the root frame."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        target_env_ids = slice(None) if env_ids is None else env_ids
        self._asset.set_joint_position_target(
            self._asset.data.default_joint_pos[target_env_ids], env_ids=target_env_ids
        )
        self._asset.set_joint_velocity_target(
            self._asset.data.default_joint_vel[target_env_ids], env_ids=target_env_ids
        )

    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_b.clone()
        if self.cfg.body_offset is None:
            return jacobian

        body_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        _, body_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            self._asset.data.body_pos_w[:, self._body_idx],
            body_quat_w,
        )
        offset_pos_b = math_utils.quat_apply(body_quat_b, self._offset_pos)
        jacobian[:, 0:3, :] -= torch.bmm(
            math_utils.skew_symmetric_matrix(offset_pos_b),
            jacobian[:, 3:, :],
        )
        return jacobian


def _arm_differential_ik_action(arm_joints: list[str], body_name: str) -> DifferentialInverseKinematicsActionCfg:
    cfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=arm_joints,
        body_name=body_name,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=TCP_OFFSET_POS,
            rot=TCP_OFFSET_ROT,
        ),
    )
    cfg.class_type = GalbotGolfDifferentialInverseKinematicsAction
    return cfg


@configclass
class GalbotGolfJointPositionActionCfg:
    """Both arms plus one binary open/close command per gripper."""

    arms = _joint_pos_action(ARM_JOINTS)
    left_finger = _gripper_binary_action(LEFT_GRIPPER_JOINTS)
    right_finger = _gripper_binary_action(RIGHT_GRIPPER_JOINTS)


@configclass
class GalbotGolfWholeBodyJointPositionActionCfg:
    """Legs, head, both arms, and binary grippers."""

    body = _joint_pos_action(BODY_JOINTS)
    left_finger = _gripper_binary_action(LEFT_GRIPPER_JOINTS)
    right_finger = _gripper_binary_action(RIGHT_GRIPPER_JOINTS)


@configclass
class GalbotGolfWholeBodyContinuousGripperActionCfg:
    """Native replay action with body and continuous gripper joint targets."""

    body = _joint_pos_action(BODY_JOINTS)
    grippers = _joint_pos_action(GRIPPER_JOINTS)


@configclass
class GalbotGolfDifferentialIKActionCfg:
    """Dual-arm absolute TCP control with binary grippers."""

    left_arm_action = _arm_differential_ik_action(LEFT_ARM_JOINTS, LEFT_EE_BODY)
    left_finger = _gripper_binary_action(LEFT_GRIPPER_JOINTS)
    right_arm_action = _arm_differential_ik_action(RIGHT_ARM_JOINTS, RIGHT_EE_BODY)
    right_finger = _gripper_binary_action(RIGHT_GRIPPER_JOINTS)


@configclass
class GalbotGolfDifferentialIKContinuousGripperActionCfg:
    """Native replay action with absolute TCP control and continuous grippers."""

    left_arm_action = _arm_differential_ik_action(LEFT_ARM_JOINTS, LEFT_EE_BODY)
    left_finger = _joint_pos_action(LEFT_GRIPPER_JOINTS)
    right_arm_action = _arm_differential_ik_action(RIGHT_ARM_JOINTS, RIGHT_EE_BODY)
    right_finger = _joint_pos_action(RIGHT_GRIPPER_JOINTS)
