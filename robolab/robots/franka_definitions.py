# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from isaaclab.assets import Articulation
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from robolab.core.actions.isaaclab_compat import (
    RobolabDifferentialInverseKinematicsActionCfg as DifferentialInverseKinematicsActionCfg,
)
from robolab.core.utils.isaaclab_compat import as_torch, quat_isaaclab_to_wxyz

########################################################
# Actions
########################################################

@configclass
class FrankaIKActionCfg:
    """Absolute end-effector pose control via differential IK.

    Cartesian targets are in the robot-root frame (see docs/frames.md).

    Note:
        if self.cfg.command_type == "position", action_dim = 3, (x, y, z)
        if self.cfg.command_type == "pose" and self.cfg.use_relative_mode, action_dim = 6, (dx, dy, dz, droll, dpitch, dyaw)
        if self.cfg.command_type == "pose" and not self.cfg.use_relative_mode, action_dim = 7, (x, y, z, qw, qx, qy, qz)
    """
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.107]),
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )

@configclass
class FrankaRelIKActionCfg:
    """Relative end-effector pose control via differential IK.

    Cartesian deltas are on robot-root axes (see docs/frames.md).

    Note:
        if self.cfg.command_type == "position", action_dim = 3, (x, y, z)
        if self.cfg.command_type == "pose" and self.cfg.use_relative_mode, action_dim = 6, (dx, dy, dz, droll, dpitch, dyaw)
        if self.cfg.command_type == "pose" and not self.cfg.use_relative_mode, action_dim = 7, (x, y, z, qw, qx, qy, qz)
    """
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.107]),
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaJointPositionActionCfg:
    """Joint-space arm + gripper actions; no Cartesian frame (see docs/frames.md)."""
    arm_action = JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )

########################################################
# Contact gripper
########################################################

# IsaacLab ContactSensor requires exactly one prim per env for filter_prim_paths_expr
# (force_matrix_w) to work. panda_.*finger matches 2 bodies per env, breaking
# filtered contact detection. Use one finger only.
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/panda_leftfinger"}

########################################################
# Definitions
########################################################

def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector position (x, y, z) in the robot-root frame (see docs/frames.md)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene["robot"]
    ee_frame_pos, _ = subtract_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        as_torch(ee_frame.data.target_pos_w)[:, 0, :],
    )
    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector orientation as quaternion (w, x, y, z) in the robot-root frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene["robot"]
    _, ee_frame_quat = subtract_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        q02=as_torch(ee_frame.data.target_quat_w)[:, 0, :],
    )
    return quat_isaaclab_to_wxyz(ee_frame_quat)


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = as_torch(robot.data.joint_pos)
    finger_joint_1 = joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)
