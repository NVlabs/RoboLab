# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from robolab.constants import ROBOTS_DIR

# Ported from Isaac Lab Arena's "Double Piper" embodiment
# (isaaclab_arena/embodiments/double_piper/double_piper.py): two independent 6-DoF
# AgileX Piper arms + parallel grippers mounted on a shared base. Joint/prim names,
# ready pose, actuator gains and camera intrinsics/offsets below are ported from
# that already-tuned config. Note that IsaacLab ArticulationCfg root quaternions
# use wxyz, while Arena's Pose helper is named rotation_xyzw.

_frame_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TF")
_frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

_LEFT_HAND_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/piper_L/hand_link_l/left_hand_camera",
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=19.3,
        focus_distance=400.0,
        horizontal_aperture=31.31,
        vertical_aperture=23.50,
        clipping_range=(0.01, 1.0e5),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.053, 0.0, 0.054),
        # Arena records this offset as xyzw; IsaacLab Camera OffsetCfg expects wxyz.
        rot=(-0.12279, -0.69636, 0.69636, 0.12279),
        convention="opengl",
    ),
)

_RIGHT_HAND_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/piper_R/hand_link_r/right_hand_camera",
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=19.3,
        focus_distance=400.0,
        horizontal_aperture=31.29,
        vertical_aperture=23.49,
        clipping_range=(0.01, 1.0e5),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.053, 0.0, 0.054),
        # Arena records this offset as xyzw; IsaacLab Camera OffsetCfg expects wxyz.
        rot=(-0.12279, -0.69636, 0.69636, 0.12279),
        convention="opengl",
    ),
)

_FIRST_PERSON_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/piper_R/dummy_link/first_person_camera",
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=19.3,
        focus_distance=400.0,
        horizontal_aperture=20.31,
        vertical_aperture=15.25,
        clipping_range=(0.01, 1.0e5),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.02, 0.33, 0.8),
        # Arena records this offset as xyzw; IsaacLab Camera OffsetCfg expects wxyz.
        rot=(0.6861, 0.17106, -0.17106, -0.6861),
        convention="opengl",
    ),
)


@configclass
class PiperCfg:
    """Cfg class that adds the dual-arm Double Piper robot articulation to scene configurations."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ROBOTS_DIR, "double_piper.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Matches DoublePiperSingleObjectEvalEnvironment's initial_pose.
            pos=(0.03, 0.0, 0.05),
            # Identity in IsaacLab's wxyz order. Using (0, 0, 0, 1) here rotates
            # the root 180 deg around Z and makes the robot/cameras face backward.
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "joint1_l": -0.6379,
                "joint2_l": 0.1,
                "joint3_l": -0.4208,
                "joint4_l": 0.3144,
                "joint5_l": 0.7449,
                "joint6_l": -0.3596,
                "joint1_r": 0.3084,
                "joint2_r": 0.1,
                "joint3_r": -0.4139,
                "joint4_r": -0.2013,
                "joint5_r": 0.6952,
                "joint6_r": 0.2756,
                "finger_joint_left_l": 0.035,
                "finger_joint_right_l": -0.035,
                "finger_joint_left_r": 0.035,
                "finger_joint_right_r": -0.035,
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]_l"],
                effort_limit=50.0,
                velocity_limit=20.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]_r"],
                effort_limit=50.0,
                velocity_limit=20.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint.*_l"],
                effort_limit=500.0,
                velocity_limit=0.5,
                stiffness=5000.0,
                damping=200.0,
            ),
            "right_gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint.*_r"],
                effort_limit=500.0,
                velocity_limit=0.5,
                stiffness=5000.0,
                damping=200.0,
            ),
        },
    )

    left_hand_camera = _LEFT_HAND_CAM
    right_hand_camera = _RIGHT_HAND_CAM
    first_person_camera = _FIRST_PERSON_CAM

    frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/root",
        debug_vis=False,
        visualizer_cfg=_frame_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/piper_L/hand_link_l",
                name="ee_tcp_l",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/piper_R/hand_link_r",
                name="ee_tcp_r",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )


@configclass
class LeftHandCameraCfg:
    """Introspection wrapper so the left-hand camera can be passed to generate_image_obs_from_cameras."""
    left_hand_camera = _LEFT_HAND_CAM


@configclass
class RightHandCameraCfg:
    """Introspection wrapper so the right-hand camera can be passed to generate_image_obs_from_cameras."""
    right_hand_camera = _RIGHT_HAND_CAM


@configclass
class FirstPersonCameraCfg:
    """Introspection wrapper so the first-person camera can be passed to generate_image_obs_from_cameras."""
    first_person_camera = _FIRST_PERSON_CAM


########################################################
# Contact gripper
########################################################

# IsaacLab ContactSensor requires exactly one prim per env for filter_prim_paths_expr
# (force_matrix_w) to work; a regex matching both fingers breaks filtered contact
# detection. Use one contact sensor per finger, then OR across the relevant
# finger sensor names in task conditionals. Right-arm finger paths are confirmed
# against the double_piper.usd prim tree; left-arm paths follow the same naming
# pattern and should be verified in Isaac Sim if contact is missing.
contact_gripper = {
    "left_left_finger": "{ENV_REGEX_NS}/robot/piper_L/left_finger_link",
    "left_right_finger": "{ENV_REGEX_NS}/robot/piper_L/right_finger_link",
    "right_left_finger": "{ENV_REGEX_NS}/robot/piper_R/left_finger_link",
    "right_right_finger": "{ENV_REGEX_NS}/robot/piper_R/right_finger_link",
}

########################################################
# Observations
########################################################

_LEFT_ARM_JOINT_PATTERN = re.compile(r"^joint[1-6]_l$")
_RIGHT_ARM_JOINT_PATTERN = re.compile(r"^joint[1-6]_r$")
_LEFT_GRIPPER_PATTERN = re.compile(r"^finger_joint.*_l$")
_RIGHT_GRIPPER_PATTERN = re.compile(r"^finger_joint.*_r$")


def _to_torch(value):
    """Return robot/frame data as a torch tensor regardless of backend (torch or warp)."""
    if isinstance(value, torch.Tensor):
        return value
    return wp.to_torch(value)


def left_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Returns left arm joint positions (joint1_l ... joint6_l)."""
    robot = env.scene[asset_cfg.name]
    indices = [i for i, n in enumerate(robot.data.joint_names) if _LEFT_ARM_JOINT_PATTERN.match(n)]
    return _to_torch(robot.data.joint_pos)[:, indices]


def right_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Returns right arm joint positions (joint1_r ... joint6_r)."""
    robot = env.scene[asset_cfg.name]
    indices = [i for i, n in enumerate(robot.data.joint_names) if _RIGHT_ARM_JOINT_PATTERN.match(n)]
    return _to_torch(robot.data.joint_pos)[:, indices]


def left_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Returns left gripper joint positions (finger_joint*_l), normalized to [0, 1]."""
    robot = env.scene[asset_cfg.name]
    indices = [i for i, n in enumerate(robot.data.joint_names) if _LEFT_GRIPPER_PATTERN.match(n)]
    return _to_torch(robot.data.joint_pos)[:, indices] / 0.035


def right_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Returns right gripper joint positions (finger_joint*_r), normalized to [0, 1]."""
    robot = env.scene[asset_cfg.name]
    indices = [i for i, n in enumerate(robot.data.joint_names) if _RIGHT_GRIPPER_PATTERN.match(n)]
    return _to_torch(robot.data.joint_pos)[:, indices] / 0.035


def left_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Returns left end-effector (hand_link_l) position (x, y, z) in world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("hand_link_l")
    return _to_torch(robot.data.body_pos_w)[:, body_idx, :]


def left_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Returns left end-effector (hand_link_l) orientation as quaternion (w, x, y, z) in world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("hand_link_l")
    return _to_torch(robot.data.body_quat_w)[:, body_idx, :]


def right_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Returns right end-effector (hand_link_r) position (x, y, z) in world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("hand_link_r")
    return _to_torch(robot.data.body_pos_w)[:, body_idx, :]


def right_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Returns right end-effector (hand_link_r) orientation as quaternion (w, x, y, z) in world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("hand_link_r")
    return _to_torch(robot.data.body_quat_w)[:, body_idx, :]


@configclass
class ProprioceptionObservationCfg(ObsGroup):
    left_arm_joint_pos = ObsTerm(func=left_arm_joint_pos)
    right_arm_joint_pos = ObsTerm(func=right_arm_joint_pos)
    left_gripper_pos = ObsTerm(func=left_gripper_pos, clip=(0, 1))
    right_gripper_pos = ObsTerm(func=right_gripper_pos, clip=(0, 1))
    left_ee_pos = ObsTerm(func=left_ee_pos)
    left_ee_quat = ObsTerm(func=left_ee_quat)
    right_ee_pos = ObsTerm(func=right_ee_pos)
    right_ee_quat = ObsTerm(func=right_ee_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False  # must include
        self.concatenate_terms = False  # must include


########################################################
# Actions
########################################################

class SymmetricGripperPositionAction(BinaryJointPositionAction):
    """Continuous gripper control: single scalar -> two symmetric finger joints.

    Input: g in [0, max_opening], where 0 = fully closed, max_opening = fully open.
    Maps to: finger_left = g, finger_right = -g (mirrors open_command scaling).
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        g = actions.clamp(0.0, self.cfg.max_opening)  # (N, 1)
        scale = g / self.cfg.max_opening  # (N, 1)
        self._processed_actions = scale * self._open_command
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class SymmetricGripperPositionActionCfg(BinaryJointPositionActionCfg):
    """Gripper action for policy evaluation: g in [0, max_opening], 0 = closed, max_opening = open."""

    class_type: type = SymmetricGripperPositionAction
    max_opening: float = 0.035


@configclass
class PiperAbsoluteJointPositionActionCfg:
    left_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1_l", "joint2_l", "joint3_l", "joint4_l", "joint5_l", "joint6_l"],
        preserve_order=True,
        use_default_offset=False,
    )
    right_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1_r", "joint2_r", "joint3_r", "joint4_r", "joint5_r", "joint6_r"],
        preserve_order=True,
        use_default_offset=False,
    )
    left_gripper_action = SymmetricGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint.*_l"],
        open_command_expr={"finger_joint_left_l": 0.035, "finger_joint_right_l": -0.035},
        close_command_expr={"finger_joint_left_l": -0.07, "finger_joint_right_l": 0.07},
    )
    right_gripper_action = SymmetricGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint.*_r"],
        open_command_expr={"finger_joint_left_r": 0.035, "finger_joint_right_r": -0.035},
        close_command_expr={"finger_joint_left_r": -0.07, "finger_joint_right_r": 0.07},
    )
