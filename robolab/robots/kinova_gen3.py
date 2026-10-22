# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kinova Gen3 7-DoF with a Robotiq 2F-85 gripper."""

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass

from robolab.constants import ROBOTS_DIR
from robolab.core.utils.isaaclab_compat import as_torch, quat_isaaclab_to_wxyz
from robolab.robots.droid import BinaryJointPositionZeroToOneActionCfg

ARM_JOINT_NAMES = [f"joint_{index}" for index in range(1, 8)]
GRIPPER_JOINT_NAME = "robotiq_85_left_knuckle_joint"
GRIPPER_JOINT_COMMANDS = {
    "robotiq_85_left_inner_knuckle_joint": 0.8,
    GRIPPER_JOINT_NAME: 0.8,
    "robotiq_85_right_inner_knuckle_joint": -0.8,
    "robotiq_85_right_knuckle_joint": -0.8,
    "robotiq_85_left_finger_tip_joint": -0.8,
    "robotiq_85_right_finger_tip_joint": 0.8,
}
END_EFFECTOR_LINK_NAME = "robotiq_85_base_link"

_WRIST_CAM = TiledCameraCfg(
    # Keep the camera on bracelet_link so it follows the physical Kinova mount.
    prim_path="{ENV_REGEX_NS}/robot/bracelet_link/wrist_cam",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        # Match the 1280x720 intrinsics packaged with the Kinova description.
        focal_length=3.895,
        focus_distance=28.0,
        horizontal_aperture=3.84,
        vertical_aperture=2.16,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        # Kortex's simulation-camera pose expressed in the bracelet frame.
        pos=(0.0, -0.05639, -0.074575),
        rot=(0.5, 0.5, 0.5, -0.5),
        convention="world",
    ),
)


@configclass
class KinovaGen3Cfg:
    """Fixed-base Kinova Gen3 articulation with joint-position control."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                ROBOTS_DIR,
                "kinova_gen3_robotiq_2f85",
                "kinova_gen3_7dof_robotiq_2f85.usd",
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.26,
                "joint_3": np.pi,
                "joint_4": -2.27,
                "joint_5": 0.0,
                "joint_6": 0.96,
                "joint_7": np.pi / 2,
                "robotiq_85_.*_(knuckle|finger_tip)_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "arm_1_to_4": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-4]"],
                effort_limit=39.0,
                velocity_limit=1.3963,
                stiffness=80.0,
                damping=4.0,
            ),
            "arm_5_to_7": ImplicitActuatorCfg(
                joint_names_expr=["joint_[5-7]"],
                effort_limit=9.0,
                velocity_limit=1.2218,
                stiffness=80.0,
                damping=4.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=list(GRIPPER_JOINT_COMMANDS),
                effort_limit=50.0,
                velocity_limit=0.5,
                stiffness=100.0,
                damping=5.0,
            ),
        },
    )

    wrist_cam = _WRIST_CAM

    frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/robot/{END_EFFECTOR_LINK_NAME}",
                name="eef_frame",
            )
        ],
    )


def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    indices = [robot.data.joint_names.index(name) for name in ARM_JOINT_NAMES]
    return as_torch(robot.data.joint_pos)[:, indices]


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    index = robot.data.joint_names.index(GRIPPER_JOINT_NAME)
    return as_torch(robot.data.joint_pos)[:, index : index + 1] / 0.8


def eef_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")
):
    frames = env.scene[asset_cfg.name]
    index = frames.data.target_frame_names.index("eef_frame")
    return (
        as_torch(frames.data.target_pos_w)[:, index, :] - env.scene.env_origins[:, :3]
    )


def eef_quat(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")
):
    frames = env.scene[asset_cfg.name]
    index = frames.data.target_frame_names.index("eef_frame")
    return quat_isaaclab_to_wxyz(as_torch(frames.data.target_quat_w)[:, index, :])


@configclass
class KinovaJointPositionActionCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-7]"],
        preserve_order=True,
        use_default_offset=False,
    )
    # Match RoboLab policy inputs: 0 opens the gripper and 1 closes it.
    gripper = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=list(GRIPPER_JOINT_COMMANDS),
        open_command_expr={name: 0.0 for name in GRIPPER_JOINT_COMMANDS},
        close_command_expr=GRIPPER_JOINT_COMMANDS,
    )


@configclass
class KinovaWristCameraCfg:
    """Expose the robot-mounted camera to image observation generation.

    The scene still gets this sensor from ``KinovaGen3Cfg`` so the parent robot
    is spawned first. This wrapper only exposes the policy observation name.
    """

    wrist_cam = _WRIST_CAM


@configclass
class KinovaProprioceptionObservationCfg(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)
    gripper_pos = ObsTerm(func=gripper_pos, clip=(0.0, 1.0))
    eef_pos = ObsTerm(func=eef_pos)
    eef_quat = ObsTerm(func=eef_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


contact_gripper = {
    "gripper": "{ENV_REGEX_NS}/robot/robotiq_85_.*_finger_tip_link",
}

# Class-level label, assigned after the class body so configclass does not turn
# it into a config field. EE-pose recorder channels (HDF5 channel name -> EE
# body name), consumed by create_recorder_config. The Kinova articulation root
# is also named "base_link", so the gripper base must be named explicitly.
KinovaGen3Cfg.ee_recorder_bodies = {"ee_pose": END_EFFECTOR_LINK_NAME}
