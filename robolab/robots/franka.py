# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

from robolab.core.environments.scene_fixture import FRANKA_TABLE_FIXTURE
from robolab.core.utils.isaaclab_compat import as_torch, quat_isaaclab_to_wxyz
from robolab.robots.franka_definitions import *  # noqa

# Create a copy of the default frame marker config
frame_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TF")
frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)  # Desired marker size

@configclass
class FrankaCfg:
    """Cfg class that adds robot articulation to scene configurations."""

    # Adapted from isaaclab_assets.robots.franka.FRANKA_PANDA_CFG
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),

    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },

    soft_joint_pos_limit_factor=1.0,
    )

    frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=frame_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos = [0.0, 0.0, 0.0],
                    # pos=[0.0, 0.0, 0.1034],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/panda_hand",
                name="curobo_control_frame",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/panda_hand",
                name="contact_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                    rot=[0.0, 0.0, 0.0, 1.0],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/panda_hand",
                name="acronym_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                    rot=[0.70711, 0.0, 0.0, -0.70711],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/panda_rightfinger",
                name="rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/panda_leftfinger",
                name="leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=frame_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )


# Class-level label, assigned after the class body so configclass does not turn
# it into a config field. See docs/robots.md#table-fixture.
FrankaCfg.table_fixture = FRANKA_TABLE_FIXTURE
# EE-pose recorder channels (HDF5 channel name -> EE body name), consumed by
# create_recorder_config.
FrankaCfg.ee_recorder_bodies = {"ee_pose": "panda_hand"}


########################################################
# Helper functions for observations. Use these for both franka and franka_high_pd.
########################################################

def ee_frame_pos(env: ManagerBasedRLEnv, frames_cfg: SceneEntityCfg = SceneEntityCfg("frames")) -> torch.Tensor:
    """End-effector position (x, y, z) in the robot-root frame (see docs/frames.md)."""
    frames: FrameTransformer = env.scene[frames_cfg.name]
    robot: Articulation = env.scene["robot"]
    ee_frame_pos, _ = subtract_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        as_torch(frames.data.target_pos_w)[:, 0, :],
    )
    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, frames_cfg: SceneEntityCfg = SceneEntityCfg("frames")) -> torch.Tensor:
    """End-effector orientation as quaternion (w, x, y, z) in the robot-root frame."""
    frames: FrameTransformer = env.scene[frames_cfg.name]
    robot: Articulation = env.scene["robot"]
    _, ee_frame_quat = subtract_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        q02=as_torch(frames.data.target_quat_w)[:, 0, :],
    )
    return quat_isaaclab_to_wxyz(ee_frame_quat)


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = as_torch(robot.data.joint_pos)
    finger_joint_1 = joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)
