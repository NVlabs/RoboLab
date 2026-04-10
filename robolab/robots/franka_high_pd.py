# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

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
                disable_gravity=True,
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
            stiffness=400.0,# 80.0, # HIGH_PD_CFG
            damping=80.0, # 4.0, # HIGH_PD_CFG
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0, # 80.0, # HIGH_PD_CFG
            damping=80.0, # 4.0, # HIGH_PD_CFG
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
                    pos=[0.0, 0.0, 0.0],
                ),
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
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ],
    )

# IsaacLab ContactSensor requires exactly one prim per env for filter_prim_paths_expr
# (force_matrix_w) to work. panda_.*finger matches 2 bodies per env, breaking
# filtered contact detection. Use one finger only.
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/panda_leftfinger"}
