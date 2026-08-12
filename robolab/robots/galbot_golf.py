# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-base ``galbot_one_golf`` dual-arm robot configuration."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.constants import ROBOTS_DIR
from robolab.robots.galbot_golf_definitions import *  # noqa

GALBOT_GOLF_USD_PATH = os.environ.get(
    "GALBOT_GOLF_USD_PATH",
    os.path.join(ROBOTS_DIR, "galbot_one_golf_description", "usd", "galbot_one_golf.usda"),
)

GALBOT_GOLF_USD_VARIANTS = {
    "Physics": "physx",
    "Robot": "robot",
    "Sensor": "sensors",
}

# The Golf's lowest colliders — the passive caster-wheel spheres — reach
# 26.4 mm below the root origin (measured by check_standing_contact.py's
# static scan). The env factory places the root this far above each scene's
# authored /GroundPlane via the root_z_above_ground label, so the wheels rest
# exactly on the floor at any scene ground height: -0.6706 in canonical
# -0.697 scenes, -0.6236 in legacy -0.65 scenes (tests/test_scene_ground.py
# locks the per-scene heights). The root is welded (fix_root_link), so the
# robot needs no ground support either way.
GALBOT_GOLF_ROOT_Z_ABOVE_GROUND = 0.0264
# Canonical-scene default, for instantiating the cfg outside the env factory.
GALBOT_GOLF_TABLETOP_ROOT_POS = (0.0, 0.0, -0.697 + GALBOT_GOLF_ROOT_Z_ABOVE_GROUND)


########################################################
# Robot-attached and replay cameras
########################################################


def _source_replay_camera(
    name: str,
    *,
    parent_path: str = "base_footprint/base_link",
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    focal_length: float,
    focus_distance: float,
    horizontal_aperture: float,
) -> TiledCameraCfg:
    """Robot-attached camera matching the existing Galbot replay/test camera convention."""
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/robot/{parent_path}/{name}",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=focus_distance,
            horizontal_aperture=horizontal_aperture,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
    )


def _wrist_camera(side: str) -> TiledCameraCfg:
    """Camera parented to Golf's URDF camera frame, using its +Z optical axis."""
    return TiledCameraCfg(
        prim_path=(
            f"{{ENV_REGEX_NS}}/robot/{side}_arm_link7/"
            f"{side}_arm_wrist_camera_stand/{side}_wrist_camera_link/{side}_wrist_cam"
        ),
        height=224,
        width=400,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=202 * 0.03,
            focus_distance=0.0,
            horizontal_aperture=400 * 0.03,
            vertical_aperture=224 * 0.03,
            clipping_range=(0.03, 10.0),
        ),
        # The URDF camera link supplies the optical +Z direction, but its image
        # axes are rolled 90 degrees relative to the upright D405 recording.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.7071067812, 0.0, 0.0, 0.7071067812),
            convention="ros",
        ),
    )


_LEFT_WRIST_CAM = _wrist_camera("left")
_RIGHT_WRIST_CAM = _wrist_camera("right")
_FRONT_CAM = _source_replay_camera(
    "front_camera",
    pos=(2.016, 0.0, 1.826),
    rot=(0.603996, 0.367682, 0.367682, 0.603996),
    focal_length=16.0,
    focus_distance=150.0,
    horizontal_aperture=20.955,
)
_LEFT_EGO_CAM = TiledCameraCfg(
    prim_path=(
        "{ENV_REGEX_NS}/robot/head_link2/head_end_effector_mount_link/"
        "camera_front_head_left_rgb"
    ),
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=198.96716907989332 * 0.02,
        focus_distance=0.0,
        horizontal_aperture=640 * 0.02,
        vertical_aperture=480 * 0.02,
        clipping_range=(0.1, 15.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.082781626, -0.053015078, 0.031427263),
        # golf_sensors.json stores xyzw; Isaac Lab expects wxyz.
        rot=(0.698790116, -0.094725615, 0.702703943, 0.094480690),
        convention="ros",
    ),
)


def _galbot_golf_robot_cfg(
    *,
    init_joint_pos: dict[str, float] | None = None,
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ArticulationCfg:
    """Build a Golf articulation config while preserving the received USD physics properties."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GALBOT_GOLF_USD_PATH,
            variants=GALBOT_GOLF_USD_VARIANTS,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=init_joint_pos
            or {
                "leg_joint.*": 0.0,
                "head_joint.*": 0.0,
                "left_arm_joint.*": 0.0,
                "right_arm_joint.*": 0.0,
                "left_gripper_joint": GRIPPER_OPEN,
                "right_gripper_joint": GRIPPER_OPEN,
                ".*_gripper_.*_joint": 0.0,
                "wheel.*joint": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            # Preserve all drive properties authored in the supplied PhysX USD.
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["leg_joint.*"],
                effort_limit_sim=None,
                velocity_limit_sim=None,
                stiffness=None,
                damping=None,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_joint.*"],
                effort_limit_sim=None,
                velocity_limit_sim=None,
                stiffness=None,
                damping=None,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint.*", "right_arm_joint.*"],
                effort_limit_sim=None,
                velocity_limit_sim=None,
                stiffness=None,
                damping=None,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=["left_gripper_joint", "right_gripper_joint"],
                effort_limit_sim=None,
                velocity_limit_sim=None,
                stiffness=None,
                damping=None,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=WHEEL_JOINTS,
                effort_limit_sim=None,
                velocity_limit_sim=None,
                stiffness=None,
                damping=None,
            ),
        },
    )


@configclass
class GalbotGolfFixedBaseCfg:
    """Fixed-root Galbot One Golf articulation for static-base manipulation replay."""

    robot = _galbot_golf_robot_cfg()

    left_wrist_cam = _LEFT_WRIST_CAM
    right_wrist_cam = _RIGHT_WRIST_CAM


@configclass
class GalbotGolfFixedBaseDefaultPoseCfg(GalbotGolfFixedBaseCfg):
    """Fixed-root Golf with a RoboLab reset posture for manipulation tasks."""

    robot = _galbot_golf_robot_cfg(init_joint_pos=GALBOT_GOLF_DEFAULT_JOINT_POS)


@configclass
class GalbotGolfTabletopCfg(GalbotGolfFixedBaseDefaultPoseCfg):
    """Fixed-base Golf placed on the floor of standard RoboLab table scenes."""

    robot = _galbot_golf_robot_cfg(
        init_joint_pos=GALBOT_GOLF_DEFAULT_JOINT_POS,
        init_pos=GALBOT_GOLF_TABLETOP_ROOT_POS,
    )


@configclass
class GalbotGolfReplayCfg(GalbotGolfFixedBaseDefaultPoseCfg):
    """Fixed-base Golf plus report-only front and left-ego cameras."""

    front_cam = _FRONT_CAM
    left_ego_cam = _LEFT_EGO_CAM


@configclass
class GalbotGolfTabletopReplayCfg(GalbotGolfTabletopCfg):
    """Tabletop Golf plus report-only front and left-ego cameras."""

    front_cam = _FRONT_CAM
    left_ego_cam = _LEFT_EGO_CAM


@configclass
class GalbotGolfLeftWristCameraCfg:
    left_wrist_cam = _LEFT_WRIST_CAM


@configclass
class GalbotGolfRightWristCameraCfg:
    right_wrist_cam = _RIGHT_WRIST_CAM


@configclass
class GalbotGolfFrontCameraCfg:
    front_cam = _FRONT_CAM


@configclass
class GalbotGolfLeftEgoCameraCfg:
    left_ego_cam = _LEFT_EGO_CAM


# Class-level labels, assigned after ALL Golf cfg classes: configclass
# decoration of a subclass converts inherited plain attributes into config
# fields, so these must come after the last subclass definition.
# See docs/robots.md#table-fixture.
GalbotGolfFixedBaseCfg.table_fixture = None  # Golf has its own base
# Bimanual: one EE-pose channel per arm. A single-arm channel would be
# meaningless here, and a "base_link" body name would match the Golf's torso
# base instead of either gripper.
GalbotGolfFixedBaseCfg.ee_recorder_bodies = {
    "left_ee_pose": LEFT_EE_BODY,
    "right_ee_pose": RIGHT_EE_BODY,
}
# Floor-standing: the env factory rebases the root onto each scene's authored
# ground. Only the Tabletop cfgs stand on the task-scene floor; the plain
# fixed-base cfgs keep their explicit init pos.
GalbotGolfTabletopCfg.root_z_above_ground = GALBOT_GOLF_ROOT_Z_ABOVE_GROUND
