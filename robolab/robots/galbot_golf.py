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

GALBOT_GOLF_ASSET_DIR = os.path.join(ROBOTS_DIR, "galbot_one_golf_description")
GALBOT_GOLF_USD_PATH = os.environ.get(
    "GALBOT_GOLF_USD_PATH",
    os.path.join(GALBOT_GOLF_ASSET_DIR, "usd", "galbot_one_golf.usda"),
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
    """Yundonghui policy calibration resized from 400x224 to 640x360."""
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/robot/{side}_arm_link7/{side}_wrist_cam",
        height=360,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # Scaled intrinsics: fx=323.2, fy=324.642857, cx=320, cy=180.
            focal_length=6.06,
            focus_distance=0.0,
            horizontal_aperture=12.0,
            vertical_aperture=6.72,
            clipping_range=(0.03, 10.0),
        ),
        # Mount-relative calibration expressed in arm_link7, quaternion WXYZ.
        # Both wrists have the same rotation (the source quaternions differ only in sign).
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.18015459385344976, 0.011084636915734618, -0.0475356786953811),
            rot=(-0.5860286987303605, -0.38397145780165504, 0.3981361647004496, 0.5921350168801781),
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
        focal_length=0.488472287905,
        horizontal_aperture=1.280137468212909,
        vertical_aperture=0.9598969209810818,
        clipping_range=(0.08, 15.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.07026692, -0.05334402, 0.02098859),
        rot=(0.69873623, -0.09626416, 0.70197894, 0.09862283),
        convention="ros",
    ),
)


def _galbot_golf_robot_cfg(
    *,
    init_joint_pos: dict[str, float] | None = None,
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gain_profile: str = "default",
) -> ArticulationCfg:
    """Build a Golf articulation config with Sim2Real gains and controller gravity compensation.

    Stiffness, damping, and effort limits are sourced from the
    ``galbot-one-golf-sim2real-v1`` system-identification profile
    (synthnova/src/synthnova/robots/cfg/galbot_one_golf.toml).
    Left/right arm gains are averaged, then rounded to the nearest integer;
    other gains are rounded directly (half values round up). Effort and
    specified velocity limits are retained from the profile. The profile
    leaves head/gripper velocity limits unspecified: retain the USD head limit
    and RoboLab's existing 3.5 rad/s gripper limit.

    Gravity compensation is realised by disabling per-link gravity in the
    physics engine (``disable_gravity=True``). This approximates ideal
    controller-side compensation for this fixed-base embodiment.
    """
    if gain_profile not in GALBOT_GOLF_ARM_GAIN_PROFILES:
        raise ValueError(f"Unknown gain_profile {gain_profile!r}. Choose from {sorted(GALBOT_GOLF_ARM_GAIN_PROFILES)}.")
    arm_gains = GALBOT_GOLF_ARM_GAIN_PROFILES[gain_profile]
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GALBOT_GOLF_USD_PATH,
            variants=GALBOT_GOLF_USD_VARIANTS,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # Simulate controller-side gravity compensation: the real robot
                # compensates gravity in its low-level controller, so the
                # joint-position PD loops only see tracking error, not gravity load.
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                solver_position_iteration_count=128,
                solver_velocity_iteration_count=4,
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
            # --- Leg linkage (Sim2Real profile, per-joint) ---
            # The five leg joints span three torque tiers; damping is rounded
            # from the system-identification profile.
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["leg_joint.*"],
                effort_limit_sim={
                    "leg_joint1": 433,
                    "leg_joint2": 433,
                    "leg_joint3": 204,
                    "leg_joint4": 70,
                    "leg_joint5": 70,
                },
                velocity_limit_sim={
                    "leg_joint1": 2.094395,
                    "leg_joint2": 2.094395,
                    "leg_joint3": 3.141593,
                    "leg_joint4": 3.141593,
                    "leg_joint5": 3.141593,
                },
                stiffness={
                    "leg_joint1": 8660,
                    "leg_joint2": 8660,
                    "leg_joint3": 4080,
                    "leg_joint4": 1400,
                    "leg_joint5": 1400,
                },
                damping={
                    "leg_joint1": 727,
                    "leg_joint2": 567,
                    "leg_joint3": 222,
                    "leg_joint4": 88,
                    "leg_joint5": 85,
                },
            ),
            # --- Head pan / tilt ---
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_joint.*"],
                effort_limit_sim=4,
                stiffness=80,
                damping=4,
            ),
            # --- Arm shoulder (joints 1–2, ±180 N·m) ---
            # Left/right values are symmetrised (averaged) and rounded to integers.
            "arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint[12]", "right_arm_joint[12]"],
                effort_limit_sim=180,
                velocity_limit_sim=3.141593,
                stiffness=arm_gains["shoulder"]["stiffness"],
                damping=arm_gains["shoulder"]["damping"],
            ),
            # --- Arm elbow (joints 3–4, ±90 N·m) ---
            "arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint[34]", "right_arm_joint[34]"],
                effort_limit_sim=90,
                velocity_limit_sim=3.926991,
                stiffness=arm_gains["elbow"]["stiffness"],
                damping=arm_gains["elbow"]["damping"],
            ),
            # --- Arm wrist (joints 5–7, ±30 N·m) ---
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint[5-7]", "right_arm_joint[5-7]"],
                effort_limit_sim=30,
                velocity_limit_sim=3.926991,
                stiffness=arm_gains["wrist"]["stiffness"],
                damping=arm_gains["wrist"]["damping"],
            ),
            # --- DESC grippers (±1.5 N·m hardware rating) ---
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=["left_gripper_joint", "right_gripper_joint"],
                effort_limit_sim=1.5,
                velocity_limit_sim=3.5,
                stiffness=77,
                damping=4,
            ),
            # --- Holonomic wheels: preserve USD-authored drive properties ---
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


def make_galbot_golf_tabletop_cfg(
    *,
    init_pose: str = "default",
    gain_profile: str = "default",
    gripper_effort: float | None = None,
    solver_iterations: tuple[int, int] = (128, 4),
):
    """Return a tabletop Golf robot cfg class with the left-ego head camera attached.

    ``init_pose`` is a key of ``GALBOT_GOLF_INIT_POSES``, ``gain_profile`` a key of
    ``GALBOT_GOLF_ARM_GAIN_PROFILES``. Besides the head camera, the defaults match
    ``GalbotGolfTabletopCfg``. ``gripper_effort`` optionally overrides both
    gripper drive torque limits in Nm without changing the shared defaults.
    ``solver_iterations`` sets the articulation's position / velocity counts;
    use the same pair for the environment's scene iteration caps.
    """
    if init_pose not in GALBOT_GOLF_INIT_POSES:
        raise ValueError(f"Unknown init_pose {init_pose!r}. Choose from {sorted(GALBOT_GOLF_INIT_POSES)}.")
    if gripper_effort is not None and not 0 < gripper_effort < float("inf"):
        raise ValueError("gripper_effort must be a finite positive torque in Nm")
    if (
        len(solver_iterations) != 2
        or any(type(v) is not int for v in solver_iterations)
        or not 1 <= solver_iterations[0] <= 255
        or not 0 <= solver_iterations[1] <= 255
    ):
        raise ValueError("solver_iterations must be position in [1, 255] and velocity in [0, 255]")

    @configclass
    class _GalbotGolfTabletopVariantCfg(GalbotGolfTabletopCfg):
        robot = _galbot_golf_robot_cfg(
            init_joint_pos=GALBOT_GOLF_INIT_POSES[init_pose],
            init_pos=GALBOT_GOLF_TABLETOP_ROOT_POS,
            gain_profile=gain_profile,
        )
        left_ego_cam = _LEFT_EGO_CAM

        if gripper_effort is not None:
            robot.actuators["grippers"].effort_limit_sim = gripper_effort
        robot.spawn.articulation_props.solver_position_iteration_count = solver_iterations[0]
        robot.spawn.articulation_props.solver_velocity_iteration_count = solver_iterations[1]

    return _GalbotGolfTabletopVariantCfg
