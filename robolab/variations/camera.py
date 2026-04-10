# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass


@configclass
class OverShoulderLeftCameraCfg:
    external_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"
        ),
    )

################################################################################
# Egocentric cameras
################################################################################
@configclass
class EgocentricWideAngleCameraCfg:
    egocentric_wide_angle_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_wide_angle_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.5), rot=(0.653, 0.271, -0.271, -0.653), convention="opengl"
        ),
    )


################################################################################
# Egocentric mirrored, means the camera is looking at the robot from the front,
# Assuming the robot is at origin.
################################################################################
@configclass
class EgocentricMirroredWideAngleHighCameraCfg:
   egocentric_mirrored_wide_angle_high_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_mirrored_wide_angle_high_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.9, 0, 1), rot=(0.653, 0.271, 0.271, 0.653), convention="opengl"
        ),
    )

@configclass
class EgocentricMirroredWideAngleCameraCfg:
   egocentric_mirrored_wide_angle_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_mirrored_wide_angle_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.9, 0, 0.5), rot=(0.653, 0.271, 0.271, 0.653), convention="opengl"
        ),
    )

@configclass
class EgocentricMirroredCameraCfg:
   egocentric_mirrored_camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/egocentric_mirrored_camera",
    # height=720,
    # width=1280,
    height = 480,
    width = 864,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        vertical_aperture=15.29,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(1.5, 0.0, 1.0),
        rot=(0.653, 0.271, 0.271, 0.653),
        convention="opengl"
    ),
)
