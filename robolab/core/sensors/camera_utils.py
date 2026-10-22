# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.sensors import Camera, CameraCfg


def is_camera(sensor) -> bool:
    """Recognize public and backend-factory Isaac Lab camera sensors."""
    return isinstance(sensor, Camera) or isinstance(getattr(sensor, "cfg", None), CameraCfg)


def get_cameras(scene):
    """
    Get all camera sensors.

    Example:
        env = create_env(...)
        contact_sensors = get_contact_sensors(env.scene)

    Args:
        scene (InteractiveScene): The scene to get the cameras from.
    """
    dict_cameras = {}
    for name, sensor in scene.sensors.items():
        if is_camera(sensor):
            dict_cameras[name] = sensor
    return dict_cameras
