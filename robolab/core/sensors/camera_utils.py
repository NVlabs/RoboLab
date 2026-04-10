# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from isaaclab.sensors import Camera


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
        if isinstance(sensor, Camera):
            dict_cameras[name] = sensor
    return dict_cameras