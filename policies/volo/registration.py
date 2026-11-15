# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoLo-specific environment registration.

Registers the standard DROID environments with depth-rendering cameras: every
free-standing scene camera and the viewport camera get ``"depth"`` added to
their ``data_types``, which in turn generates ``<camera>_depth`` and
``<camera>_pos/_quat/_K`` observation terms (see
``robolab.core.observations.observation_utils``). Lazy sensor updates are
disabled so the depth annotators render eagerly in headless mode.

This is the only place depth is turned on; standard backends register through
``auto_register_droid_envs`` directly and keep their rgb-only render cost.
"""

from robolab.constants import DEFAULT_TASK_SUBFOLDERS

# The externally-installed RoboVoLo content pack drops its tasks under
# robolab/tasks/robovolo; discovery silently skips the folder when the pack is
# not installed. Only VoLo evaluations look there.
VOLO_TASK_SUBFOLDERS = [*DEFAULT_TASK_SUBFOLDERS, "robovolo"]


def register_volo_envs(task_dirs=None, task=None, cameras=None, lighting_selection=None):
    """Register DROID envs for VoLo proxy evaluation (depth cameras enabled).

    Args:
        task_dirs: Subdirectories to search for tasks. Defaults to
            ``VOLO_TASK_SUBFOLDERS`` (the standard benchmark subfolders plus
            the ``robovolo`` content pack).
        task: Optional task name(s) to register instead of discovering all tasks.
        lighting_selection: Optional complete light and HDR preset.
        cameras: Camera preset (list of camera config classes) observed by the
            backend policy. Defaults to ``WRIST_LEFT``. The robot-mounted wrist
            camera is left rgb-only (its config is owned by ``DroidCfg``).
    """
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import WristCameraCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg, with_depth

    if task_dirs is None:
        task_dirs = VOLO_TASK_SUBFOLDERS
    if cameras is None:
        cameras = WRIST_LEFT

    depth_cameras = [c if c is WristCameraCfg else with_depth(c) for c in cameras]

    auto_register_droid_envs(
        task_dirs=task_dirs,
        task=task,
        lighting_selection=lighting_selection,
        cameras=depth_cameras,
        viewport_camera=with_depth(EgocentricMirroredCameraCfg),
        # Depth annotators only populate when sensors update eagerly.
        lazy_sensor_update=False,
    )
