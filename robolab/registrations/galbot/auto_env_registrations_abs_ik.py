# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register RoboLab tasks with fixed-base Galbot One Golf differential IK."""

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


def auto_register_galbot_abs_ik_envs(
    task_dirs=DEFAULT_TASK_SUBFOLDERS,
    task=None,
    cameras=None,
    env_postfix="GalbotGolfFixedBaseDifferentialIK",
    include_viewport=True,
    background_cfg=None,
    robot_cfg=None,
    actions_cfg=None,
    proprioception_cfg=None,
    contact_gripper_cfg=None,
    dt=1 / 60,
    render_interval=6,
    decimation=6,
):
    """Discover and register tasks with fixed-base Golf absolute TCP control."""
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.robots.galbot_golf import (
        GalbotGolfDifferentialIKActionCfg,
        GalbotGolfLeftWristCameraCfg,
        GalbotGolfRightWristCameraCfg,
        GalbotGolfTabletopCfg,
        ProprioceptionObservationCfg,
        contact_gripper,
        gripper_closure_cfg,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import (
        EgocentricMirroredWideAngleHighCameraCfg,
        OverShoulderLeftCameraCfg,
        OverShoulderRightCameraCfg,
    )
    from robolab.variations.lighting import SphereLightCfg

    robot_cfg = robot_cfg or GalbotGolfTabletopCfg
    actions_cfg = actions_cfg or GalbotGolfDifferentialIKActionCfg()
    proprioception_cfg = proprioception_cfg or ProprioceptionObservationCfg
    contact_gripper_cfg = contact_gripper_cfg or contact_gripper
    background_cfg = background_cfg or HomeOfficeBackgroundCfg
    cameras = cameras or [
        OverShoulderLeftCameraCfg,
        OverShoulderRightCameraCfg,
        GalbotGolfLeftWristCameraCfg,
        GalbotGolfRightWristCameraCfg,
    ]

    def is_robot_attached(camera_cls):
        camera_cfg = camera_cls()
        return any(
            hasattr(value, "prim_path") and "/robot/" in value.prim_path
            for name in dir(camera_cfg)
            if not name.startswith("_")
            for value in [getattr(camera_cfg, name)]
        )

    scene_cameras = [camera for camera in cameras if not is_robot_attached(camera)]
    obs_groups = {
        "image_obs": generate_image_obs_from_cameras(cameras)(),
        "proprio_obs": proprioception_cfg(),
    }
    camera_cfg = list(scene_cameras)
    if include_viewport:
        obs_groups["viewport_cam"] = generate_image_obs_from_cameras([EgocentricMirroredWideAngleHighCameraCfg])()
        camera_cfg.append(EgocentricMirroredWideAngleHighCameraCfg)
    ObservationCfg = generate_obs_cfg(obs_groups)

    generated_envs = auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix=env_postfix,
        observations_cfg=ObservationCfg(),
        actions_cfg=actions_cfg,
        robot_cfg=robot_cfg,
        camera_cfg=camera_cfg,
        lighting_cfg=SphereLightCfg,
        background_cfg=background_cfg,
        contact_gripper=contact_gripper_cfg,
        gripper_closure_cfg=gripper_closure_cfg,
        dt=dt,
        render_interval=render_interval,
        decimation=decimation,
        seed=1,
    )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table

        print_env_table()

    return generated_envs
