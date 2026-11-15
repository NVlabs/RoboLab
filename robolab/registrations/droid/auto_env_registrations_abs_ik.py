# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


def auto_register_droid_abs_ik_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None, cameras=None,
                                     randomize_background=False, background_seed=None,
                                     env_postfix="", lighting_selection=None):
    """Register tasks against ``DroidIKActionCfg`` (absolute end-effector pose IK).

    Mirrors :func:`robolab.registrations.droid.auto_env_registrations_jointpos.auto_register_droid_envs`
    but swaps the action config from joint-position to absolute differential-IK and
    appends ``AbsIK`` to every registered env name so they coexist with the
    joint-pos variants.

    lighting_selection selects a complete rig from LIGHTING_SELECTIONS, including its HDR dome. Cannot be combined with randomize_background.
    None preserves existing defaults.
    """
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import (
        DroidCfg,
        DroidIKActionCfg,
        ProprioceptionObservationCfg,
        WristCameraCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    if cameras is None:
        cameras = WRIST_LEFT

    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])

    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg()})

    # WristCameraCfg is robot-mounted (wrist_cam is already attached via DroidCfg).
    # Including it as a scene mixin puts wrist_cam before robot in dataclass field
    # order, causing the camera to spawn before its parent prim exists.
    scene_cameras = [c for c in cameras if c is not WristCameraCfg]

    lighting_cfg = SphereLightCfg
    if lighting_selection is not None:
        if randomize_background:
            raise ValueError("lighting_selection includes its HDR dome; disable randomize_background.")
        from robolab.variations.lighting_selections import get_lighting_selection

        lighting_cfg = get_lighting_selection(lighting_selection)
        background_cfg = None
    elif randomize_background:
        from robolab.variations.backgrounds import find_background_files, generate_background_config

        rng = random.Random(background_seed)
        all_bgs = find_background_files()
        default_bg_path = HomeOfficeBackgroundCfg().dome_light.spawn.texture_file
        all_bgs = [p for p in all_bgs if p != default_bg_path]
        if not all_bgs:
            raise FileNotFoundError(
                "No backgrounds available for randomization after excluding the default."
            )

        def _bg_factory():
            return generate_background_config(rng.choice(all_bgs))

        background_cfg = _bg_factory
    else:
        background_cfg = HomeOfficeBackgroundCfg

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix=env_postfix,
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidIKActionCfg(),
        robot_cfg=DroidCfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=lighting_cfg,
        background_cfg=background_cfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
