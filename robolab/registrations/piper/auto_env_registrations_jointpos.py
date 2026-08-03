# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import robolab.constants


def auto_register_piper_envs(task_dirs=("piper",), lighting_intensity=None, task=None, cameras=None,
                              randomize_background=False, background_seed=None):
    """Automatically discover and register piper (dual-arm) tasks.

    Mirrors ``robolab.registrations.droid.auto_env_registrations_jointpos.auto_register_droid_envs``,
    swapping in the dual-arm Piper robot/action/observation/contact configs.

    Args:
        task_dirs: Subdirectories under robolab/tasks to search for tasks.
        lighting_intensity: Optional lighting intensity override.
        task: If provided, only register the specified task(s) instead of discovering
              all tasks. Accepts a single task name/filename/path (str) or a list of them.
        cameras: List of camera config classes observed by the policy. Pass one of
              the presets from ``camera_presets`` (e.g. ``ALL``, ``HANDS_ONLY``),
              or your own list. Defaults to ``ALL`` (both hand cameras + first-person).
        randomize_background: If True, sample a random background per task at registration time
              (excluding the default home_office background).
        background_seed: Seed for reproducible per-task background sampling. Ignored if
              ``randomize_background`` is False.
    """
    import random

    from robolab.constants import TASK_DIR
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.registrations.piper.camera_presets import ALL
    from robolab.robots.piper import (
        FirstPersonCameraCfg,
        LeftHandCameraCfg,
        PiperAbsoluteJointPositionActionCfg,
        PiperCfg,
        ProprioceptionObservationCfg,
        RightHandCameraCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    if cameras is None:
        cameras = ALL

    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])

    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg()})

    # All three piper cameras are robot-mounted (hand/first-person links) already
    # attached via PiperCfg, so — like Droid's WristCameraCfg — they must be excluded
    # from the scene-mixin list (image-obs group only), else they'd spawn before
    # their parent robot prim exists.
    _robot_mounted = {LeftHandCameraCfg, RightHandCameraCfg, FirstPersonCameraCfg}
    scene_cameras = [c for c in cameras if c not in _robot_mounted]

    if randomize_background:
        from robolab.variations.backgrounds import find_background_files, generate_background_config

        rng = random.Random(background_seed)
        all_bgs = find_background_files()
        default_bg_path = HomeOfficeBackgroundCfg.dome_light.spawn.texture_file
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
        task_subdirs=list(task_dirs),
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix="",
        observations_cfg=ObservationCfg(),
        actions_cfg=PiperAbsoluteJointPositionActionCfg(),
        robot_cfg=PiperCfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=background_cfg,
        contact_gripper=contact_gripper,
        ee_body_name="hand_link_r",
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=4,
        seed=1,
    )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
