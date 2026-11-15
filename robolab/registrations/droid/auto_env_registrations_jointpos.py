# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR

"""
Scene registration:

For the same task, we can register multiple variants. For example, the following script will register something like this:

Task Name                | Environment                       | Config Class                            | Reg | Tags
---------------------------------------------------------------------------------------------------------------------------------------------------
BagelOnPlateTableTask    | BagelOnPlateTableTaskHomeOffice   | BagelOnPlateTableTaskHomeOfficeEnvCfg   | ✓   | all, pick_place
BagelOnPlateTableTask    | BagelOnPlateTableTaskBilliardHall | BagelOnPlateTableTaskBilliardHallEnvCfg | ✓   | all, pick_place
BananaInBowlTableTask    | BananaInBowlTableTaskHomeOffice   | BananaInBowlTableTaskHomeOfficeEnvCfg   | ✓   | all, pick_place
BananaInBowlTableTask    | BananaInBowlTableTaskBilliardHall | BananaInBowlTableTaskBilliardHallEnvCfg | ✓   | all, pick_place

The columns are:
- Task Name: The base task class name (groups variants together)
- Environment: The registered environment name (also the Gymnasium ID)
- Config Class: The generated configuration class name
- Reg: Registration status (✓ = registered with Gymnasium)
- Tags: Tag names this environment belongs to

"""
def auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, lighting_intensity=None, task=None, cameras=None,
                             randomize_background=False, background_seed=None, viewport_camera=None,
                             lazy_sensor_update=True, object_state_obs=False, lighting_selection=None):
    """Automatically discover and register tasks.

    Args:
        task_dirs: Subdirectories to search for tasks.
        lighting_intensity: Optional lighting intensity override. Cannot be combined
              with lighting_selection.
        lighting_selection: Optional complete lighting name from LIGHTING_SELECTIONS. Includes its HDR dome; cannot be combined
              with randomize_background. None preserves existing defaults.
        task: If provided, only register the specified task(s) instead of discovering
              all tasks. Accepts a single task name/filename/path (str) or a list of them.
              Significantly faster when running a subset of tasks.
        cameras: List of camera config classes observed by the policy. Pass one of
              the presets from ``camera_presets`` (e.g. ``WRIST_LEFT``,
              ``WRIST_LEFT_RIGHT_HEAD``) or your own list. Defaults to ``WRIST_LEFT``.
              The viewport camera is attached separately for video recording.
        randomize_background: If True, sample a random background per task at registration time
              (excluding the default home_office background). Each registered env gets one fixed
              background; assignments are reproducible when ``background_seed`` is provided.
        background_seed: Seed for reproducible per-task background sampling. Ignored if
              ``randomize_background`` is False.
        viewport_camera: Camera config class for the video-recording viewport.
              Defaults to ``EgocentricMirroredCameraCfg``.
        lazy_sensor_update: Passed to the scene config. Set False when sensor
              annotators beyond rgb (e.g. depth) must render eagerly in headless
              mode. Default True (IsaacLab default).
        object_state_obs: If True, each registered env gets an
              ``object_state_obs`` observation group with ground-truth
              ``<object>_pos`` (env-local, meters), ``<object>_quat``
              (world-frame w, x, y, z), and ``<object>_vel`` (world-frame)
              terms generated from the task's ``contact_object_list``.
              Default False.
    """
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import (
        DroidCfg,
        DroidJointPositionActionCfg,
        ProprioceptionObservationCfg,
        WristCameraCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    if cameras is None:
        cameras = WRIST_LEFT
    if viewport_camera is None:
        viewport_camera = EgocentricMirroredCameraCfg

    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([viewport_camera])

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
        if lighting_intensity is not None:
            raise ValueError("lighting_selection fixes light intensities; omit lighting_intensity.")
        from robolab.variations.lighting_selections import get_lighting_selection

        lighting_cfg = get_lighting_selection(lighting_selection)
        background_cfg = None
    elif randomize_background:
        from robolab.variations.backgrounds import find_background_files, generate_background_config

        rng = random.Random(background_seed)
        all_bgs = find_background_files()
        # Exclude the default so this is genuinely "besides the default"
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
        env_postfix="",
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidJointPositionActionCfg(),
        robot_cfg=DroidCfg,
        camera_cfg=[*scene_cameras, viewport_camera],
        lighting_cfg=lighting_cfg,
        background_cfg=background_cfg,
        contact_gripper=contact_gripper,
        lazy_sensor_update=lazy_sensor_update,
        object_state_obs=object_state_obs,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
