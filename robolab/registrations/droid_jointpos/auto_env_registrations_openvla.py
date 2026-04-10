# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

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
def auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, lighting_intensity=None):
    """Automatically discover and register all available tasks."""
    # Import auto environment factory for automatic task registration
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.registrations.droid_jointpos.observations import ImageObsCfg, ProprioceptionObservationCfg
    from robolab.robots.droid import (
        DroidCfg,
        DroidJointPositionActionCfg,
        ProprioceptionObservationCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg, OverShoulderLeftCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    print(f"Registering tasks in {task_dirs}")

    subdir_tags = {subdir: subdir for subdir in task_dirs}

    # Generate Observations
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg()})

    for subdir, tag in subdir_tags.items():

        # Auto-discover and create environments for all task files
        _ = auto_discover_and_create_cfgs(
            task_dir=TASK_DIR,
            task_subdirs=[subdir],
            # add_tags=[tag],
            pattern="*.py",  # Match files ending with _task.py or .py
            env_prefix="",
            env_postfix="",
            observations_cfg=ObservationCfg(),
            actions_cfg=DroidJointPositionActionCfg(),
            robot_cfg=DroidCfg,
            camera_cfg=[OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg],
            lighting_cfg=SphereLightCfg,
            background_cfg=HomeOfficeBackgroundCfg,
            contact_gripper=contact_gripper,
            dt=1 / (60 * 2),
            render_interval=8,
            decimation=8,
            seed=1,
        )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
