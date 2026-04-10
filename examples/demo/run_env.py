# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demo for running a generated environment configuration.

This script demonstrates how to create an environment from a task file
and run episodes with gripper toggling or policy control.

Usage:
    $ python run_env.py

    Run headless:
    $ python run_env.py --headless
"""
import argparse
import os
import cv2 # Must import this before isaaclab. Do not remove

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo on using the mimic joints for Robotiq 140 gripper.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.activate_contact_sensors = True
args_cli.save_videos = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import tyro

from robolab.constants import PACKAGE_DIR, get_timestamp, DEFAULT_OUTPUT_DIR, TASK_DIR

from episodes import run_gripper_toggle_episode, run_prerecorded_episode, run_empty_episode

from robolab.core.environments.runtime import create_env # noqa
from robolab.core.environments.config import generate_env_cfg_from_task # noqa
from robolab.policies.droid_jointpos.observations import ImageObsCfg, ProprioceptionObservationCfg # noqa
from robolab.robots.droid import DroidCfg, contact_gripper, DroidJointPositionActionCfg # noqa
from robolab.variations.camera import OverShoulderLeftCameraCfg # noqa
from robolab.variations.lighting import SphereLightCfg # noqa


def main():
    """Main function."""

    num_episodes = 2

    # # Setup environment
    EnvCfg, _ = generate_env_cfg_from_task(
        task_file_path=f"{TASK_DIR}/sauce_bottles_crate_task.py",
        env_name="SauceBottles",
        robot_cfg=DroidCfg,
        camera_cfg=OverShoulderLeftCameraCfg,
        lighting_cfg=SphereLightCfg,
        contact_gripper=contact_gripper,
        actions_cfg=DroidJointPositionActionCfg(),
        observations_cfg=ObservationCfg(),
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        eye=(1.5, 0.0, 1.0),
        lookat=(0.2, 0.0, 0.0),
        env_spacing=2.0,
        num_envs=1,
        seed=0,
    )

    env_cfg = EnvCfg()

    env, _ = create_env(scene=env_cfg,
                     device=args_cli.device,
                     num_envs=args_cli.num_envs,
                     use_fabric=True)

    output_dir = os.path.join(env.output_dir, get_timestamp())
    for i in range(num_episodes):
        env.output_dir = os.path.join(output_dir, f"episode_{i}")

        # # Pre-recorded episode
        # run_prerecorded_episode(env,
        #             save_videos=args_cli.save_videos,
        #             headless=args_cli.headless)

        # Just toggle the gripper episode
        run_gripper_toggle_episode(env,
                    save_videos=args_cli.save_videos,
                    headless=args_cli.headless)


        # # Policy (import from policy.episode import run_episode)
        # from policy.episode import run_episode
        # run_episode(env=env,
        #             env_cfg=env_cfg,
        #             save_videos=args_cli.save_videos,
        #             headless=args_cli.headless)


        # run_empty_episode(env, num_envs=args_cli.num_envs, num_steps=10)

    env.close()
    return

if __name__ == "__main__":
    main()
    # args = tyro.cli(main)
