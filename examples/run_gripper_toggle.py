# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""
Run a gripper-toggle episode against any registered task.

Holds the arm at its current joint positions while toggling the gripper
between open and closed every `--toggle-every` steps. Useful for sanity-
checking the gripper action path on a new robot/scene.

Usage:
    Basic usage (default task: BananaInBowlTask):
    $ python examples/run_gripper_toggle.py

    Specific task:
    $ python examples/run_gripper_toggle.py --task RubiksCubeTask

    Headless (no viewer, no on-screen rendering):
    $ python examples/run_gripper_toggle.py --task RubiksCubeTask --headless

Output:
    Per-env videos saved to output/run_gripper_toggle/<task_env>/<instruction>[_envN][_viewport].mp4
    fps follows env_cfg.sim render rate.
"""

import argparse
import cv2  # noqa: F401  must be imported before isaaclab
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run gripper-toggle episode on a registered task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs="+", default=None,
                    help="List of tasks to run on (default: all registered).")
parser.add_argument("--tag", nargs="+", default=None,
                    help="List of tags of tasks to run on.")
parser.add_argument("--num-steps", type=int, default=100, help="Number of steps per episode.")
parser.add_argument("--settle-steps", type=int, default=0,
                    help="Physics steps to run after reset before recording starts.")
parser.add_argument("--toggle-every", type=int, default=15, help="Toggle gripper every N steps.")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"],
                    help="Which videos to save: 'all' (sensor + viewport), 'viewport' only, "
                         "'sensor' only, or 'none' (default: all)")

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = args_cli.video_mode != "none"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import PACKAGE_DIR, set_output_dir  # noqa: E402
from episodes import run_gripper_toggle_episode  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402

auto_register_droid_envs()


def main():
    output_dir = os.path.join(PACKAGE_DIR, "output", "run_gripper_toggle")
    os.makedirs(output_dir, exist_ok=True)

    if args_cli.task:
        task_envs = get_envs(task=args_cli.task)
    elif args_cli.tag:
        task_envs = get_envs(tag=args_cli.tag)
    else:
        task_envs = get_envs(task="BananaInBowlTask")
    print(f"Running gripper toggle on {len(task_envs)} environments: {task_envs}")

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        env, env_cfg = create_env(task_env,
                                  device=args_cli.device,
                                  num_envs=args_cli.num_envs,
                                  use_fabric=True)
        try:
            print(f"Running {task_env}: '{env_cfg.instruction}'")
            run_gripper_toggle_episode(
                env,
                env_cfg,
                save_videos=args_cli.save_videos,
                video_mode=args_cli.video_mode,
                headless=args_cli.headless,
                num_steps=args_cli.num_steps,
                settle_steps=args_cli.settle_steps,
                toggle_every=args_cli.toggle_every,
            )
            end_episode(env)
        finally:
            env.close()

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
