# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Test environment initialization, results logging, and summarization functionality.

This script tests that environments can be created and run successfully without
executing full episodes. Useful for quick validation of environment setup and
configuration without running policies. If no policy or tag is specified, it will run all tasks in the spatial tag.

Usage:
    Basic usage:
    $ python run_empty.py --headless[optional]

    Basic usage with specific task:
    $ python run_empty.py --task PickCubeTask

    Test multiple tasks:
    $ python run_empty.py --task PickCubeTask PlaceCubeTask

    Test a tag:
    $ python run_empty.py --tag spatial

    Run headless (no rendering):
    $ python run_empty.py --task MyTask --headless

Requirements:
    - Task must be registered in the environment factory

Output:
    Results are saved to: output/run_empty_env/
    - Episode logs: <task_name>/log_<episode>.json
    - Summary: results.json and episode_results.json
"""

import argparse
import cv2 # Must import this before isaaclab. Do not remove
import os
import json
import sys
import traceback

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs='+', default=None,
                       help="List of tasks to evaluate on ")
parser.add_argument("--tag", nargs='+', default=None,
                       help="List of tags of tasks to evaluate on ")
parser.add_argument("--num-steps", type=int, default=50, help="Number of steps to run the environment for.")

# parse the arguments
args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import PACKAGE_DIR, set_output_dir # noqa
from episodes import run_empty_episode # noqa
from robolab.core.environments.runtime import create_env, end_episode # noqa
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs # noqa
from robolab.core.logging.recorder_manager import patch_recorder_manager # noqa
from robolab.core.environments.factory import get_envs # noqa
from robolab.constants import get_timestamp # noqa
from robolab.core.logging.results import dump_results_to_file, summarize_experiment_results # noqa
from robolab.core.logging.results import init_experiment, update_experiment_results # noqa
import robolab.constants # noqa

# Fix recorder manager
patch_recorder_manager()
# Run automatic factory generation before main
auto_register_droid_envs()

robolab.constants.VERBOSE = True
robolab.constants.DEBUG = False
robolab.constants.RECORD_IMAGE_DATA = False

def main():
    """Main function."""
    num_episodes = 1
    output_dir = os.path.join(PACKAGE_DIR, "output", "run_empty_env")
    os.makedirs(output_dir, exist_ok=True)

    if args_cli.task:
        task_envs = get_envs(task=args_cli.task)
    elif args_cli.tag:
        task_envs = get_envs(tag=args_cli.tag)
    else:
        task_envs = get_envs()
    print(f"Running {len(task_envs)} environments: {task_envs}")

    episode_results_file, episode_results = init_experiment(output_dir)

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        env, env_cfg = create_env(task_env,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True)

        for i in range(num_episodes):

            # Policy
            run_name = task_env + f"_{i}"
            print(f"Running {run_name}: '{env_cfg.instruction}'")

            succ, msgs = run_empty_episode(env,
                env_cfg=env_cfg,
                num_envs=args_cli.num_envs,
                num_steps=args_cli.num_steps,
                episode=i,
                save_image=False,
                save_videos=False)

            end_episode(env)

            dump_results_to_file(os.path.join(scene_output_dir, f"log_{i}.json"), msgs, append=False)

            # Update run results
            if robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING:
                if len(msgs) > 0 and msgs[-1] is not None:
                    subtask_info = msgs[-1]
                    score = subtask_info.get("score", None)
                    info = subtask_info.get("info", None)
                else:
                    score = None
                    info = None
                run_summary = {"env_name": task_env,
                                "episode": i,
                                "success": succ,
                                "instruction": env_cfg.instruction,
                                "score": score,
                                "reason": info}
            else:
                run_summary = {"env_name": task_env,
                                "episode": i,
                                "success": succ,
                                "instruction": env_cfg.instruction,
                                }

            episode_results = update_experiment_results(run_summary=run_summary, episode_results=episode_results, episode_results_file=episode_results_file)

        env.close()

    summarize_experiment_results(episode_results)
    simulation_app.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
