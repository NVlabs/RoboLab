# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Run pre-recorded episodes from HDF5 files for testing and validation.

This script replays previously recorded robot trajectories to verify task behavior,
subtask progress tracking, and environment consistency.

Usage:
    Basic usage with default task:
    $ python run_recorded.py

    Specify a custom task:
    $ python run_recorded.py --task RubiksCubeOrBananaTask

    Run headless (no rendering):
    $ python run_recorded.py --task MyTask --headless

Requirements:
    - Recorded data must exist at: examples/demo/recorded_data/<task>/data.hdf5
    - Task must be registered in the environment factory

Output:
    Results are saved to: output/playback_output/
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
from robolab.constants import PACKAGE_DIR, set_output_dir # noqa

# add argparse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--task", '-t', type=str, default="RubiksCubeAndBananaTask", help="Task name to run.")
parser.add_argument("--recorded-data-folder", '--dir', type=str, default=os.path.join(PACKAGE_DIR, 'examples', 'demo', 'recorded_data'), help="Recorded data folder to run.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--enable-subtask", "--enable_subtask", action="store_true", help="Enable subtask progress checking.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from episodes import run_prerecorded_episode_hdf5 # noqa
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

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.VERBOSE = True
robolab.constants.DEBUG = False
robolab.constants.RECORD_IMAGE_DATA = False

def main():
    """Main function."""
    task = args_cli.task
    output_dir = os.path.join(PACKAGE_DIR, "output", "playback_"+os.path.basename(args_cli.recorded_data_folder) + "_" + task)
    os.makedirs(output_dir, exist_ok=True)


    # Check if task is a folder inside the recorded_data folder
    if os.path.isdir(os.path.join(args_cli.recorded_data_folder, task)):
        hdf5_path = os.path.join(args_cli.recorded_data_folder, task, 'data.hdf5')
    else:
        raise ValueError(f"Task {task} not found in {args_cli.recorded_data_folder}")

    task_envs = get_envs(task=task)
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

        # Can be used to loop through multiple runs; but just 1 for now.
        for i in [0]:

            run_name = task_env + f"_run{i}"
            print(f"Running {run_name}: '{env_cfg.instruction}'")

            env_results, msgs = run_prerecorded_episode_hdf5(env,
                hdf5_path=hdf5_path,
                episode=i,
                save_videos=args_cli.save_videos,
                headless=args_cli.headless)

            dump_results_to_file(os.path.join(scene_output_dir, f"log_{i}.json"), msgs, append=False)

            # Emit one run_summary per env (each env is an independent episode)
            for r in env_results:
                episode_id = i * args_cli.num_envs + r['env_id']

                run_summary = {
                    "env_name": task_env,
                    "run": i,
                    "episode": episode_id,
                    "env_id": r['env_id'],
                    "success": r['success'],
                    "step": r['step'],
                    "instruction": env_cfg.instruction,
                }

                if robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING:
                    if len(msgs) > 0 and msgs[-1] is not None:
                        subtask_info = msgs[-1]
                        run_summary["score"] = subtask_info.get("score", None)
                        run_summary["reason"] = subtask_info.get("info", None)

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
