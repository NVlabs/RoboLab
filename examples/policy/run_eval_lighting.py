# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Run policy evaluation with lighting variations.

Usage:
    $ python run_eval_lighting.py --headless
    $ python run_eval_lighting.py --task BananaInBowlTableTask --headless

Output:
    Results are saved to: output/<output_folder_name>/
"""

import argparse
import cv2 # Must import this before isaaclab. Do not remove
import os
import re
import traceback
import sys
from isaaclab.app import AppLauncher
from robolab.constants import get_timestamp, DEFAULT_TASK_SUBFOLDERS # noqa

# add argparse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs='+', default=['BananaInBowlTableTask', 'RubiksCubeAndBananaTask'],
                       help="List of tasks to evaluate on ")
parser.add_argument("--tag", nargs='+', default=None,
                       help="List of tags of tasks to evaluate on ")
parser.add_argument("--task-dirs", nargs='+', default=DEFAULT_TASK_SUBFOLDERS,
                       help="List of task directories to evaluate on")
parser.add_argument("--policy", choices=["pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05", "gr00t", "dreamzero", "molmo", "openvla", "openvla_oft"], default="pi05",
                       help="Action-prediction backend to use (default: pi05)")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1,
                       help="Number of sequential runs per task (default: 1). Total episodes = num_runs * num_envs. Prefer increasing --num_envs for more episodes. Only increase --num-runs if you run out of GPU memory with the desired num_envs.")
parser.add_argument("--enable-subtask", "--enable_subtask", action="store_true",
                       help="Enable subtask progress checking (default: False)")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                       help="Enable proprio image data recording (default: False)")
parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                       help="Output folder name under /robolab/output.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                       help="Verbose output (default: False)")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                       help="Debug output (default: False)")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                       help="Remote host for policy server (default: localhost)")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                       help="Remote port for policy server (default: 8000)")
args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import PACKAGE_DIR, set_output_dir # noqa
from robolab.core.environments.runtime import create_env # noqa
from robolab.eval import create_client, run_episode, summarize_run # noqa
from robolab.registrations.droid_jointpos.auto_env_registrations_lighting_variations import auto_register_droid_envs_light_intensity, auto_register_droid_envs_shadows, auto_register_droid_envs_colored_lights # noqa
from robolab.core.logging.recorder_manager import patch_recorder_manager # noqa
from robolab.core.environments.factory import get_envs # noqa
from robolab.core.logging.results import check_all_episodes_complete, check_run_complete # noqa
from robolab.core.logging.results import init_experiment, summarize_experiment_results # noqa
import robolab.constants # noqa

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

patch_recorder_manager()

########################################################
# Lighting variations
########################################################
lighting_variations = [10, 5000]
for intensity in lighting_variations:
    auto_register_droid_envs_light_intensity(task_dirs=args_cli.task_dirs, lighting_intensity=intensity)
auto_register_droid_envs_shadows(task_dirs=args_cli.task_dirs)
auto_register_droid_envs_colored_lights(task_dirs=args_cli.task_dirs)

tasks = []
for task in args_cli.task:
    tasks.extend([task + f"_LightingIntensity{lighting_intensity}" for lighting_intensity in lighting_variations])
    tasks.extend([task + "_Directional"])
    tasks.extend([task + "_RedLight", task + "_BlueLight", task + "_GreenLight"])


def extract_lighting_params(task_env: str) -> dict:
    """Extract lighting parameters from the task environment name."""
    intensity_match = re.search(r'_LightingIntensity(\d+)', task_env)
    if intensity_match:
        return {
            "lighting_intensity": int(intensity_match.group(1)),
            "lighting_color": "natural",
            "lighting_type": "domelight",
        }
    if "_Directional" in task_env:
        return {
            "lighting_intensity": 200,
            "lighting_color": "natural",
            "lighting_type": "directional",
        }
    color_patterns = {
        "_RedLight": "red",
        "_BlueLight": "blue",
        "_GreenLight": "green",
    }
    for pattern, color in color_patterns.items():
        if pattern in task_env:
            return {
                "lighting_intensity": 100,
                "lighting_color": color,
                "lighting_type": "sphere",
            }
    return {
        "lighting_intensity": 5000,
        "lighting_color": "natural",
        "lighting_type": "sphere",
    }
########################################################

def main():
    """Main function."""
    if args_cli.output_folder_name is None:
        args_cli.output_folder_name = get_timestamp() + f"_{args_cli.policy}_lighting_variation"

    output_dir = os.path.join(PACKAGE_DIR, "output", args_cli.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    if tasks:
        task_envs = get_envs(task=tasks)
    elif args_cli.tag:
        task_envs = get_envs(tag=args_cli.tag)
    else:
        task_envs = get_envs()

    num_envs = args_cli.num_envs
    num_runs = args_cli.num_runs
    total_episodes = num_runs * num_envs

    print(f"Output directory: {output_dir}")
    print(f"Running {len(task_envs)} lighting variation environments, {total_episodes} episodes each")

    episode_results_file, episode_results = init_experiment(output_dir)

    for task_env in task_envs:
        task_name = task_env.split("_")[0]
        lighting_params = extract_lighting_params(task_env)
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        if check_all_episodes_complete(episode_results=episode_results, env_name=task_env, num_episodes=total_episodes):
            print(f"\033[96m[RoboLab] Task `{task_env}` already done. Skipping.\033[0m")
            continue

        env, env_cfg = create_env(task_env,
            device=args_cli.device,
            num_envs=num_envs,
            use_fabric=True,
            policy=args_cli.policy)

        client = create_client(
            args_cli.policy,
            remote_host=args_cli.remote_host,
            remote_port=args_cli.remote_port,
        )

        for run_idx in range(num_runs):

            run_episode_ids = [run_idx * num_envs + eid for eid in range(num_envs)]
            if all(check_run_complete(episode_results=episode_results, env_name=task_env, episode=ep_id) for ep_id in run_episode_ids):
                print(f"\033[96m[RoboLab] Task `{task_env}` run `{run_idx}` already done. Skipping.\033[0m")
                continue

            run_name = task_env + f"_{run_idx}"
            print(f"\033[96m[RoboLab] Running {run_name}: '{env_cfg.instruction}' (run {run_idx}, {num_envs} envs)\033[0m")

            env_results, msgs, timing = run_episode(env=env,
                        env_cfg=env_cfg,
                        episode=run_idx,
                        client=client,
                        save_videos=args_cli.save_videos,
                        headless=args_cli.headless)

            episode_results = summarize_run(
                env_results=env_results,
                msgs=msgs,
                env=env,
                env_cfg=env_cfg,
                num_envs=num_envs,
                run_idx=run_idx,
                run_name=run_name,
                task_env=task_env,
                scene_output_dir=scene_output_dir,
                policy=args_cli.policy,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
                enable_subtask_progress=robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING,
                task_name=task_name,
                extra_fields={
                    "lighting_intensity": lighting_params["lighting_intensity"],
                    "lighting_color": lighting_params["lighting_color"],
                    "lighting_type": lighting_params["lighting_type"],
                },
            )

            env.reset_eval_state()

        env.close()

    summarize_experiment_results(episode_results)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
