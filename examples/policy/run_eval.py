# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Run policy evaluation across multiple tasks.

This script runs policy evaluation on multiple registered tasks, supporting various
policy backends (pi0, etc.) with options for subtask tracking and result logging.

Supports multi-env: each "run" spawns num_envs parallel episodes.
Total episodes = num_runs * num_envs.

Usage:
    Run on all registered tasks:
    $ python run_eval.py

    Run on specific tasks:
    $ python run_eval.py --task BananaInBowlTask RubiksCubeTask

    Run on a tag:
    $ python run_eval.py --tag spatial

    Use specific policy:
    $ python run_eval.py --policy pi05

    Run multiple episodes with 2 parallel envs:
    $ python run_eval.py --num-runs 2 --num_envs 4

Output:
    Results are saved to: output/<output_folder_name>/
"""

import argparse
import cv2 # Must import this before isaaclab. Do not remove
import os
import re
import traceback
import sys
from collections import Counter
from isaaclab.app import AppLauncher
from robolab.constants import get_timestamp, DEFAULT_TASK_SUBFOLDERS # noqa

# add argparse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs='+', default=None,
                       help="List of tasks to evaluate on ")
parser.add_argument("--tag", nargs='+', default=None,
                       help="List of tags of tasks to evaluate on ")
parser.add_argument("--task-dirs", nargs='+', default=DEFAULT_TASK_SUBFOLDERS,
                       help="List of task directories to evaluate on")
parser.add_argument("--policy",
                    choices=["pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05", "gr00t", "dreamzero", "molmo", "openvla", "openvla_oft"], default="pi05",
                       help="Action-prediction backend to use (default: pi05)")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1,
                       help="Number of sequential runs per task (default: 1). Total episodes = num_runs * num_envs. Prefer increasing --num_envs for more episodes. Only increase --num-runs if you run out of GPU memory with the desired num_envs.")
parser.add_argument("--enable-subtask", "--enable_subtask", action="store_true",
                       help="Enable subtask progress checking (default: False)")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                       help="Enable proprio image data recording (default: False)")
parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                       help="Output folder name under /robolab/output. Default is <timestamp>_<policy>. If you provide the output folder name for a previous run, the script will skip the tasks and episodes that have already been run.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                       help="Verbose output (default: False)")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                       help="Debug output (default: False)")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                       help="Remote host for policy server (default: localhost)")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                       help="Remote port for policy server (default: 8000)")
parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default",
                       help="Which instruction variant to use when a task defines multiple (default, vague, specific, etc.)")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"],
                    help="Which videos to save: 'all' (sensor + viewport), 'viewport' only, 'sensor' only, or 'none' (default: all)")
# parse the arguments
args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = args_cli.video_mode != "none"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import PACKAGE_DIR, set_output_dir # noqa
from episode import run_episode # noqa
from robolab.core.environments.runtime import create_env # noqa
from robolab.core.logging.recorder_manager import patch_recorder_manager # noqa
from robolab.core.environments.factory import get_envs # noqa
from robolab.core.utils.print_utils import print_experiment_summary # noqa
from robolab.core.logging.results import check_all_episodes_complete, check_run_complete, dump_results_to_file # noqa
from robolab.core.logging.results import init_experiment, update_experiment_results, summarize_experiment_results, get_final_subtask_info # noqa
from robolab.core.metrics import load_demo_data, compute_episode_metrics # noqa
from robolab.core.logging.results import extract_subtask_status_changes # noqa
from robolab.core.task.status import StatusCode, get_status_name # noqa
from robolab.core.utils.file_utils import load_file # noqa
import robolab.constants # noqa

# Update robolab.constants module settings from command line arguments
robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

# Fix recorder manager
patch_recorder_manager()

# Run automatic factory generation before main
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs # noqa
auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)

EVENT_STATUS_CODES = {
    StatusCode.WRONG_OBJECT_GRABBED_FAILURE,
    StatusCode.GRIPPER_HIT_TABLE,
    StatusCode.WRONG_OBJECT_DETACHED,
    StatusCode.OBJECT_BUMPED,
    StatusCode.OBJECT_MOVED,
    StatusCode.OBJECT_OUT_OF_SCENE,
    StatusCode.OBJECT_TIPPED_OVER,
    StatusCode.TARGET_OBJECT_DROPPED,
    StatusCode.GRIPPER_HIT_OBJECT,
    StatusCode.MULTIPLE_OBJECTS_GRABBED,
    StatusCode.GRIPPER_FULLY_CLOSED,
}

def _extract_events_from_log(log_file: str) -> dict:
    """Extract error events from a log file. Returns dict of event counts and details."""
    if not os.path.exists(log_file):
        return {}

    log_data = load_file(log_file)
    if log_data is None:
        return {}

    status_changes = extract_subtask_status_changes(log_data)
    if not status_changes:
        return {}

    event_counts: Counter = Counter()
    wrong_objects_grabbed: list[str] = []

    for change in status_changes:
        status_code = change.get("status", 0)
        if status_code not in EVENT_STATUS_CODES:
            continue

        event_name = get_status_name(status_code)
        if event_name.endswith("_FAILURE"):
            event_name = event_name[:-8]

        event_counts[event_name] += 1

        if status_code == StatusCode.WRONG_OBJECT_GRABBED_FAILURE:
            info = change.get("info", "")
            match = re.search(r"Wrong object grabbed: '([^']+)'", info)
            if match:
                wrong_objects_grabbed.append(match.group(1))

    events: dict = {}
    for event_name, count in event_counts.items():
        events[event_name] = count
    if wrong_objects_grabbed:
        events["wrong_objects_grabbed"] = wrong_objects_grabbed

    return events

def main():
    """Main function."""
    if args_cli.output_folder_name is None:
        args_cli.output_folder_name = get_timestamp() + f"_{args_cli.policy}"
        if args_cli.instruction_type != "default":
            args_cli.output_folder_name += f"_{args_cli.instruction_type}"

    output_dir = os.path.join(PACKAGE_DIR, "output", args_cli.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    if args_cli.task:
        task_envs = get_envs(task=args_cli.task)
        filter_str = f"tasks: {', '.join(args_cli.task)}"
    elif args_cli.tag:
        task_envs = get_envs(tag=args_cli.tag)
        filter_str = f"tags: {', '.join(args_cli.tag)}"
    else:
        task_envs = get_envs()
        filter_str = "all"

    num_envs = args_cli.num_envs
    num_runs = args_cli.num_runs
    total_episodes = num_runs * num_envs

    print_experiment_summary(
        task_envs=task_envs,
        filter_str=filter_str,
        num_envs=num_envs,
        num_episodes=total_episodes,
        policy=args_cli.policy,
        instruction_type=args_cli.instruction_type,
        output_dir=output_dir,
    )

    episode_results_file, episode_results = init_experiment(output_dir)

    for task_env in task_envs:
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
            instruction_type=args_cli.instruction_type,
            policy=args_cli.policy)

        for run_idx in range(num_runs):

            # Check if all episodes in this run are already complete
            run_episode_ids = [run_idx * num_envs + eid for eid in range(num_envs)]
            if all(check_run_complete(episode_results=episode_results, env_name=task_env, episode=ep_id) for ep_id in run_episode_ids):
                print(f"\033[96m[RoboLab] Task `{task_env}` run `{run_idx}` already done. Skipping.\033[0m")
                continue

            # Policy
            if args_cli.instruction_type != "default":
                run_name = task_env + f"_{args_cli.instruction_type}_{run_idx}"
            else:
                run_name = task_env + f"_{run_idx}"
            print(f"\033[96m[RoboLab] Running {run_name}: '{env_cfg.instruction}' (run {run_idx}, {num_envs} envs)\033[0m")

            env_results, msgs, timing = run_episode(env=env,
                        env_cfg=env_cfg,
                        episode=run_idx,
                        save_videos=args_cli.save_videos,
                        video_mode=args_cli.video_mode,
                        headless=args_cli.headless,
                        remote_host=args_cli.remote_host,
                        remote_port=args_cli.remote_port)

            # Get per-env final info for incomplete episodes
            final_infos = get_final_subtask_info(env, env_id=None)  # list[dict | None]

            # Split msgs (list[list[dict] | None]) into per-env log streams
            per_env_msgs: dict[int, list] = {eid: [] for eid in range(num_envs)}
            for step_infos in msgs:
                if step_infos is None:
                    for eid in range(num_envs):
                        per_env_msgs[eid].append(None)
                else:
                    for eid in range(num_envs):
                        per_env_msgs[eid].append(step_infos[eid] if eid < len(step_infos) else None)

            # Write per-env log files and extract per-env events
            per_env_events: dict[int, dict] = {}
            for eid in range(num_envs):
                log_file = os.path.join(scene_output_dir, f"log_{run_idx}_env{eid}.json")
                dump_results_to_file(log_file, per_env_msgs[eid], append=False)
                per_env_events[eid] = _extract_events_from_log(log_file)

            dt = env_cfg.sim.dt * env_cfg.decimation

            # Emit one run_summary per env
            for r in env_results:
                env_id = r['env_id']
                episode_id = run_idx * num_envs + env_id

                # Compute trajectory metrics from per-run HDF5
                hdf5_path = os.path.join(scene_output_dir, f"run_{run_idx}.hdf5")
                demo_key = f"demo_{env_id}"
                traj_data = load_demo_data(hdf5_path, demo_key)
                traj_metrics = compute_episode_metrics(traj_data, dt=dt) if traj_data else None

                events = per_env_events.get(env_id, {})

                run_summary = {
                    "env_name": task_env,
                    "task_name": env_cfg._task_name,
                    "run_name": run_name,
                    "run": run_idx,
                    "episode": episode_id,
                    "env_id": env_id,
                    "policy": args_cli.policy,
                    "instruction": env_cfg.instruction,
                    "instruction_type": args_cli.instruction_type,
                    "attributes": env_cfg._task_attributes,
                    "success": r['success'],
                    "episode_step": r['step'],
                    "duration": r['step'] * dt if r['step'] else 0,
                    "dt": dt,
                    "metrics": traj_metrics if traj_metrics else {},
                    "events": events if events else {},
                    "timing": timing,
                }

                if robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING:
                    env_msgs = per_env_msgs.get(env_id, [])
                    # Find last non-None msg for this env
                    last_msg = None
                    for m in reversed(env_msgs):
                        if m is not None:
                            last_msg = m
                            break

                    if last_msg is not None:
                        run_summary["score"] = last_msg.get("score", None)
                        run_summary["reason"] = last_msg.get("info", None)
                    else:
                        run_summary["score"] = None
                        run_summary["reason"] = None

                    # For failed episodes, use per-env final_info
                    final_info = final_infos[env_id] if final_infos else None
                    if not r['success'] and final_info is not None:
                        run_summary["reason"] = final_info.get("info", run_summary.get("reason"))

                episode_results = update_experiment_results(run_summary=run_summary, episode_results=episode_results, episode_results_file=episode_results_file)

            # Reset eval state for next run (unfreeze all envs)
            env.reset_eval_state()

        env.close()

    # This will print the results to the terminal, summarized.
    # Alternatively, you can run `python analysis/read_results.py <output_dir>` to read the results from the file.
    summarize_experiment_results(episode_results, show_timing=True)

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
