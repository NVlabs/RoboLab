# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Generate and record a Mechanical Search clutter-box scene."""

import argparse
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher

PACKAGE_DIR = Path(__file__).resolve().parents[1]


TARGET_POSES = {
    "bottom": (0.545, -0.010, 0.070),
    "middle": (0.546, -0.025, 0.116),
    "top": (0.545, -0.080, 0.165),
}


def _write_runtime_scene(target: str, target_placement: str) -> Path:
    if target != "banana":
        raise ValueError("Only --target banana is supported by the current minimal scene.")

    source_scene = Path(PACKAGE_DIR) / "assets" / "scenes" / "mechanical_search_clutter_box.usda"
    scene_dir = source_scene.parent
    runtime_scene = scene_dir / f"mechanical_search_clutter_box_runtime_{target_placement}.usda"
    text = source_scene.read_text()
    x, y, z = TARGET_POSES[target_placement]

    pattern = r'(def "banana" \([\s\S]*?double3 xformOp:translate = \()[^)]+(\)[\s\S]*?uniform token\[\] xformOpOrder)'
    replacement = rf"\g<1>{x:.3f}, {y:.3f}, {z:.3f}\g<2>"
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update banana pose in {source_scene}")

    runtime_scene.write_text(updated)
    return runtime_scene


parser = argparse.ArgumentParser(description="Record a Mechanical Search clutter-box scene.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs="+", default=["MechanicalSearchClutterBoxTask"],
                    help="Task to run. Defaults to MechanicalSearchClutterBoxTask.")
parser.add_argument("--target", default="banana", choices=["banana"], help="Target object.")
parser.add_argument("--target-placement", default="middle", choices=sorted(TARGET_POSES),
                    help="Initial target layer in the clutter box.")
parser.add_argument("--num-steps", type=int, default=100, help="Recorded episode length.")
parser.add_argument("--settle-steps", type=int, default=80,
                    help="Physics steps to run after reset before recording starts.")
parser.add_argument("--toggle-every", type=int, default=15, help="Toggle gripper every N recorded steps.")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"],
                    help="Which videos to save.")

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = args_cli.video_mode != "none"

runtime_scene = _write_runtime_scene(args_cli.target, args_cli.target_placement)
os.environ["ROBOLAB_MECHANICAL_SEARCH_SCENE"] = str(runtime_scene)
os.environ["ROBOLAB_MECHANICAL_SEARCH_TARGET"] = args_cli.target
os.environ["ROBOLAB_MECHANICAL_SEARCH_TARGET_PLACEMENT"] = args_cli.target_placement

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from episodes import run_gripper_toggle_episode  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

auto_register_droid_envs(task=args_cli.task)


def main():
    output_root = Path(PACKAGE_DIR) / "output" / "mechanical_search_visibility"
    task_envs = get_envs(task=args_cli.task)
    print(f"Running {len(task_envs)} environment(s): {task_envs}")
    print(f"Runtime scene: {runtime_scene}")

    for task_env in task_envs:
        scene_output_dir = output_root / task_env / args_cli.target_placement
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        set_output_dir(str(scene_output_dir))
        shutil.copy2(runtime_scene, scene_output_dir / runtime_scene.name)

        env, env_cfg = create_env(
            task_env,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
        )
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
            setup_path = scene_output_dir / "scene_setup.json"
            setup_path.write_text(json.dumps({
                "task": "MechanicalSearchClutterBoxTask",
                "task_env": task_env,
                "target_object": args_cli.target,
                "target_placement": args_cli.target_placement,
                "runtime_scene": str(runtime_scene),
                "recorded_steps": args_cli.num_steps,
                "settle_steps": args_cli.settle_steps,
                "box_asset": "procedural_usda_open_top_box",
                "box_scale": [1.0, 1.0, 1.0],
                "approx_box_size_m": [1.00, 0.50, 0.30],
                "visibility_metrics": "not_computed_yet",
            }, indent=2))
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
