# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a Pi0-family policy backend against the dual-arm Piper robot's tasks.

Mirrors ``policies/pi0_family/run.py``, swapping in
:class:`Pi0PiperDualArmClient` and ``auto_register_piper_envs``.
"""

import argparse
import os
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

PI0_VARIANTS = ["pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"]

parser = argparse.ArgumentParser(
    description="Evaluate a Pi0-family policy backend on Piper tasks.",
    allow_abbrev=False,
)
parser.add_argument("--policy", choices=PI0_VARIANTS, default="pi05",
                    help=("Which Pi0-family variant to evaluate (default: pi05). "
                          "Selects per-variant defaults inside Pi0PiperDualArmClient."))
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                    help="Remote host for policy server (default: localhost).")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                    help="Remote port for policy server (default: 8000).")
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None,
                    help=("Full WebSocket URI for policy server, e.g. wss://host.lepton.run. "
                          "Overrides --remote-host and --remote-port when set."))
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                    help=("Number of actions to execute from each predicted chunk before "
                          "requesting a new one. If omitted, the client uses its per-variant "
                          "default. Must match the model's action_horizon for best performance."))
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                    help="Verbose output (default: False).")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                    help="Debug output (default: False).")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                    help="Enable proprio image data recording (default: False).")
parser.add_argument("--randomize-background", "--randomize_background", action="store_true",
                    help=("Sample a random non-default background per task at registration time. "
                          "Each registered env gets one fixed background; the chosen texture is "
                          "recorded in the per-task env_cfg.json."))
parser.add_argument("--background-seed", "--background_seed", type=int, default=None,
                    help="Seed for reproducible per-task background sampling. Used with --randomize-background.")
parser.add_argument("--livestream-signal-port", "--livestream_signal_port", type=int, default=None,
                    help=("WebRTC livestream signaling port. This is converted to "
                          "--/app/livestream/port before Isaac Sim starts."))
parser.add_argument("--livestream-stream-port", "--livestream_stream_port", type=int, default=None,
                    help=("WebRTC livestream fixed media/host port. This is converted to "
                          "--/app/livestream/fixedHostPort and min/maxHostPort before Isaac Sim starts."))
parser.add_argument("--livestream-http-port", "--livestream_http_port", type=int, default=None,
                    help=("Optional HTTP service port for omni.services.livestream.nvcf. "
                          "Useful when the default 8011 is already occupied."))
parser.add_argument("--dynamic-object", "--dynamic_object", type=str, default=None,
                    help=("Catalog object name to insert into a generated Piper pick-place scene "
                          "(for example: banana, rubiks_cube)."))
parser.add_argument("--dynamic-object-usd", "--dynamic_object_usd", type=str, default=None,
                    help="Object USD path to insert into a generated Piper pick-place scene.")
parser.add_argument("--dynamic-object-name", "--dynamic_object_name", type=str, default=None,
                    help=("Prim/task object name for --dynamic-object-usd, or an override for "
                          "--dynamic-object. Defaults to the catalog name or USD filename."))
parser.add_argument("--dynamic-object-count", "--dynamic_object_count", type=int, default=1,
                    help=("Total number of runtime-spawned objects in the generated scene. "
                          "Includes the target object; additional objects are random distractors."))
episode_length_group = parser.add_mutually_exclusive_group()
episode_length_group.add_argument("--dynamic-seconds-per-object", "--dynamic_seconds_per_object",
                    type=float, default=None, metavar="SECONDS",
                    help=("Policy-control time budget per generated object. The dynamic episode length becomes "
                          "SECONDS × --dynamic-object-count; reset-time sequential drop is excluded."))
episode_length_group.add_argument("--dynamic-episode-length-s", "--dynamic_episode_length_s",
                    type=float, default=None, metavar="SECONDS",
                    help=("Total policy-control time budget for the dynamic task in seconds, independent of "
                          "object count; reset-time sequential drop is excluded."))
parser.add_argument("--dynamic-object-categories", "--dynamic-object-classes",
                    "--dynamic_object_categories", "--dynamic_object_classes",
                    nargs="+", default=None,
                    help=("Catalog class names to sample random objects from, e.g. fruit toy food. "
                          "Used when --dynamic-object-count > 1, or to choose a random target if "
                          "--dynamic-object is omitted."))
parser.add_argument("--dynamic-object-datasets", "--dynamic_object_datasets", nargs="+", default=None,
                    help="Catalog dataset names to sample from, e.g. ycb hot3d objaverse.")
parser.add_argument("--dynamic-object-pool", "--dynamic_object_pool", nargs="+", default=None,
                    help="Explicit catalog object names to sample random objects from.")
parser.add_argument("--dynamic-object-sample-with-replacement", "--dynamic_object_sample_with_replacement",
                    action="store_true",
                    help=("Allow repeated catalog objects among dynamic-scene instances. "
                          "Repeated instances receive unique prim names."))
parser.add_argument("--dynamic-object-seed", "--dynamic_object_seed", type=int, default=None,
                    help="Seed for runtime object sampling and placement.")
parser.add_argument("--dynamic-scene-base", "--dynamic_scene_base", type=str,
                    default="piper_pick_place_base.usda",
                    help="Base scene to receive the dynamic object (default: piper_pick_place_base.usda).")
parser.add_argument("--dynamic-scene-output-dir", "--dynamic_scene_output_dir", type=str, default=None,
                    help=("Directory for generated scene files. Default: "
                          "output/generated_scenes/<timestamp>."))
parser.add_argument("--dynamic-object-pos", "--dynamic_object_pos", type=float, nargs=3,
                    default=(0.328, 0.0, 0.20), metavar=("X", "Y", "Z"),
                    help="Center of the generated object layout in the base scene frame.")
parser.add_argument("--dynamic-object-area", "--dynamic_object_area", type=float, nargs=2,
                    default=(0.22, 0.20), metavar=("X_SIZE", "Y_SIZE"),
                    help="XY area used to spread multiple generated objects.")
parser.add_argument("--dynamic-object-rot", "--dynamic_object_rot", type=float, nargs=4,
                    default=(1.0, 0.0, 0.0, 0.0), metavar=("QW", "QX", "QY", "QZ"),
                    help="Initial object orientation in qw qx qy qz order.")
parser.add_argument("--dynamic-object-scale", "--dynamic_object_scale", type=float, nargs=3,
                    default=(1.0, 1.0, 1.0), metavar=("SX", "SY", "SZ"),
                    help="Initial object scale.")
parser.add_argument("--dynamic-instruction", "--dynamic_instruction", type=str, default=None,
                    help="Instruction prompt for the generated task.")
parser.add_argument("--settle-dynamic-scene", "--settle_dynamic_scene", action="store_true",
                    help="Run physics settling on the generated scene before registering the env.")
parser.add_argument("--dynamic-sequential-drop", "--dynamic_sequential_drop", action="store_true",
                    help=("Release generated dynamic objects one by one during the formal env.reset() "
                          "before policy inference begins. This supersedes the former offline USD "
                          "settling behavior and does not imply --settle-dynamic-scene."))
parser.add_argument("--dynamic-record-setup-video", "--dynamic_record_setup_video", action="store_true",
                    help=("With --dynamic-sequential-drop, prepend the reset-time drop/settle process "
                          "to the same dashboard MP4 as the policy episode. Default: start video only "
                          "after all objects are settled."))
parser.add_argument("--dynamic-settle-steps", "--dynamic_settle_steps", type=int, default=300,
                    help="All-at-once dynamic settle step count (default: 300).")
parser.add_argument("--dynamic-settle-steps-per-object", "--dynamic_settle_steps_per_object",
                    type=int, default=120,
                    help=("Maximum runtime physics steps to wait for stability after each sequential "
                          "release (default: 120)."))
parser.add_argument("--dynamic-settle-final-steps", "--dynamic_settle_final_steps", type=int, default=120,
                    help="Minimum extra runtime settle steps after the final sequential release (default: 120).")
parser.add_argument("--dynamic-settle-max-steps", "--dynamic_settle_max_steps", type=int, default=600,
                    help="Maximum additional runtime settle steps while waiting for final stability (default: 600).")
parser.add_argument("--dynamic-stable-linear-velocity", "--dynamic_stable_linear_velocity", type=float, default=0.02,
                    help="Linear-speed threshold in m/s used by runtime sequential settling (default: 0.02).")
parser.add_argument("--dynamic-stable-angular-velocity", "--dynamic_stable_angular_velocity", type=float, default=0.2,
                    help="Angular-speed threshold in rad/s used by runtime sequential settling (default: 0.2).")
parser.add_argument("--dynamic-stable-frames", "--dynamic_stable_frames", type=int, default=15,
                    help="Consecutive quiet physics frames required for runtime sequential settling (default: 15).")

from robolab.constants import DEFAULT_TASK_SUBFOLDERS, PACKAGE_DIR  # noqa: E402
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

livestream_kit_args = []
if args_cli.livestream_signal_port is not None:
    livestream_kit_args.append(f"--/app/livestream/port={args_cli.livestream_signal_port}")
if args_cli.livestream_stream_port is not None:
    stream_port = args_cli.livestream_stream_port
    livestream_kit_args.extend([
        f"--/app/livestream/fixedHostPort={stream_port}",
        f"--/app/livestream/minHostPort={stream_port}",
        f"--/app/livestream/maxHostPort={stream_port}",
    ])
if args_cli.livestream_http_port is not None:
    livestream_kit_args.append(
        f"--/exts/omni.services.transport.server.http/port={args_cli.livestream_http_port}"
    )
if livestream_kit_args:
    args_cli.kit_args = " ".join(arg for arg in [args_cli.kit_args, *livestream_kit_args] if arg)

# add_common_eval_args defaults --task-dirs to the benchmark folder; the Piper
# task lives under robolab/tasks/piper instead, so retarget it unless the
# caller explicitly overrode --task-dirs.
if args_cli.task_dirs == DEFAULT_TASK_SUBFOLDERS:
    args_cli.task_dirs = ["piper"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.piper.auto_env_registrations_jointpos import auto_register_piper_envs  # noqa: E402
from robolab.tasks.piper.dynamic_scene_utils import (  # noqa: E402
    build_instruction,
    build_runtime_sequential_drop_plan,
    export_dynamic_env,
    generate_dynamic_pick_place_scene_from_specs,
    sample_dynamic_objects,
    sanitize_prim_name,
    settle_scene_in_place,
)

from policies.pi0_family.piper_client import Pi0PiperDualArmClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

dynamic_requested = (
    args_cli.dynamic_object
    or args_cli.dynamic_object_usd
    or args_cli.dynamic_object_count != 1
    or args_cli.dynamic_object_categories
    or args_cli.dynamic_object_datasets
    or args_cli.dynamic_object_pool
    or args_cli.dynamic_object_sample_with_replacement
    or args_cli.dynamic_seconds_per_object is not None
    or args_cli.dynamic_episode_length_s is not None
)

if dynamic_requested:
    if args_cli.dynamic_seconds_per_object is not None and args_cli.dynamic_seconds_per_object <= 0:
        parser.error("--dynamic-seconds-per-object must be positive.")
    if args_cli.dynamic_episode_length_s is not None and args_cli.dynamic_episode_length_s <= 0:
        parser.error("--dynamic-episode-length-s must be positive.")

    target_spec, distractor_specs = sample_dynamic_objects(
        target_object_name=args_cli.dynamic_object,
        target_object_usd_path=args_cli.dynamic_object_usd,
        count=args_cli.dynamic_object_count,
        categories=args_cli.dynamic_object_categories,
        datasets=args_cli.dynamic_object_datasets,
        object_pool=args_cli.dynamic_object_pool,
        sample_with_replacement=args_cli.dynamic_object_sample_with_replacement,
        seed=args_cli.dynamic_object_seed,
        center=args_cli.dynamic_object_pos,
        area=args_cli.dynamic_object_area,
        object_rot=args_cli.dynamic_object_rot,
        scale=args_cli.dynamic_object_scale,
    )
    if args_cli.dynamic_object_name:
        from dataclasses import replace
        target_spec = replace(target_spec, name=sanitize_prim_name(args_cli.dynamic_object_name))

    object_specs = [target_spec, *distractor_specs]
    dynamic_episode_length_s = (
        args_cli.dynamic_seconds_per_object * len(object_specs)
        if args_cli.dynamic_seconds_per_object is not None
        else args_cli.dynamic_episode_length_s
    )
    runtime_drop_plan = None
    initial_positions = None
    if args_cli.dynamic_sequential_drop:
        runtime_drop_plan = build_runtime_sequential_drop_plan(
            object_specs,
            seed=args_cli.dynamic_object_seed,
            steps_per_object=args_cli.dynamic_settle_steps_per_object,
            final_steps=args_cli.dynamic_settle_final_steps,
            max_final_steps=args_cli.dynamic_settle_max_steps,
            stable_linear_velocity=args_cli.dynamic_stable_linear_velocity,
            stable_angular_velocity=args_cli.dynamic_stable_angular_velocity,
            stable_frames=args_cli.dynamic_stable_frames,
            record_setup_video=args_cli.dynamic_record_setup_video,
        )
        initial_positions = {
            name: pose["pos"]
            for name, pose in runtime_drop_plan["holding_poses"].items()
        }

    scene_path = generate_dynamic_pick_place_scene_from_specs(
        target=target_spec,
        distractors=distractor_specs,
        base_scene=args_cli.dynamic_scene_base,
        output_dir=args_cli.dynamic_scene_output_dir,
        initial_positions=initial_positions,
    )
    object_name = sanitize_prim_name(target_spec.name)
    object_names = [target_spec.name, *(spec.name for spec in distractor_specs)]
    if args_cli.settle_dynamic_scene and not args_cli.dynamic_sequential_drop:
        print(f"\033[96m[RoboLab] Settling generated dynamic scene: {scene_path}\033[0m")
        settle_scene_in_place(
            scene_path,
            simulation_app,
            object_names=object_names,
            sequential_drop=False,
            steps=args_cli.dynamic_settle_steps,
            steps_per_object=args_cli.dynamic_settle_steps_per_object,
            final_steps=args_cli.dynamic_settle_final_steps,
        )
    elif args_cli.settle_dynamic_scene and args_cli.dynamic_sequential_drop:
        print(
            "\033[96m[RoboLab] --dynamic-sequential-drop uses reset-time physics; "
            "skipping offline --settle-dynamic-scene.\033[0m"
        )

    instruction = args_cli.dynamic_instruction or build_instruction(object_name)
    export_dynamic_env(
        scene_path,
        object_name,
        instruction,
        object_names=object_names,
        drop_plan=runtime_drop_plan,
        episode_length_s=dynamic_episode_length_s,
    )
    dynamic_task_file = os.path.join(PACKAGE_DIR, "robolab", "tasks", "piper", "piper_dynamic_pick_place_task.py")
    args_cli.task = ["PiperDynamicPickPlaceTask"]
    registration_task = [dynamic_task_file]
    episode_length_message = (
        f"[RoboLab] Policy episode length: {dynamic_episode_length_s:g}s\n"
        if dynamic_episode_length_s is not None else ""
    )
    print(
        f"\033[96m[RoboLab] Generated dynamic Piper scene: {scene_path}\n"
        f"[RoboLab] Target object: {object_name} ({target_spec.usd_path})\n"
        f"[RoboLab] All objects: {', '.join(object_names)}\n"
        + episode_length_message
        + f"[RoboLab] Instruction: {instruction}\033[0m"
    )

else:
    registration_task = args_cli.task

auto_register_piper_envs(
    task_dirs=args_cli.task_dirs,
    task=registration_task,
    randomize_background=args_cli.randomize_background,
    background_seed=args_cli.background_seed,
)


def make_client(args: argparse.Namespace) -> Pi0PiperDualArmClient:
    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        open_loop_horizon=args.open_loop_horizon,
        policy_variant=args.policy,
    )
    return Pi0PiperDualArmClient(**{k: v for k, v in kwargs.items() if v is not None})


def main() -> None:
    run_evaluation(args_cli, policy=args_cli.policy, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
