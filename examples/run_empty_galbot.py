# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Run the random-action RoboLab smoke test with fixed-base Galbot One Golf.

This is the Galbot counterpart to ``examples/run_empty.py``. It registers ordinary
RoboLab tasks, resets one environment, and samples actions from the registered action
space for a short integration check. It is not a policy or a task-success evaluation.

The fixed-base Golf placement is derived from each task scene's authored ground plane.

Examples:

    python examples/run_empty_galbot.py --task BananaInBowlTask --headless
    python examples/run_empty_galbot.py --task PickDrillTask --headless
"""

import argparse
import os
import sys
import traceback

import cv2  # noqa: F401  # Must import before isaaclab. Do not remove.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a regular RoboLab task with fixed-base Galbot One Golf.")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", nargs="+", default=["BananaInBowlTask"], help="Task class name(s) to smoke-test.")
parser.add_argument("--num-steps", type=int, default=1, help="Number of random-action steps per task.")
parser.add_argument("--save-image", action="store_true", help="Save the final over-shoulder camera frame.")
parser.add_argument("--save-video", action="store_true", help="Save wrist, ego, and front rollout views.")
parser.add_argument(
    "--embodiment",
    choices=("galbot_one_golf_fixed_base",),
    default="galbot_one_golf_fixed_base",
    help="Robot embodiment to register (default: galbot_one_golf_fixed_base).",
)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.envs.utils.spaces import sample_space  # noqa: E402
import torch  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.constants import PACKAGE_DIR, get_output_dir, set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.registrations.galbot.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_galbot_envs,
)
from robolab.core.logging.results import (  # noqa: E402
    dump_results_to_file,
    extract_subtask_info,
    get_all_env_events,
    init_experiment,
    summarize_experiment_results,
    update_experiment_results,
)
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.robots.galbot_golf_definitions import BODY_JOINTS  # noqa: E402


GALBOT_VIDEO_VIEWS = {
    "left_wrist": "left_wrist_cam",
    "left_ego": "left_ego_cam",
    "right_wrist": "right_wrist_cam",
    "front": "front_cam",
}


def _camera_images(obs, group_name: str, env_id: int) -> dict:
    """Copy camera observations independently so mixed resolutions remain valid."""
    return {
        name: image[env_id].detach().cpu().numpy()
        for name, image in obs[group_name].items()
    }


def _initial_hold_action(env, num_envs: int) -> torch.Tensor:
    """Hold the configured reset pose for the first rendered frame."""
    robot = env.scene["robot"]
    missing = [name for name in BODY_JOINTS if name not in robot.data.joint_names]
    if missing:
        raise ValueError(f"Loaded Golf articulation is missing action joints: {missing}")

    indices = [robot.data.joint_names.index(name) for name in BODY_JOINTS]
    body = robot.data.joint_pos[:, indices]
    grippers_open = torch.zeros((num_envs, 2), device=env.device, dtype=body.dtype)
    action = torch.cat((body, grippers_open), dim=1)
    if action.shape[1] != env.action_manager.total_action_dim:
        raise ValueError(
            f"Golf hold action has {action.shape[1]} values; "
            f"environment expects {env.action_manager.total_action_dim}"
        )
    return action


def _write_video_frames(
    obs,
    *,
    num_envs: int,
    num_steps: int,
    video_writers: dict[str, VideoWriter],
    wrote_first_frame: set[str],
) -> None:
    """Write independent wrist, left-ego, and front smoke-test views."""
    for env_id in range(num_envs):
        suffix = f"_env{env_id}" if num_envs > 1 else ""
        prefix = f"empty_0_numsteps{num_steps}{suffix}"
        image_frames = _camera_images(obs, "image_obs", env_id)
        for view_name, camera_name in GALBOT_VIDEO_VIEWS.items():
            frame = image_frames.get(camera_name)
            if frame is None:
                raise KeyError(f"Galbot {view_name!r} camera {camera_name!r} is missing from image_obs")
            writer_key = f"{view_name}:{env_id}"
            video_writers[writer_key].write(frame)
            if writer_key not in wrote_first_frame:
                cv2.imwrite(
                    os.path.join(get_output_dir(), f"{prefix}_{view_name}_first_frame.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
                wrote_first_frame.add(writer_key)


def _run_random_episode(env, env_cfg, *, num_envs: int, num_steps: int, save_image: bool, save_video: bool):
    """Run random actions, recording the reset pose before the first random command."""
    obs, _ = env.reset()
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    video_writers: dict[str, VideoWriter] = {}
    wrote_first_frame: set[str] = set()
    statuses = []
    last_frame = None

    try:
        if save_video:
            for env_id in range(num_envs):
                suffix = f"_env{env_id}" if num_envs > 1 else ""
                prefix = f"empty_0_numsteps{num_steps}{suffix}"
                for view_name in GALBOT_VIDEO_VIEWS:
                    video_writers[f"{view_name}:{env_id}"] = VideoWriter(
                        os.path.join(get_output_dir(), f"{prefix}_{view_name}.mp4"),
                        fps=video_fps,
                    )

            obs, _, _, _, _ = env.step(_initial_hold_action(env, num_envs))
            _write_video_frames(
                obs,
                num_envs=num_envs,
                num_steps=num_steps,
                video_writers=video_writers,
                wrote_first_frame=wrote_first_frame,
            )

        for _ in range(num_steps):
            actions = sample_space(env.single_action_space, device=env.device, batch_size=num_envs)
            obs, _, _, _, info = env.step(actions)
            image_frames = _camera_images(obs, "image_obs", 0)
            last_frame = image_frames.get("over_shoulder_left_camera")
            statuses.append(extract_subtask_info(info))
            if save_video:
                _write_video_frames(
                    obs,
                    num_envs=num_envs,
                    num_steps=num_steps,
                    video_writers=video_writers,
                    wrote_first_frame=wrote_first_frame,
                )
    finally:
        for video_writer in video_writers.values():
            video_writer.release()

    if save_image and last_frame is not None:
        cv2.imwrite(
            os.path.join(get_output_dir(), "empty_0.png"),
            cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR),
        )
    return False, statuses


def _register_envs() -> str:
    """Register fixed-base Golf and return its environment-name postfix."""
    from robolab.robots.galbot_golf import (
        GalbotGolfLeftEgoCameraCfg,
        GalbotGolfFrontCameraCfg,
        GalbotGolfLeftWristCameraCfg,
        GalbotGolfTabletopReplayCfg,
        GalbotGolfRightWristCameraCfg,
        GalbotGolfWholeBodyJointPositionActionCfg,
        ProprioceptionObservationCfg as GalbotGolfProprioceptionObservationCfg,
        contact_gripper as golf_contact_gripper,
    )
    from robolab.variations.camera import OverShoulderLeftCameraCfg, OverShoulderRightCameraCfg

    postfix = "GalbotGolfFixedBaseEmpty"
    auto_register_galbot_envs(
        task=args_cli.task,
        action="whole_body",
        cameras=[
            OverShoulderLeftCameraCfg,
            OverShoulderRightCameraCfg,
            GalbotGolfLeftWristCameraCfg,
            GalbotGolfRightWristCameraCfg,
            GalbotGolfLeftEgoCameraCfg,
            GalbotGolfFrontCameraCfg,
        ],
        env_postfix=postfix,
        robot_cfg=GalbotGolfTabletopReplayCfg,
        actions_cfg=GalbotGolfWholeBodyJointPositionActionCfg(),
        proprioception_cfg=GalbotGolfProprioceptionObservationCfg,
        contact_gripper_cfg=golf_contact_gripper,
    )
    return postfix


def main() -> None:
    robolab.constants.VERBOSE = True
    robolab.constants.DEBUG = False
    robolab.constants.RECORD_IMAGE_DATA = False

    env_postfix = _register_envs()
    task_envs = [name for name in get_envs(task=args_cli.task) if name.endswith(env_postfix)]
    if not task_envs:
        raise RuntimeError(f"No registered environments found for tasks {args_cli.task!r}.")

    output_dir = os.path.join(PACKAGE_DIR, "output", "run_empty_galbot", args_cli.embodiment)
    os.makedirs(output_dir, exist_ok=True)
    episode_results_file, episode_results = init_experiment(output_dir)
    print(f"Running {len(task_envs)} environments: {task_envs}")

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        env, env_cfg = create_env(
            task_env,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
        )
        try:
            print(f"Running {task_env}: '{env_cfg.instruction}'")
            success, messages = _run_random_episode(
                env,
                env_cfg,
                num_envs=args_cli.num_envs,
                num_steps=args_cli.num_steps,
                save_image=args_cli.save_image,
                save_video=args_cli.save_video,
            )

            per_env_events = get_all_env_events(env) or []
            end_episode(env)
            for env_id in range(args_cli.num_envs):
                events = per_env_events[env_id] if env_id < len(per_env_events) else []
                dump_results_to_file(
                    os.path.join(scene_output_dir, f"log_0_env{env_id}.json"),
                    {
                        "schema_version": 2,
                        "task": task_env,
                        "env_id": env_id,
                        "run": 0,
                        "events": events,
                    },
                    append=False,
                )

            final_message = messages[-1] if messages and messages[-1] is not None else {}
            run_summary = {
                "env_name": task_env,
                "episode": 0,
                "success": success,
                "instruction": env_cfg.instruction,
            }
            if robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING:
                run_summary.update(score=final_message.get("score"), reason=final_message.get("info"))
            episode_results = update_experiment_results(
                run_summary=run_summary,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
            )
        finally:
            env.close()

    summarize_experiment_results(episode_results)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Terminated with error: {exc}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
