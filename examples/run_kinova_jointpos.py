# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# isort: skip_file

"""Run a rendered joint-position smoke test for the fixed-base Kinova Gen3."""

import argparse
import math
import os
import sys
import traceback

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=180)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

from robolab.constants import PACKAGE_DIR  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import (
    unpack_image_obs,
    unpack_viewport_cams,
)  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.core.utils.isaaclab_compat import as_torch  # noqa: E402
from robolab.robots.kinova_gen3 import GRIPPER_JOINT_COMMANDS  # noqa: E402
from robolab.registrations.kinova.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_kinova_envs,
)


def compose_diagnostic_frame(policy_images, viewport):
    views = [
        ("OVER-SHOULDER", policy_images["over_shoulder_left_camera"]),
        ("WRIST CAMERA", policy_images["wrist_cam"]),
        ("VIEWPORT", viewport),
    ]
    labeled_views = []
    for label, view in views:
        labeled = view.copy()
        cv2.putText(
            labeled,
            label,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        labeled_views.append(labeled)
    return cv2.hconcat(labeled_views)


def main() -> None:
    auto_register_kinova_envs(task="BananaInBowlTask")
    env_name = get_envs(task="BananaInBowlTask")[0]
    env, env_cfg = create_env(
        env_name, device=args_cli.device, num_envs=1, use_fabric=True
    )
    output_dir = os.path.join(PACKAGE_DIR, "output", "kinova_jointpos_smoke")
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "kinova_gen3_joint_position.mp4")
    diagnostic_path = os.path.join(
        output_dir, "kinova_gen3_three_camera_diagnostic.mp4"
    )
    fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    video = VideoWriter(video_path, fps)
    diagnostic_video = VideoWriter(diagnostic_path, fps)

    try:
        obs, _ = env.reset()
        robot = env.scene["robot"]
        home = as_torch(robot.data.default_joint_pos)[0, :7].clone()
        print(f"Environment: {env_name}")
        print(f"Bodies: {robot.data.body_names}")
        print(f"Joints: {robot.data.joint_names}")
        print(f"Action dimension: {env.action_manager.total_action_dim}")

        gripper_error = None
        for step in range(args_cli.num_steps):
            phase = 2.0 * math.pi * step / args_cli.num_steps
            arm_target = home.clone()
            arm_target[0] += 0.25 * math.sin(phase)
            arm_target[5] += 0.15 * math.sin(2.0 * phase)
            gripper_command = 1.0 if math.sin(phase) > 0.0 else 0.0
            action = torch.cat(
                [arm_target, torch.tensor([gripper_command], device=env.device)]
            ).unsqueeze(0)
            obs, _, _, _, _ = env.step(action)
            if step == args_cli.num_steps // 2 - 1:
                joint_positions = dict(
                    zip(robot.data.joint_names, as_torch(robot.data.joint_pos)[0].tolist())
                )
                gripper_error = max(
                    abs(joint_positions[name] - target)
                    for name, target in GRIPPER_JOINT_COMMANDS.items()
                )
            policy_images = unpack_image_obs(obs)
            frame = unpack_viewport_cams(obs).get("combined_image")
            if frame is not None:
                video.write(frame)
                diagnostic_video.write(
                    compose_diagnostic_frame(policy_images, frame)
                )

        if gripper_error is None or gripper_error > 1.0e-3:
            raise RuntimeError(f"Gripper coupling error is {gripper_error}")
        print(f"Maximum gripper coupling error: {gripper_error:.6f} rad")
        tracking_error = torch.max(
            torch.abs(as_torch(robot.data.joint_pos)[0, :7] - arm_target)
        ).item()
        print(f"Final maximum arm tracking error: {tracking_error:.6f} rad")
        print(f"Saved video: {video_path}")
        print(f"Saved diagnostic video: {diagnostic_path}")
    finally:
        video.release()
        diagnostic_video.release()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Terminated with error: {error}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
