# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# isort: skip_file
import argparse
import cv2  # Must import before isaaclab. Do not remove.
import math
import sys
import traceback
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Absolute IK controller demo + eef_frame command tests.")
parser.add_argument("--task", type=str, default="BananaInBowlTask")
parser.add_argument("--hold_steps", type=int, default=30, help="Steps to hold each target before measuring.")
parser.add_argument("--pos_delta", type=float, default=0.05, help="Translation magnitude per test (m).")
parser.add_argument("--rot_deg", type=float, default=20.0, help="Rotation magnitude per test (degrees).")
parser.add_argument("--pos_tol_mm", type=float, default=5.0, help="Position pass/fail tolerance (mm).")
parser.add_argument("--rot_tol_deg", type=float, default=2.0, help="Rotation pass/fail tolerance (degrees).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa
from robolab.core.environments.factory import get_envs  # noqa
from robolab.core.environments.runtime import create_env, end_episode  # noqa
from robolab.core.utils.isaaclab_compat import as_torch, quat_isaaclab_to_wxyz  # noqa
from robolab.core.utils.vis_utils import visualize_axes  # noqa
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs  # noqa
from robolab.robots.droid import EEF_OFFSET_ROT  # noqa

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False


# DroidIKActionCfg tracks base_link (body_offset = identity). To let the user
# specify targets in eef_frame coordinates, the demo converts on the command side:
# target_base_quat = target_eef_quat ⊗ R_offset⁻¹. See DroidIKActionCfg's docstring
# for why we don't put R_offset in body_offset.
_EEF_OFFSET_ROT_T = torch.tensor(EEF_OFFSET_ROT, dtype=torch.float32)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2, both in (w, x, y, z)."""
    w1, x1, y1, z1 = q1[0].item(), q1[1].item(), q1[2].item(), q1[3].item()
    w2, x2, y2, z2 = q2[0].item(), q2[1].item(), q2[2].item(), q2[3].item()
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=torch.float32)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    return torch.tensor([q[0].item(), -q[1].item(), -q[2].item(), -q[3].item()], dtype=torch.float32)


def quat_about_axis(angle_rad: float, axis_idx: int) -> torch.Tensor:
    """Unit quaternion for rotation by `angle_rad` about world x/y/z (axis_idx 0/1/2)."""
    q = torch.zeros(4, dtype=torch.float32)
    q[0] = math.cos(angle_rad / 2)
    q[1 + axis_idx] = math.sin(angle_rad / 2)
    return q


def quat_angle_mag(q: torch.Tensor) -> float:
    """Shortest-path rotation angle of a unit quaternion, in radians ∈ [0, π]."""
    w = abs(q[0].item())
    return 2.0 * math.acos(max(-1.0, min(1.0, w)))


def main():
    auto_register_droid_abs_ik_envs(task=args_cli.task)
    task_envs = get_envs(task=args_cli.task)
    if not task_envs:
        print(f"No environments found for task '{args_cli.task}'. Check --task name.")
        simulation_app.close()
        return
    env_name = task_envs[0]
    print(f"Running environment: {env_name}")
    env, _ = create_env(env_name, num_envs=1, use_fabric=True)
    obs, _ = env.reset()

    # Read eef_frame's world pose from the FrameTransformer; this is the same frame
    # the IK action drives (base_link @ body_offset = eef_frame).
    frames = env.scene["frames"]
    eef_idx = frames.data.target_frame_names.index("eef_frame")

    def read_eef_pose():
        return (
            as_torch(frames.data.target_pos_w)[0, eef_idx, :].cpu().clone(),
            quat_isaaclab_to_wxyz(
                as_torch(frames.data.target_quat_w)[0, eef_idx, :]
            ).cpu().clone(),
        )

    initial_pos, initial_quat = read_eef_pose()
    print(f"initial eef_frame pos:  ({initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f})")
    print(f"initial eef_frame quat: ({initial_quat[0]:.3f}, {initial_quat[1]:.3f}, {initial_quat[2]:.3f}, {initial_quat[3]:.3f})")

    d = args_cli.pos_delta
    rot_rad = math.radians(args_cli.rot_deg)
    # Each entry: (label, target_pos in world, target_quat in world)
    # All targets are absolute and built from the captured initial pose, so failures
    # in earlier tests don't bias later targets.
    pos_zero = torch.zeros(3, dtype=torch.float32)
    rot_id = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    test_cases = [
        ("hold initial",          pos_zero,                                       rot_id),
        ("translate +x",          torch.tensor([+d, 0.0, 0.0], dtype=torch.float32), rot_id),
        ("translate -x",          torch.tensor([-d, 0.0, 0.0], dtype=torch.float32), rot_id),
        ("translate +y",          torch.tensor([0.0, +d, 0.0], dtype=torch.float32), rot_id),
        ("translate -y",          torch.tensor([0.0, -d, 0.0], dtype=torch.float32), rot_id),
        ("translate +z",          torch.tensor([0.0, 0.0, +d], dtype=torch.float32), rot_id),
        ("translate -z",          torch.tensor([0.0, 0.0, -d], dtype=torch.float32), rot_id),
        ("rotate world +X",       pos_zero, quat_about_axis(+rot_rad, 0)),
        ("rotate world -X",       pos_zero, quat_about_axis(-rot_rad, 0)),
        ("rotate world +Y",       pos_zero, quat_about_axis(+rot_rad, 1)),
        ("rotate world -Y",       pos_zero, quat_about_axis(-rot_rad, 1)),
        ("rotate world +Z",       pos_zero, quat_about_axis(+rot_rad, 2)),
        ("rotate world -Z",       pos_zero, quat_about_axis(-rot_rad, 2)),
    ]

    action = torch.zeros(1, 8, device=env.device)
    pos_tol = args_cli.pos_tol_mm / 1000.0
    rot_tol = math.radians(args_cli.rot_tol_deg)
    n_pass = n_fail = 0

    # For per-step debug prints during a specific test (the "+x" case currently
    # diverges; this gives a frame-by-frame trace of joints + eef pose).
    VERBOSE_TESTS = {"translate +x"}
    robot = env.scene["robot"]
    panda_joint_indices = [i for i, n in enumerate(robot.data.joint_names) if n.startswith("panda_joint")]
    panda_joint_names = [robot.data.joint_names[i] for i in panda_joint_indices]

    for name, dpos, dquat_world in test_cases:
        target_pos = initial_pos + dpos
        # Q = world-frame rotation applied on top of initial orientation
        target_quat = quat_mul(dquat_world, initial_quat)

        print()
        print(f"=== TEST: {name} ===")
        print(f"  target_pos:  ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
        print(f"  target_quat: ({target_quat[0]:.3f}, {target_quat[1]:.3f}, {target_quat[2]:.3f}, {target_quat[3]:.3f})")

        # Update the single goal-pose axis triad at /Visuals/Axes/target_goal.
        # visualize_axes deletes and respawns the prim under the same name.
        visualize_axes(target_pos, target_quat, "target_goal", axis_length=0.2)

        # Convert eef_frame target → base_link target (since IK now tracks base_link).
        # body_offset.pos = 0, so positions are equal; only the orientation needs to
        # be un-offset: target_base_quat = target_eef_quat ⊗ R_offset⁻¹.
        action_pos = target_pos
        action_quat = quat_mul(target_quat, quat_inv(_EEF_OFFSET_ROT_T))
        action[0, :3] = action_pos.to(env.device)
        action[0, 3:7] = action_quat.to(env.device)
        action[0, 7] = 0.0

        verbose = name in VERBOSE_TESTS
        if verbose:
            print(f"  per-step trace: joints={panda_joint_names}")

        terminated_during_test = False
        for step in range(args_cli.hold_steps):
            obs, _, term, trunc, _ = env.step(action)
            if verbose:
                eef_p, eef_q = read_eef_pose()
                joint_pos = as_torch(robot.data.joint_pos)[0, panda_joint_indices].cpu().tolist()
                joints_str = " ".join(f"{j:+.3f}" for j in joint_pos)
                print(
                    f"  [s{step:03d}] eef pos=({eef_p[0]:+.3f},{eef_p[1]:+.3f},{eef_p[2]:+.3f}) "
                    f"quat=({eef_q[0]:+.3f},{eef_q[1]:+.3f},{eef_q[2]:+.3f},{eef_q[3]:+.3f}) "
                    f"q=[{joints_str}]"
                )
            if term or trunc:
                print(f"  [step {step}] episode terminated/truncated — resetting and re-reading initial pose")
                obs, _ = env.reset()
                initial_pos, initial_quat = read_eef_pose()
                terminated_during_test = True
                break

        if terminated_during_test:
            print("  result: SKIPPED (episode reset mid-test)")
            continue

        actual_pos, actual_quat = read_eef_pose()
        pos_err = torch.norm(actual_pos - target_pos).item()
        quat_err = quat_angle_mag(quat_mul(actual_quat, quat_inv(target_quat)))

        pos_ok = pos_err <= pos_tol
        rot_ok = quat_err <= rot_tol
        passed = pos_ok and rot_ok
        n_pass += int(passed)
        n_fail += int(not passed)

        print(f"  actual_pos:  ({actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f})")
        print(f"  actual_quat: ({actual_quat[0]:.3f}, {actual_quat[1]:.3f}, {actual_quat[2]:.3f}, {actual_quat[3]:.3f})")
        print(f"  pos_err = {pos_err * 1000:6.2f} mm   [{'PASS' if pos_ok else 'FAIL'}  tol={args_cli.pos_tol_mm} mm]")
        print(f"  rot_err = {math.degrees(quat_err):6.2f} deg  [{'PASS' if rot_ok else 'FAIL'}  tol={args_cli.rot_tol_deg} deg]")
        print(f"  result: {'PASS' if passed else 'FAIL'}")

    print()
    print(f"=== SUMMARY: {n_pass} pass / {n_fail} fail / {len(test_cases) - n_pass - n_fail} skipped ===")

    end_episode(env)
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
