# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import isaaclab.utils.math as PoseUtils
import torch

from robolab.core.world.world_state import WorldState


def target_eef_pose_to_delta_eef_pose(world: WorldState, target_eef_pose: torch.Tensor, gripper_action: torch.Tensor, ee_frame: str="end_effector", frame_cfg_name: str="frames", env_id: int = 0) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """

        # target position and rotation
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = world.get_frame_pose(frame=ee_frame, frame_cfg_name=frame_cfg_name, as_matrix=True)
        # curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # Test if target_eef_pose is equivalent to curr_pose with some tolerance
        if torch.allclose(target_eef_pose, curr_pose, atol=1e-4):
            # If poses are equivalent, return a zero delta action with the given gripper action
            zero_delta = torch.zeros(1, 6, device=target_eef_pose.device, dtype=target_eef_pose.dtype)
            gripper_action = gripper_action.unsqueeze(0)  # shape: [1, 1]
            action = torch.cat([zero_delta, gripper_action], dim=1)  # shape: [1, 7]
            return action

        # normalized delta position action
        delta_position = target_pos - curr_pos

        # normalized delta rotation action
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        pose_action = torch.cat([delta_position, delta_rotation], dim=-1)  # shape: [1, 6]
        gripper_action = gripper_action.unsqueeze(0)  # shape: [1, 1]
        action = torch.cat([pose_action, gripper_action], dim=1)  # shape: [1, 7]
        return action
