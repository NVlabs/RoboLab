# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch


def transform_pose_from_b_to_w_vectorized(pose_b_f1: torch.Tensor, T_f1_w: torch.Tensor):
    """
    Transform poses of bodies from being expressed in frame f1's coordinate system to the world coordinate system.

    This function takes poses of bodies (objects, end-effectors, etc.) that are currently expressed relative
    to frame f1's coordinate system and transforms them to be expressed in the world coordinate system.
    This is a fundamental operation in robotics for converting local coordinates to global coordinates.

    Args:
        pose_b_f1 (torch.Tensor): (N, 4, 4) Poses of bodies expressed in frame f1's coordinate system.
                                  Each 4x4 matrix represents a homogeneous transformation (rotation + translation)
                                  of a body relative to frame f1's origin and orientation.
        T_f1_w (torch.Tensor): (4, 4) Transform from frame f1 to world frame. This matrix describes
                              where frame f1 is located and oriented relative to the world coordinate system.

    Returns:
        torch.Tensor: (N, 4, 4) Poses of the same bodies now expressed in the world coordinate system.
                      These represent the same physical poses, but now relative to the world frame instead of f1.

    Mathematical Operation:
        pose_b_w = T_f1_w × pose_b_f1

        This matrix multiplication applies the transformation chain:
        body → frame f1 → world

    Example:
        If you have robot end-effector poses relative to the robot's base frame (f1) and you want
        to express these same poses in world coordinates (for collision checking, path planning, etc.),
        this function performs that coordinate transformation.

    Note:
        The function is vectorized to handle multiple poses simultaneously for efficiency.
    """
    # Validate input shapes
    if pose_b_f1.shape[-2:] != (4, 4):
        raise ValueError(f"pose_b_f1 must have shape (N, 4, 4), got {pose_b_f1.shape}")

    if T_f1_w.dim() != 2 or T_f1_w.shape != (4, 4):
        raise ValueError(f"T_f1_w must have shape (4, 4), got {T_f1_w.shape}")

    pose_b_w = torch.matmul(T_f1_w.unsqueeze(0), pose_b_f1)
    return pose_b_w

def transform_pose_from_w_to_b_vectorized(pose_b_w: torch.Tensor, T_f1_w: torch.Tensor):
    """
    Transform poses from being expressed in the world coordinate system to being expressed in frame f1's coordinate system.

    Args:
        pose_b_w: (N, 4, 4) or (4, 4) poses in world frame.
        T_f1_w: (4, 4) or (N, 4, 4) transform from frame f1 to world.
                When (N, 4, 4), each env has its own reference frame.
    """
    if pose_b_w.shape[-2:] != (4, 4):
        raise ValueError(f"pose_b_w must have shape (..., 4, 4), got {pose_b_w.shape}")

    if T_f1_w.shape[-2:] != (4, 4):
        raise ValueError(f"T_f1_w must have shape (..., 4, 4), got {T_f1_w.shape}")

    pose_b_f1 = torch.matmul(torch.linalg.inv(T_f1_w), pose_b_w)
    return pose_b_f1


def transform_pose_in_f1_to_f2_vectorized(pose_f1: torch.Tensor, T_f1_w: torch.Tensor, T_f2_w: torch.Tensor):
    """
    Transform poses from being expressed in frame f1's coordinate system to being expressed in frame f2's coordinate system.

    This function takes poses that are currently expressed relative to frame f1's coordinate system,
    and re-expresses them relative to frame f2's coordinate system. This is useful when you want to
    change the reference frame for a set of poses.

    Args:
        pose_f1 (torch.Tensor): (N, 4, 4) Poses currently expressed relative to frame f1's coordinate system.
                               These represent transformations from some objects to frame f1.
        T_f1_w (torch.Tensor): (4, 4) Transform from frame f1 to world frame. This tells us where frame f1 is
                              located and oriented relative to the world frame.
        T_f2_w (torch.Tensor): (4, 4) Transform from frame f2 to world frame. This tells us where frame f2 is
                              located and oriented relative to the world frame.

    Returns:
        torch.Tensor: (N, 4, 4) Poses now expressed relative to frame f2's coordinate system.
                      These represent the same objects/transformations, but now relative to frame f2 instead of f1.

    Mathematical Operation:
        1. Compute T_f2_f1 = T_f1_w^(-1) × T_f2_w  (transform from f1 to f2)
        2. Apply transformation: pose_f2 = pose_f1 × T_f2_f1

    Example:
        If you have object poses relative to a robot's base frame (f1) and want to express them
        relative to a camera frame (f2), this function performs that coordinate transformation.
    """
    # Validate input shapes
    if pose_f1.shape[-2:] != (4, 4):
        raise ValueError(f"pose_f1 must have shape (N, 4, 4), got {pose_f1.shape}")

    if T_f1_w.dim() != 2 or T_f1_w.shape != (4, 4):
        raise ValueError(f"T_f1_w must have shape (4, 4), got {T_f1_w.shape}")

    if T_f2_w.dim() != 2 or T_f2_w.shape != (4, 4):
        raise ValueError(f"T_f2_w must have shape (4, 4), got {T_f2_w.shape}")

    T_f2_f1 = torch.matmul(torch.linalg.inv(T_f1_w), T_f2_w)
    pose_f2 = torch.matmul(pose_f1, T_f2_f1)

    return pose_f2
