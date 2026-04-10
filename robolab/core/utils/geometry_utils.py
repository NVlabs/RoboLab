# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import torch
from isaaclab.utils.math import unmake_pose
from scipy.spatial.transform import Rotation as R


def get_bbox_corners(lower, upper):
    corners = np.array([
        [lower[0], lower[1], lower[2]],
        [lower[0], lower[1], upper[2]],
        [lower[0], upper[1], lower[2]],
        [lower[0], upper[1], upper[2]],
        [upper[0], lower[1], lower[2]],
        [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], lower[2]],
        [upper[0], upper[1], upper[2]],
    ])
    return corners

def transform_bbox_to_pose(lower, upper, translation, quaternion_wxyz, inverse=True):
    # Generate all 8 corners of the box
    corners = np.array([
        [lower[0], lower[1], lower[2]],
        [lower[0], lower[1], upper[2]],
        [lower[0], upper[1], lower[2]],
        [lower[0], upper[1], upper[2]],
        [upper[0], lower[1], lower[2]],
        [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], lower[2]],
        [upper[0], upper[1], upper[2]],
    ])
    # Convert wxyz to xyzw for scipy
    quat_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    r = R.from_quat(quat_xyzw)
    # Build the homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = translation

    if inverse:
        # Compute the inverse transformation
        R_inv = r.as_matrix().T
        t_inv = -R_inv @ translation
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        T_use = T_inv
    else:
        T_use = T

    # Convert corners to homogeneous coordinates
    corners_h = np.hstack([corners, np.ones((corners.shape[0], 1))])
    # Apply transformation
    transformed_corners_h = (T_use @ corners_h.T).T
    # Return only the xyz part
    return transformed_corners_h[:, :3]


def pose_from_pos_quat(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Build a 4×4 pose given xy (Tensor[2]), z, and quaternion."""
    import isaaclab.utils.math as math_utils
    rot = math_utils.matrix_from_quat(quat)
    return math_utils.make_pose(pos, rot) # Returns a [..., 4, 4] tensor

def spatial_condition_check_position_based(pose1: torch.Tensor,
                                           pose2: torch.Tensor,
                                           spatial_condition: str,
                                           mirrored: bool=False):
    valid_spatial_conditions = ["left_of", "right_of", "in_front_of", "behind"]
    if spatial_condition not in valid_spatial_conditions:
        raise ValueError(f"Invalid spatial condition: {spatial_condition}")

    pos1, _ = unmake_pose(pose1)
    pos2, _ = unmake_pose(pose2)

    if spatial_condition == "left_of":
        return pos1[1] > pos2[1]

    elif spatial_condition == "right_of":
        return pos1[1] < pos2[1]

    elif spatial_condition == "in_front_of":
        return pos1[0] > pos2[0]

    elif spatial_condition == "behind":
        return pos1[0] < pos2[0]

    else:
        raise ValueError(f"Invalid spatial condition: {spatial_condition}")


def spatial_condition_check_vector_based(pose1: torch.Tensor,
                                       pose2: torch.Tensor,
                                       spatial_condition: str,
                                       mirrored: bool=False,
                                       cone_deg: int=45):
    """
    Check if the spatial condition is satisfied between two objects based on their vectors.
    This is a more general check than the position based check, and should yield a more observation-based check.

    Supports both single-env (4, 4) and batched (N, 4, 4) poses.
    Returns bool for single-env, Tensor(N,) for batched.
    """
    valid_spatial_conditions = ["left_of", "right_of", "in_front_of", "behind"]
    if spatial_condition not in valid_spatial_conditions:
        raise ValueError(f"Invalid spatial condition: {spatial_condition}")

    batched = pose1.dim() == 3  # (N, 4, 4) vs (4, 4)

    pos1, _ = unmake_pose(pose1)
    pos2, _ = unmake_pose(pose2)

    # Compute vector from obj2 to obj1
    vector_12 = pos1 - pos2  # (3,) or (N, 3)

    if batched:
        # Batched path: (N, 3) tensors
        vector_12_xy = vector_12.clone()
        vector_12_xy[..., 2] = 0.0  # zero out z
        norm_12_xy = torch.norm(vector_12_xy, dim=-1)  # (N,)

        cone_rad = torch.deg2rad(torch.tensor(cone_deg, dtype=torch.float32, device=pose1.device))
        cos_cone = torch.cos(cone_rad)

        valid = norm_12_xy > 1e-6  # (N,)

        # Determine which axis/sign to check
        if (spatial_condition == "left_of" and not mirrored) or \
           (spatial_condition == "right_of" and mirrored):
            # y > 0 and cos(angle to +y) >= cos_cone
            cos_theta = vector_12_xy[..., 1] / norm_12_xy.clamp(min=1e-8)
            success = valid & (vector_12_xy[..., 1] > 0) & (cos_theta >= cos_cone)

        elif (spatial_condition == "right_of" and not mirrored) or \
             (spatial_condition == "left_of" and mirrored):
            cos_theta = vector_12_xy[..., 1] / norm_12_xy.clamp(min=1e-8)
            success = valid & (vector_12_xy[..., 1] < 0) & (-cos_theta >= cos_cone)

        elif (spatial_condition == "behind" and not mirrored) or \
             (spatial_condition == "in_front_of" and mirrored):
            cos_theta = vector_12_xy[..., 0] / norm_12_xy.clamp(min=1e-8)
            success = valid & (vector_12_xy[..., 0] > 0) & (cos_theta >= cos_cone)

        elif (spatial_condition == "in_front_of" and not mirrored) or \
             (spatial_condition == "behind" and mirrored):
            cos_theta = vector_12_xy[..., 0] / norm_12_xy.clamp(min=1e-8)
            success = valid & (vector_12_xy[..., 0] < 0) & (-cos_theta >= cos_cone)
        else:
            raise ValueError("Invalid spatial_condition.")

        return success  # Tensor(N,) bool

    else:
        # Single-env path: original scalar logic
        vector_12_xy = torch.tensor([vector_12[0], vector_12[1], 0.0], dtype=vector_12.dtype)
        norm_12_xy = torch.norm(vector_12_xy)

        x_axis = torch.tensor([1, 0, 0], dtype=torch.float32)
        y_axis = torch.tensor([0, 1, 0], dtype=torch.float32)

        cone_rad = torch.deg2rad(torch.tensor(cone_deg, dtype=torch.float32))
        cos_cone = torch.cos(cone_rad)

        success = False

        if norm_12_xy > 1e-6:
            if (spatial_condition == "left_of" and not mirrored) or \
                (spatial_condition == "right_of" and mirrored):
                cos_theta = torch.dot(vector_12_xy, y_axis) / norm_12_xy
                success = bool(vector_12_xy[1] > 0) and bool(cos_theta >= cos_cone)

            elif (spatial_condition == "right_of" and not mirrored) or \
                (spatial_condition == "left_of" and mirrored):
                cos_theta = torch.dot(vector_12_xy, y_axis) / norm_12_xy
                success = bool(vector_12_xy[1] < 0) and bool(-cos_theta >= cos_cone)

            elif (spatial_condition == "behind" and not mirrored) or \
                 (spatial_condition == "in_front_of" and mirrored):
                cos_theta = torch.dot(vector_12_xy, x_axis) / norm_12_xy
                success = bool(vector_12_xy[0] > 0) and bool(cos_theta >= cos_cone)

            elif (spatial_condition == "in_front_of" and not mirrored) or \
                (spatial_condition == "behind" and mirrored):
                cos_theta = torch.dot(vector_12_xy, x_axis) / norm_12_xy
                success = bool(vector_12_xy[0] < 0) and bool(-cos_theta >= cos_cone)

            else:
                raise ValueError("Invalid spatial_condition. Must be 'left_of', 'right_of', 'in_front_of', or 'behind'.")

        return success
