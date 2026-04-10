# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
predicate_logic: "Is condition X true?"

Spatial predicates: left_of(), above_top(), inside(), enclosed()
State predicates: stationary(), upright()
Multi-object predicates: check_stacked(), in_line(), between()

Functions take a `world: WorldState` as their first parameter.
All functions support an `env_id` parameter:
  env_id=None → vectorized, returns Tensor(num_envs,) of bools
  env_id=int  → single env, returns bool (backward compat)
"""

from typing import Callable

import numpy as np
import torch

import robolab.constants
from robolab.constants import DEBUG
from robolab.core.utils.geometry_utils import spatial_condition_check_vector_based
from robolab.core.utils.transform_utils import transform_pose_from_w_to_b_vectorized
from robolab.core.world.world_state import get_world


def get_task_conditional_func(func_name: str):
    """Load a conditional function by name from the conditionals module."""
    from robolab.core.task import conditionals
    from robolab.core.utils.function_loader import load_callable_from_module
    return load_callable_from_module(conditionals, func_name)


def evaluate_logicals(results: list[bool], logical: str, N: int = 1) -> bool:
    """Evaluates a list of boolean results according to the logical selection strategy."""
    if logical == "all":
        result = all(results)
    elif logical == "any":
        result = any(results)
    elif logical == "choose":
        result = results.count(True) == N
    else:
        raise ValueError(f"Invalid logical: {logical}")
    if DEBUG:
        print(f"evaluate_logicals: Evaluating {len(results)} boolean results with logical='{logical}' (N={N}) -> {result}")
    return result


def evaluate_logicals_vectorized(results: list[torch.Tensor], logical: str, N: int = 1) -> torch.Tensor:
    """Vectorized version — each result is Tensor(num_envs,), returns Tensor(num_envs,)."""
    stacked = torch.stack(results, dim=0)  # (num_objects, num_envs)
    if logical == "all":
        return stacked.all(dim=0)
    elif logical == "any":
        return stacked.any(dim=0)
    elif logical == "choose":
        return stacked.sum(dim=0) == N
    else:
        raise ValueError(f"Invalid logical: {logical}")


def evaluate_spatial_condition(
    env,
    object: str | list[str],
    condition_func: Callable,
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
    **kwargs
):
    """
    Generic helper to evaluate a spatial condition across multiple objects.

    Args:
        env: Environment or WorldState
        object: Single object name or list of object names
        condition_func: Function that takes (world, obj, env_id=..., **kwargs) and returns
                       bool (when env_id=int) or Tensor(num_envs,) (when env_id=None)
        logical: "all", "any", or "choose"
        K: Number of objects for "choose" logical
        env_id: None → vectorized, int → single env
        **kwargs: Additional arguments to pass to condition_func
    """
    if logical not in ["any", "all", "choose"]:
        raise ValueError(f"Invalid logical: {logical}")

    world = get_world(env)
    objects = [object] if isinstance(object, str) else list(object)

    if env_id is not None:
        # Single env — scalar path
        results = [condition_func(world, obj, env_id=env_id, **kwargs) for obj in objects]
        result = evaluate_logicals(results, logical, K)
        if DEBUG:
            print(f"evaluate_spatial_condition: Evaluating '{condition_func.__name__}' on {objects} with logical='{logical}' (K={K}) -> {result}")
        return result
    else:
        # Vectorized path
        results = [condition_func(world, obj, env_id=None, **kwargs) for obj in objects]
        result = evaluate_logicals_vectorized(results, logical, K)
        if DEBUG:
            print(f"evaluate_spatial_condition: Evaluating '{condition_func.__name__}' on {objects} with logical='{logical}' (K={K}) [vectorized]")
        return result


#########################################################
# Helpers for dual scalar/tensor logic
#########################################################

def _and(a, b):
    """Logical AND that works for both bool and Tensor."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return a & b
    return a and b

def _not(a):
    """Logical NOT that works for both bool and Tensor."""
    if isinstance(a, torch.Tensor):
        return ~a
    return not a


#########################################################
# Spatial relationship functions
#########################################################

def _spatial_condition(world, object: str, reference_object: str,
                       spatial_condition: str,
                       frame_of_reference: str = "robot",
                       mirrored: bool = False,
                       cone_deg: int = 45,
                       env_id: int | None = None):
    """
    Internal helper to evaluate directional spatial relationships between two objects.

    Args:
        env_id: None → Tensor(num_envs,) bool, int → bool
    """
    pose1 = world.get_pose(object, as_matrix=True, env_id=env_id)
    pose2 = world.get_pose(reference_object, as_matrix=True, env_id=env_id)

    if frame_of_reference == "world":
        result = spatial_condition_check_vector_based(pose1, pose2, spatial_condition, mirrored=mirrored, cone_deg=cone_deg)
    else:
        ref_pose = world.get_pose(frame_of_reference, as_matrix=True, env_id=env_id)
        pose1_r = transform_pose_from_w_to_b_vectorized(pose1, ref_pose)
        pose2_r = transform_pose_from_w_to_b_vectorized(pose2, ref_pose)
        result = spatial_condition_check_vector_based(pose1_r, pose2_r, spatial_condition, mirrored=mirrored, cone_deg=cone_deg)
    # Ensure scalar bool for single-env path
    if env_id is not None and not isinstance(result, bool):
        result = bool(result)
    if DEBUG:
        print(f"_spatial_condition: Checking if '{object}' is {spatial_condition} '{reference_object}' (frame={frame_of_reference}, mirrored={mirrored}, cone_deg={cone_deg}) -> {result}")
    return result


# Directional spatial checks
def left_of(world, object: str, reference_object: str,
            frame_of_reference: str = "robot", mirrored: bool = False, cone_deg: int = 45, env_id: int | None = None) -> bool:
    """Check if object is to the left of reference_object."""
    return _spatial_condition(world, object, reference_object, "left_of", frame_of_reference, mirrored, cone_deg, env_id=env_id)


def right_of(world, object: str, reference_object: str,
             frame_of_reference: str = "robot", mirrored: bool = False, cone_deg: int = 45, env_id: int | None = None) -> bool:
    """Check if object is to the right of reference_object."""
    return _spatial_condition(world, object, reference_object, "right_of", frame_of_reference, mirrored, cone_deg, env_id=env_id)


def in_front_of(world, object: str, reference_object: str,
                frame_of_reference: str = "robot", mirrored: bool = False, cone_deg: int = 45, env_id: int | None = None) -> bool:
    """Check if object is in front of reference_object."""
    return _spatial_condition(world, object, reference_object, "in_front_of", frame_of_reference, mirrored, cone_deg, env_id=env_id)


def behind(world, object: str, reference_object: str,
           frame_of_reference: str = "robot", mirrored: bool = False, cone_deg: int = 45, env_id: int | None = None) -> bool:
    """Check if object is behind reference_object."""
    return _spatial_condition(world, object, reference_object, "behind", frame_of_reference, mirrored, cone_deg, env_id=env_id)


# Containment spatial checks — these use get_bbox which now supports vectorized
def _bbox_min_max(corners, env_id):
    """Extract min/max from corners. Returns (min, max) as appropriate type."""
    if env_id is not None:
        # corners is list[Gf.Vec3d]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        zs = [c[2] for c in corners]
        return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
    else:
        # corners is Tensor(N, 8, 3)
        mins = corners.min(dim=1).values  # (N, 3)
        maxs = corners.max(dim=1).values  # (N, 3)
        return mins, maxs


def enclosed(world, inside_obj: str, outside_obj: str, tolerance: float = 0.0, env_id: int | None = None):
    """Check if the bounding box of inside_obj is fully enclosed in outside_obj's bbox."""
    container_corners, _ = world.get_bbox(outside_obj, env_id=env_id)
    inside_corners, _ = world.get_bbox(inside_obj, env_id=env_id)

    if env_id is not None:
        min_x, max_x, min_y, max_y, min_z, max_z = _bbox_min_max(container_corners, env_id)
        min_x -= tolerance; max_x += tolerance
        min_y -= tolerance; max_y += tolerance
        min_z -= tolerance; max_z += tolerance
        result = all(
            min_x <= corner[0] <= max_x and
            min_y <= corner[1] <= max_y and
            min_z <= corner[2] <= max_z
            for corner in inside_corners
        )
        if DEBUG:
            print(f"enclosed: '{inside_obj}' fully enclosed in '{outside_obj}' (tol={tolerance}) -> {result}")
        return result
    else:
        # Vectorized: container_corners (N, 8, 3), inside_corners (N, 8, 3)
        c_mins, c_maxs = _bbox_min_max(container_corners, env_id)
        # All 8 inside corners must be within container bounds
        x_ok = (inside_corners[:, :, 0] >= c_mins[:, 0:1] - tolerance) & (inside_corners[:, :, 0] <= c_maxs[:, 0:1] + tolerance)
        y_ok = (inside_corners[:, :, 1] >= c_mins[:, 1:2] - tolerance) & (inside_corners[:, :, 1] <= c_maxs[:, 1:2] + tolerance)
        z_ok = (inside_corners[:, :, 2] >= c_mins[:, 2:3] - tolerance) & (inside_corners[:, :, 2] <= c_maxs[:, 2:3] + tolerance)
        all_corners_inside = (x_ok & y_ok & z_ok).all(dim=1)  # (N,)
        return all_corners_inside


def inside(world, inside_obj: str, outside_obj: str, tolerance: float = 0.0, env_id: int | None = None):
    """Check if the centroid of inside_obj is inside outside_obj's bbox."""
    container_corners, _ = world.get_bbox(outside_obj, env_id=env_id)
    _, centroid = world.get_bbox(inside_obj, env_id=env_id)

    if env_id is not None:
        min_x, max_x, min_y, max_y, min_z, max_z = _bbox_min_max(container_corners, env_id)
        result = bool(
            min_x - tolerance <= centroid[0] <= max_x + tolerance and
            min_y - tolerance <= centroid[1] <= max_y + tolerance and
            min_z - tolerance <= centroid[2] <= max_z + tolerance
        )
        if DEBUG:
            print(f"inside: centroid of '{inside_obj}' in bbox of '{outside_obj}' (tol={tolerance}) -> {result}")
        return result
    else:
        mins, maxs = _bbox_min_max(container_corners, env_id)
        x_ok = (centroid[:, 0] >= mins[:, 0] - tolerance) & (centroid[:, 0] <= maxs[:, 0] + tolerance)
        y_ok = (centroid[:, 1] >= mins[:, 1] - tolerance) & (centroid[:, 1] <= maxs[:, 1] + tolerance)
        z_ok = (centroid[:, 2] >= mins[:, 2] - tolerance) & (centroid[:, 2] <= maxs[:, 2] + tolerance)
        return x_ok & y_ok & z_ok


def in_opentop_container(world, inside_obj: str, outside_obj: str, tolerance: float = 0.0, env_id: int | None = None):
    """Check if centroid of inside_obj is in an open-top container (2x height allowance)."""
    container_corners, _ = world.get_bbox(outside_obj, env_id=env_id)
    _, centroid = world.get_bbox(inside_obj, env_id=env_id)

    if env_id is not None:
        min_x, max_x, min_y, max_y, min_z, max_z = _bbox_min_max(container_corners, env_id)
        height = max_z - min_z
        result = bool(
            min_x - tolerance <= centroid[0] <= max_x + tolerance and
            min_y - tolerance <= centroid[1] <= max_y + tolerance and
            min_z - tolerance <= centroid[2] <= max_z + height
        )
        if DEBUG:
            print(f"in_opentop_container: centroid of '{inside_obj}' in open-top '{outside_obj}' (tol={tolerance}) -> {result}")
        return result
    else:
        mins, maxs = _bbox_min_max(container_corners, env_id)
        height = maxs[:, 2] - mins[:, 2]
        x_ok = (centroid[:, 0] >= mins[:, 0] - tolerance) & (centroid[:, 0] <= maxs[:, 0] + tolerance)
        y_ok = (centroid[:, 1] >= mins[:, 1] - tolerance) & (centroid[:, 1] <= maxs[:, 1] + tolerance)
        z_ok = (centroid[:, 2] >= mins[:, 2] - tolerance) & (centroid[:, 2] <= maxs[:, 2] + height)
        return x_ok & y_ok & z_ok


# Vertical spatial checks
def _vertical_check(world, obj: str, surface: str, tolerance: float, z_margin: float,
                    mode: str, use_max_z: bool, obj_above: bool, env_id: int | None = None):
    """Shared logic for above_top, above_bottom, below_top, below_bottom.

    Args:
        use_max_z: True → reference is max_z (top), False → reference is min_z (bottom)
        obj_above: True → object must be >= ref, False → object must be <= ref
    """
    surface_corners, _ = world.get_bbox(surface, env_id=env_id)

    if env_id is not None:
        min_x, max_x, min_y, max_y, min_z, max_z = _bbox_min_max(surface_corners, env_id)
        ref_z = max_z if use_max_z else min_z

        if mode == "centroid":
            _, centroid = world.get_bbox(obj, env_id=env_id)
            xy_ok = (min_x - tolerance <= centroid[0] <= max_x + tolerance and
                     min_y - tolerance <= centroid[1] <= max_y + tolerance)
            if obj_above:
                z_ok = centroid[2] >= ref_z + z_margin
            else:
                z_ok = centroid[2] <= ref_z - z_margin
            return bool(xy_ok and z_ok)
        else:  # bbox mode
            obj_corners, centroid = world.get_bbox(obj, env_id=env_id)
            xy_ok = (min_x - tolerance <= centroid[0] <= max_x + tolerance and
                     min_y - tolerance <= centroid[1] <= max_y + tolerance)
            if obj_above:
                z_ok = all(corner[2] >= ref_z + z_margin for corner in obj_corners)
            else:
                z_ok = all(corner[2] <= ref_z - z_margin for corner in obj_corners)
            return bool(xy_ok and z_ok)
    else:
        # Vectorized
        s_mins, s_maxs = _bbox_min_max(surface_corners, env_id)
        ref_z = s_maxs[:, 2] if use_max_z else s_mins[:, 2]  # (N,)

        if mode == "centroid":
            _, centroid = world.get_bbox(obj, env_id=env_id)  # (N, 3)
            x_ok = (centroid[:, 0] >= s_mins[:, 0] - tolerance) & (centroid[:, 0] <= s_maxs[:, 0] + tolerance)
            y_ok = (centroid[:, 1] >= s_mins[:, 1] - tolerance) & (centroid[:, 1] <= s_maxs[:, 1] + tolerance)
            if obj_above:
                z_ok = centroid[:, 2] >= ref_z + z_margin
            else:
                z_ok = centroid[:, 2] <= ref_z - z_margin
            return x_ok & y_ok & z_ok
        else:  # bbox mode
            obj_corners, centroid = world.get_bbox(obj, env_id=env_id)  # (N,8,3), (N,3)
            x_ok = (centroid[:, 0] >= s_mins[:, 0] - tolerance) & (centroid[:, 0] <= s_maxs[:, 0] + tolerance)
            y_ok = (centroid[:, 1] >= s_mins[:, 1] - tolerance) & (centroid[:, 1] <= s_maxs[:, 1] + tolerance)
            if obj_above:
                z_ok = (obj_corners[:, :, 2] >= ref_z.unsqueeze(1) + z_margin).all(dim=1)
            else:
                z_ok = (obj_corners[:, :, 2] <= ref_z.unsqueeze(1) - z_margin).all(dim=1)
            return x_ok & y_ok & z_ok


def above_top(world, above_obj: str, surface: str, tolerance: float = 0.01, z_margin: float = 0.0, mode: str = "bbox", env_id: int | None = None):
    """Check if above_obj is above the top surface."""
    result = _vertical_check(world, above_obj, surface, tolerance, z_margin, mode, use_max_z=True, obj_above=True, env_id=env_id)
    if DEBUG and env_id is not None:
        print(f"above_top: '{above_obj}' above top of '{surface}' (tol={tolerance}, z_margin={z_margin}, mode={mode}) -> {result}")
    return result


def above_bottom(world, above_obj: str, surface: str, tolerance: float = 0.01, z_margin: float = 0.0, mode: str = "bbox", env_id: int | None = None):
    """Check if above_obj is above the bottom surface."""
    result = _vertical_check(world, above_obj, surface, tolerance, z_margin, mode, use_max_z=False, obj_above=True, env_id=env_id)
    if DEBUG and env_id is not None:
        print(f"above_bottom: '{above_obj}' above bottom of '{surface}' (tol={tolerance}, z_margin={z_margin}, mode={mode}) -> {result}")
    return result


def below_top(world, below_obj: str, surface: str, tolerance: float = 0.01, z_margin: float = 0.0, mode: str = "bbox", env_id: int | None = None):
    """Check if below_obj is below the top surface."""
    result = _vertical_check(world, below_obj, surface, tolerance, z_margin, mode, use_max_z=True, obj_above=False, env_id=env_id)
    if DEBUG and env_id is not None:
        print(f"below_top: '{below_obj}' below top of '{surface}' (tol={tolerance}, z_margin={z_margin}, mode={mode}) -> {result}")
    return result


def below_bottom(world, below_obj: str, surface: str, tolerance: float = 0.01, z_margin: float = 0.0, mode: str = "bbox", env_id: int | None = None):
    """Check if below_obj is below the bottom surface."""
    result = _vertical_check(world, below_obj, surface, tolerance, z_margin, mode, use_max_z=False, obj_above=False, env_id=env_id)
    if DEBUG and env_id is not None:
        print(f"below_bottom: '{below_obj}' below bottom of '{surface}' (tol={tolerance}, z_margin={z_margin}, mode={mode}) -> {result}")
    return result


# Alignment spatial checks
def center_of(world, object: str, reference_object: str, tolerance: float = 0.01, env_id: int | None = None):
    """Check if object's centroid is aligned with reference_object's centroid (XY only)."""
    centroid1 = world.get_centroid(object, env_id=env_id)
    centroid2 = world.get_centroid(reference_object, env_id=env_id)

    if env_id is not None:
        result = bool(np.allclose(centroid1[:2], centroid2[:2], atol=tolerance))
        if DEBUG:
            print(f"center_of: '{object}' XY-aligned with '{reference_object}' (tol={tolerance}) -> {result}")
        return result
    else:
        # centroid1, centroid2 are Tensor(N, 3)
        diff = torch.abs(centroid1[:, :2] - centroid2[:, :2])  # (N, 2)
        return (diff <= tolerance).all(dim=1)  # (N,)


def next_to(world, object: str, reference_object: str, dist: float = 0.05, env_id: int | None = None):
    """Check if object is within dist of reference_object horizontally with z-overlap."""
    corners1, _ = world.get_bbox(object, env_id=env_id)
    corners2, _ = world.get_bbox(reference_object, env_id=env_id)

    if env_id is not None:
        bbox1_min = np.min(corners1, axis=0)
        bbox1_max = np.max(corners1, axis=0)
        bbox2_min = np.min(corners2, axis=0)
        bbox2_max = np.max(corners2, axis=0)
        z_overlap = (bbox1_min[2] <= bbox2_max[2]) and (bbox1_max[2] >= bbox2_min[2])
        dx = max(bbox2_min[0] - bbox1_max[0], bbox1_min[0] - bbox2_max[0], 0)
        dy = max(bbox2_min[1] - bbox1_max[1], bbox1_min[1] - bbox2_max[1], 0)
        horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
        result = bool(horizontal_dist <= dist and z_overlap)
        if DEBUG:
            print(f"next_to: '{object}' within {dist}m of '{reference_object}' -> {result}")
        return result
    else:
        # corners1, corners2: (N, 8, 3)
        b1_min = corners1.min(dim=1).values  # (N, 3)
        b1_max = corners1.max(dim=1).values
        b2_min = corners2.min(dim=1).values
        b2_max = corners2.max(dim=1).values
        z_overlap = (b1_min[:, 2] <= b2_max[:, 2]) & (b1_max[:, 2] >= b2_min[:, 2])
        dx = torch.clamp(torch.max(b2_min[:, 0] - b1_max[:, 0], b1_min[:, 0] - b2_max[:, 0]), min=0)
        dy = torch.clamp(torch.max(b2_min[:, 1] - b1_max[:, 1], b1_min[:, 1] - b2_max[:, 1]), min=0)
        horizontal_dist = torch.sqrt(dx ** 2 + dy ** 2)
        return (horizontal_dist <= dist) & z_overlap


def level(world, object: str, reference_object: str, tolerance: float = 0.1, env_id: int | None = None):
    """Check if object is at the same level as reference_object, based on the centroid z-coordinate."""
    centroid_obj = world.get_centroid(object, env_id=env_id)
    centroid_ref = world.get_centroid(reference_object, env_id=env_id)

    if env_id is not None:
        result = bool(np.allclose(centroid_obj[:2], centroid_ref[:2], atol=tolerance))
        if DEBUG:
            print(f"level: '{object}' at same z-level as '{reference_object}' (tol={tolerance}) -> {result}")
        return result
    else:
        diff = torch.abs(centroid_obj[:, :2] - centroid_ref[:, :2])
        return (diff <= tolerance).all(dim=1)


def between(world, object: str, reference_obj1: str, reference_obj2: str,
            check_alignment: bool = True, alignment_tolerance: float = 0.1, env_id: int | None = None):
    """Check if object is positioned between two reference objects."""
    centroid_obj = world.get_centroid(object, env_id=env_id)
    centroid_ref1 = world.get_centroid(reference_obj1, env_id=env_id)
    centroid_ref2 = world.get_centroid(reference_obj2, env_id=env_id)

    if env_id is not None:
        # Existing scalar logic
        separations = np.abs(centroid_ref1 - centroid_ref2)
        primary_axis = np.argmax(separations)
        pos_obj = centroid_obj[primary_axis]
        pos_ref1 = centroid_ref1[primary_axis]
        pos_ref2 = centroid_ref2[primary_axis]
        min_pos = min(pos_ref1, pos_ref2)
        max_pos = max(pos_ref1, pos_ref2)
        between_on_axis = min_pos <= pos_obj <= max_pos

        if not check_alignment:
            result = bool(between_on_axis)
        else:
            other_axes = [i for i in range(3) if i != primary_axis]
            result = bool(between_on_axis)
            for axis in other_axes:
                t = (pos_obj - pos_ref1) / (pos_ref2 - pos_ref1 + 1e-8)
                expected_pos = centroid_ref1[axis] + t * (centroid_ref2[axis] - centroid_ref1[axis])
                actual_pos = centroid_obj[axis]
                if abs(actual_pos - expected_pos) > alignment_tolerance:
                    result = False
                    break
        if DEBUG:
            print(f"between: '{object}' between '{reference_obj1}' and '{reference_obj2}' -> {result}")
        return result
    else:
        # Vectorized — centroid_obj, ref1, ref2 are (N, 3) tensors
        separations = torch.abs(centroid_ref1 - centroid_ref2)  # (N, 3)
        primary_axis = separations.argmax(dim=1)  # (N,)
        # Gather positions along primary axis for each env
        pos_obj = torch.gather(centroid_obj, 1, primary_axis.unsqueeze(1)).squeeze(1)
        pos_ref1 = torch.gather(centroid_ref1, 1, primary_axis.unsqueeze(1)).squeeze(1)
        pos_ref2 = torch.gather(centroid_ref2, 1, primary_axis.unsqueeze(1)).squeeze(1)
        min_pos = torch.min(pos_ref1, pos_ref2)
        max_pos = torch.max(pos_ref1, pos_ref2)
        between_on_axis = (pos_obj >= min_pos) & (pos_obj <= max_pos)

        if not check_alignment:
            return between_on_axis
        # Alignment check — simplified for vectorized: check all axes deviation
        t = (pos_obj - pos_ref1) / (pos_ref2 - pos_ref1 + 1e-8)  # (N,)
        expected = centroid_ref1 + t.unsqueeze(1) * (centroid_ref2 - centroid_ref1)  # (N, 3)
        deviation = torch.abs(centroid_obj - expected)  # (N, 3)
        aligned = (deviation <= alignment_tolerance).all(dim=1)  # (N,)
        return between_on_axis & aligned


def in_line(world, objects: list[str], axis: str | None = None,
            tolerance: float = 0.05, min_spacing: float = 0.02, env_id: int | None = None):
    """Check if objects are arranged in a line."""
    if len(objects) < 2:
        if env_id is not None:
            return True
        return torch.ones(world.env.num_envs, dtype=torch.bool, device=world.env.device)

    if env_id is not None:
        centroids = np.array([world.get_centroid(obj, env_id=env_id) for obj in objects])
        if axis is None:
            variances = np.var(centroids, axis=0)
            primary_axis_idx = np.argmax(variances)
        else:
            axis_map = {"x": 0, "y": 1, "z": 2}
            primary_axis_idx = axis_map[axis.lower()]

        result = True
        for perp_axis in [i for i in range(3) if i != primary_axis_idx]:
            if np.std(centroids[:, perp_axis]) > tolerance:
                result = False
                break

        if result:
            primary_positions = centroids[:, primary_axis_idx]
            sorted_positions = np.sort(primary_positions)
            spacings = np.diff(sorted_positions)
            if len(spacings) > 0 and np.min(spacings) < min_spacing:
                result = False

        if DEBUG:
            print(f"in_line: {objects} in a line (axis={axis}, tol={tolerance}) -> {result}")
        return result
    else:
        # Vectorized: each centroid is (N, 3)
        centroids = torch.stack([world.get_centroid(obj, env_id=None) for obj in objects], dim=1)  # (N, num_objs, 3)
        if axis is None:
            variances = centroids.var(dim=1)  # (N, 3)
            primary_axis_idx = variances.argmax(dim=1)  # (N,)
        else:
            axis_map = {"x": 0, "y": 1, "z": 2}
            primary_axis_idx = torch.full((centroids.shape[0],), axis_map[axis.lower()],
                                          dtype=torch.long, device=centroids.device)

        # For simplicity with variable primary axes per env, loop over envs
        # (in_line is rarely called in hot path)
        results = torch.ones(centroids.shape[0], dtype=torch.bool, device=centroids.device)
        for env_idx in range(centroids.shape[0]):
            pa = primary_axis_idx[env_idx].item()
            c = centroids[env_idx]  # (num_objs, 3)
            for perp in [i for i in range(3) if i != pa]:
                if c[:, perp].std().item() > tolerance:
                    results[env_idx] = False
                    break
            if results[env_idx]:
                sorted_primary = c[:, pa].sort().values
                spacings = sorted_primary.diff()
                if len(spacings) > 0 and spacings.min().item() < min_spacing:
                    results[env_idx] = False
        return results


def stationary(world, object: str, linear_threshold: float = 0.01,
               angular_threshold: float = 0.1, check_angular: bool = True, env_id: int | None = None):
    """Check if object has stopped moving (velocity near zero)."""
    velocity = world.get_velocity(object, env_id=env_id)

    if env_id is not None:
        linear_velocity = velocity[:3]
        linear_speed = np.linalg.norm(linear_velocity.cpu().numpy())
        result = linear_speed < linear_threshold
        if result and check_angular:
            angular_velocity = velocity[3:]
            angular_speed = np.linalg.norm(angular_velocity.cpu().numpy())
            result = angular_speed < angular_threshold
        if DEBUG:
            print(f"stationary: '{object}' stopped (lin_thr={linear_threshold}) -> {result}")
        return result
    else:
        # velocity: (N, 6)
        linear_speed = torch.norm(velocity[:, :3], dim=1)  # (N,)
        result = linear_speed < linear_threshold
        if check_angular:
            angular_speed = torch.norm(velocity[:, 3:], dim=1)
            result = result & (angular_speed < angular_threshold)
        return result


def upright(world, object: str, tolerance: float = 0.1, up_axis: str = "z", env_id: int | None = None):
    """Check if object is standing upright (local up-axis aligned with world up)."""
    pos, quat = world.get_pose(object, env_id=env_id)
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[up_axis.lower()]

    if env_id is not None:
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        R = torch.tensor([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        up_vector_world = R[:, axis_idx]
        world_up = torch.tensor([0.0, 0.0, 1.0])
        alignment = torch.dot(up_vector_world, world_up).item()
        threshold = np.cos(tolerance)
        result = bool(alignment >= threshold)
        if DEBUG:
            print(f"upright: '{object}' upright (up_axis={up_axis}, tol={tolerance}) -> {result}")
        return result
    else:
        # quat: (N, 4) wxyz format
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # Build rotation matrices column for axis_idx
        if axis_idx == 0:
            up_z = 2*(x*z - w*y)  # R[2, 0]
        elif axis_idx == 1:
            up_z = 2*(y*z + w*x)  # R[2, 1]
        else:
            up_z = 1 - 2*(x*x + y*y)  # R[2, 2]
        threshold = np.cos(tolerance)
        return up_z >= threshold


def check_stacked(
    world,
    objects: list[str],
    order: str | None = None,
    tolerance: float = 0.01,
    env_id: int | None = None,
):
    """Check if objects are stacked in the given order."""
    if order is not None and order not in ["bottom_to_top", "top_to_bottom"]:
        raise ValueError(f"Invalid order: {order}")

    if len(objects) < 2:
        if env_id is not None:
            return True
        return torch.ones(world.env.num_envs, dtype=torch.bool, device=world.env.device)

    if env_id is not None:
        # Existing scalar logic
        if order in ["bottom_to_top", "top_to_bottom"]:
            result = True
            for i in range(len(objects) - 1):
                if order == "bottom_to_top":
                    contact = world.in_contact(objects[i+1], objects[i], env_id=env_id)
                    on_top = above_top(world, objects[i+1], objects[i], tolerance, env_id=env_id)
                    on_bottom = above_bottom(world, objects[i+1], objects[i], tolerance=0, env_id=env_id)
                else:
                    contact = world.in_contact(objects[i], objects[i+1], env_id=env_id)
                    on_top = above_top(world, objects[i], objects[i+1], tolerance, env_id=env_id)
                    on_bottom = above_bottom(world, objects[i], objects[i+1], tolerance=0, env_id=env_id)
                if not (contact and (on_top or on_bottom)):
                    result = False
                    break
        else:
            positions = []
            for obj in objects:
                p, _ = world.get_pose(obj, env_id=env_id)
                positions.append((obj, p[2]))
            sorted_objects = [obj for (obj, z_val) in sorted(positions, key=lambda item: -item[1])]
            result = True
            for j in range(len(sorted_objects) - 1):
                contact = world.in_contact(sorted_objects[j], sorted_objects[j+1], env_id=env_id)
                on_top = above_top(world, sorted_objects[j], sorted_objects[j+1], tolerance, env_id=env_id)
                on_bottom = above_bottom(world, sorted_objects[j], sorted_objects[j+1], tolerance=0, env_id=env_id)
                if not (contact and (on_top or on_bottom)):
                    result = False
                    break

        if DEBUG:
            print(f"check_stacked: {objects} stacked (order={order}, tol={tolerance}) -> {result}")
        return result
    else:
        # Vectorized — loop over pairs, combine per-env results
        # For order-agnostic, we'd need per-env sorting which is complex.
        # Use per-env loop for this rare function.
        num_envs = world.env.num_envs
        results = torch.ones(num_envs, dtype=torch.bool, device=world.env.device)
        for eid in range(num_envs):
            results[eid] = check_stacked(world, objects, order, tolerance, env_id=eid)
        return results


# Contact spatial checks
def in_contact(world, object1: str | list[str], object2: str | list[str], force_threshold: float = 0.1, env_id: int | None = None):
    """
    Checks if multiple objects are in contact with each other.
    Returns True (or Tensor) only if ALL pairs are in contact.
    """
    object1 = [object1] if isinstance(object1, str) else list(object1)
    object2 = [object2] if isinstance(object2, str) else list(object2)

    if env_id is not None:
        results = [
            world.in_contact(o1, o2, force_threshold, env_id=env_id)
            for o1 in object1
            for o2 in object2
        ]
        result = all(results)
        if DEBUG:
            print(f"in_contact: all pairs of {object1} and {object2} in contact -> {result}")
        return result
    else:
        # Vectorized: each world.in_contact returns (N,) tensor
        results = [
            world.in_contact(o1, o2, force_threshold, env_id=None)
            for o1 in object1
            for o2 in object2
        ]
        stacked = torch.stack(results, dim=0)  # (num_pairs, N)
        return stacked.all(dim=0)  # (N,)
