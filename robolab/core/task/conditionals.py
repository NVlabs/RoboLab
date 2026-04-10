# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
conditionals: "Does task condition Y hold for these objects?"

Task-level API with @atomic/@composite decorators.
Takes `env` as first parameter.
Handles logical modes (all/any/choose) and contact requirements.

All @atomic conditionals support env_id parameter:
  env_id=None (default) → returns Tensor(num_envs,) bool (used by IsaacLab TerminationManager)
  env_id=<int>          → returns bool (used by ConditionalsStateMachine)
"""

from functools import partial
from typing import Literal, Optional, Union

import torch

import robolab.constants
from robolab.core.task.decorators import atomic, composite
from robolab.core.task.predicate_logic import *
from robolab.core.task.predicate_logic import _and, _not
from robolab.core.task.subtask import Subtask
from robolab.core.world.world_state import get_world

#########################################################
# Composite conditions
#########################################################

@composite
def pick_and_place(
    object: str | list[str],
    container: str,
    logical: Literal["all", "any", "choose"] = "all",
    K: Optional[int] = None,
    score: float = 1.0
) -> Subtask:
    """
    A composite subtask that picks up object(s) and places them in a container.

    This function creates parallel subtask sequences for each specified object, where each
    object independently progresses through: grab → lift → move → drop → verify placement.
    The completion logic determines when the entire group is considered complete.

    Args:
        object: Single object name or list of object names to manipulate in parallel
        container: Target container name where objects should be placed
        logical: Completion mode determining when this subtask group succeeds:
            - "all": All objects must complete their subtasks (default)
            - "any": Success when any single object completes
            - "choose": Success when exactly k objects complete (requires k parameter)
        k: Number of objects that must complete when logical="choose"
        score: Overall score weight for this subtask group (default: 1.0)
    """
    if isinstance(object, str):
        objects = [object]
    else:
        objects = list(object)

    conditions = {}
    for obj in objects:
        conditions[obj] = [
            (partial(object_grabbed, object=obj), 0.0),
            (partial(object_in_container, object=obj, container=container, require_contact_with=False, require_gripper_detached=True), score),
        ]

    return Subtask(name="pick_and_place", conditions=conditions, logical=logical, score=score, K=K)


@composite
def pick_and_place_on_surface(
    object: str | list[str],
    surface: str,
    logical: Literal["all", "any", "choose"] = "all",
    K: Optional[int] = None,
    score: float = 1.0
) -> Subtask:
    """
    A composite subtask that picks up object(s) and places them on a surface.

    Similar to pick_and_place, but verifies stable support on a flat surface using
    contact force cone checking rather than containment checks.

    Args:
        object: Single object name or list of object names to manipulate in parallel
        surface: Target surface name where objects should be placed
        logical: Completion mode - "all", "any", or "choose"
        k: Number of objects that must complete when logical="choose"
        score: Overall score weight for this subtask group (default: 1.0)
    """
    if isinstance(object, str):
        objects = [object]
    else:
        objects = list(object)

    conditions = {}
    for obj in objects:
        conditions[obj] = [
            (partial(object_grabbed, object=obj), 0.0),
            (partial(object_on_top, object=obj, reference_object=surface, require_gripper_detached=True), score),
        ]

    return Subtask(name="pick_and_place_on_surface", conditions=conditions, logical=logical, score=score, K=K)


#########################################################
# Atomic conditions - Contact
#########################################################

@atomic
def object_in_contact(
    env,
    object1: str | list[str],
    object2: str | list[str],
    logical: str = "any",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks contact between objects according to logical."""
    if logical not in ["any", "all", "choose"]:
        raise ValueError(f"Invalid logical: {logical}")

    world = get_world(env)
    result = in_contact(world, object1, object2, force_threshold=0.1, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_in_contact: {object1} and {object2} in contact (logical={logical}) -> {result}")
    return result

@atomic
def object_grabbed(
    env,
    object: str,
    gripper_name: str = "gripper",
    env_id: int | None = None,
):
    """Check if an object is currently being grabbed by the gripper (in contact with gripper)."""
    world = get_world(env)
    result = in_contact(world, object, gripper_name, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_grabbed: '{object}' in contact with '{gripper_name}' -> {result}")
    return result

@atomic
def object_dropped(
    env,
    object: str,
    gripper_name: str = "gripper",
    env_id: int | None = None,
):
    """Check if an object has been dropped (not in contact with gripper)."""
    world = get_world(env)
    result = _not(in_contact(world, object, gripper_name, env_id=env_id))
    if robolab.constants.DEBUG:
        print(f"object_dropped: '{object}' not in contact with '{gripper_name}' -> {result}")
    return result

@atomic
def object_picked_up(
    env,
    object: str,
    surface: str,
    distance: float = 0.05,
    env_id: int | None = None,
):
    """Check if object is grabbed and lifted at least `distance` above the surface."""
    result = _and(
        object_grabbed(env, object, env_id=env_id),
        object_above(env, object=object, reference_object=surface, env_id=env_id, z_margin=distance)
    )
    if robolab.constants.DEBUG:
        print(f"object_picked_up: '{object}' grabbed and lifted {distance}m above '{surface}' -> {result}")
    return result

#########################################################
# Unified Spatial Conditions (New API)
#########################################################
#
# Parameters:
#   require_contact_with: Contact requirement
#       - False: no contact check (default)
#       - True: must be in contact with reference_object
#       - str/list[str]: must be in contact with specified object(s)
#   require_gripper_detached: If True, object must NOT be held by gripper
#

@atomic
def object_in_container(
    env,
    object: str | list[str],
    container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    require_stationary: bool = False,
    stationary_threshold: float = 0.05,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are in an open-top container."""
    def condition(world, obj, env_id=None):
        result = in_opentop_container(world, obj, container, tolerance, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        if require_stationary:
            result = _and(result, stationary(world, obj, linear_threshold=stationary_threshold, check_angular=False, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_in_container: {object} in '{container}' (tol={tolerance}, logical={logical}) -> {result}")
    return result

@atomic
def object_on_top(
    env,
    object: str | list[str],
    reference_object: str,
    require_contact_with: Union[str, list[str]] = None,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are stably supported on the top surface of reference_object."""
    def condition(world, obj, env_id=None):
        result = world.is_supported_on_surface(obj, reference_object, env_id=env_id)
        if require_contact_with is not None:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_on_top: {object} on top of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_on_bottom(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    z_margin: float = 0.0,
    mode: str = "bbox",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are positioned above the bottom surface of reference_object."""
    def condition(world, obj, env_id=None):
        result = above_bottom(world, obj, reference_object, tolerance, z_margin, mode, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_on_bottom: {object} above bottom of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_on_center(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are centered on reference_object (XY alignment)."""
    def condition(world, obj, env_id=None):
        result = center_of(world, obj, reference_object, tolerance, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_on_center: {object} centered on '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_left_of(
    env,
    object: str | list[str],
    reference_object: str,
    frame_of_reference: str = "robot",
    mirrored: bool = False,
    cone_deg: int = 45,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are to the left of reference_object."""
    def condition(world, obj, env_id=None):
        result = left_of(world, obj, reference_object, frame_of_reference, mirrored, cone_deg, env_id=env_id)
        if not require_contact_with and not require_gripper_detached:
            result = _and(result, level(world, obj, reference_object, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_left_of: {object} left of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_right_of(
    env,
    object: str | list[str],
    reference_object: str,
    frame_of_reference: str = "robot",
    mirrored: bool = False,
    cone_deg: int = 45,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are to the right of reference_object."""
    def condition(world, obj, env_id=None):
        result = right_of(world, obj, reference_object, frame_of_reference, mirrored, cone_deg, env_id=env_id)
        if not require_contact_with and not require_gripper_detached:
            result = _and(result, level(world, obj, reference_object, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_right_of: {object} right of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_in_front_of(
    env,
    object: str | list[str],
    reference_object: str,
    frame_of_reference: str = "robot",
    mirrored: bool = False,
    cone_deg: int = 45,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are in front of reference_object."""
    def condition(world, obj, env_id=None):
        result = in_front_of(world, obj, reference_object, frame_of_reference, mirrored, cone_deg, env_id=env_id)
        if not require_contact_with and not require_gripper_detached:
            result = _and(result, level(world, obj, reference_object, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_in_front_of: {object} in front of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_behind(
    env,
    object: str | list[str],
    reference_object: str,
    frame_of_reference: str = "robot",
    mirrored: bool = False,
    cone_deg: int = 45,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are behind reference_object."""
    def condition(world, obj, env_id=None):
        result = behind(world, obj, reference_object, frame_of_reference, mirrored, cone_deg, env_id=env_id)
        if not require_contact_with and not require_gripper_detached:
            result = _and(result, level(world, obj, reference_object, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_behind: {object} behind '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_next_to(
    env,
    object: str | list[str],
    reference_object: str,
    dist: float = 0.05,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are within a certain distance of reference_object."""
    def condition(world, obj, env_id=None):
        result = next_to(world, obj, reference_object, dist, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_next_to: {object} within {dist}m of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_below_top(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    z_margin: float = 0.0,
    mode: str = "bbox",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are below the top surface of reference_object."""
    def condition(world, obj, env_id=None):
        result = below_top(world, obj, reference_object, tolerance, z_margin, mode, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_below_top: {object} below top of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_below(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    z_margin: float = 0.0,
    mode: str = "bbox",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are below the bottom surface of reference_object."""
    def condition(world, obj, env_id=None):
        result = below_bottom(world, obj, reference_object, tolerance, z_margin, mode, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_below: {object} below bottom of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_enclosed(
    env,
    object: str | list[str],
    container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects' bounding boxes are fully enclosed inside the container."""
    def condition(world, obj, env_id=None):
        result = enclosed(world, obj, container, tolerance, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_enclosed: {object} enclosed in '{container}' (logical={logical}) -> {result}")
    return result

@atomic
def object_inside(
    env,
    object: str | list[str],
    container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects' centroids are inside the container's bounding box."""
    def condition(world, obj, env_id=None):
        result = inside(world, obj, container, tolerance, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_inside: {object} inside '{container}' (logical={logical}) -> {result}")
    return result

@atomic
def object_outside_of(
    env,
    object: str | list[str],
    container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are outside of a container."""
    def condition(world, obj, env_id=None):
        result = _not(in_opentop_container(world, obj, container, tolerance, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_outside_of: {object} outside '{container}' (logical={logical}) -> {result}")
    return result

@atomic
def object_upright(
    env,
    object: str | list[str],
    tolerance: float = 0.1,
    up_axis: str = "z",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are standing upright (oriented correctly)."""
    def condition(world, obj, env_id=None):
        result = upright(world, obj, tolerance, up_axis, env_id=env_id)
        if require_contact_with and require_contact_with is not True:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_upright: {object} upright (up_axis={up_axis}, logical={logical}) -> {result}")
    return result

@atomic
def object_at(
    env,
    object: str | list[str],
    position: tuple[float, float, float],
    tolerance: float = 0.02,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Check if objects are at a specific 3D position within tolerance."""
    if logical not in ["any", "all", "choose"]:
        raise ValueError(f"Invalid logical: {logical}")

    world = get_world(env)
    object_list = [object] if isinstance(object, str) else list(object)
    pos_target = torch.tensor(position, dtype=torch.float32, device=world.env.device)

    def check_obj(obj, env_id=None):
        pos, _ = world.get_pose(obj, env_id=env_id)
        if env_id is not None:
            at_pos = torch.allclose(pos, pos_target, atol=tolerance)
            if require_gripper_detached:
                at_pos = at_pos and not in_contact(world, obj, gripper_name, env_id=env_id)
            return at_pos
        else:
            # pos: (N, 3)
            diff = torch.abs(pos - pos_target.unsqueeze(0))
            at_pos = (diff <= tolerance).all(dim=1)  # (N,)
            if require_gripper_detached:
                at_pos = at_pos & _not(in_contact(world, obj, gripper_name, env_id=None))
            return at_pos

    result = evaluate_spatial_condition(env, object, check_obj, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_at: {object} at {position} (tol={tolerance}, logical={logical}) -> {result}")
    return result

@atomic
def object_between(
    env,
    object: str | list[str],
    reference_obj1: str,
    reference_obj2: str,
    check_alignment: bool = True,
    alignment_tolerance: float = 0.1,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are positioned between two reference objects."""
    def condition(world, obj, env_id=None):
        result = between(world, obj, reference_obj1, reference_obj2, check_alignment, alignment_tolerance, env_id=env_id)
        if require_contact_with and require_contact_with is not True:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_between: {object} between '{reference_obj1}' and '{reference_obj2}' (logical={logical}) -> {result}")
    return result

@atomic
def objects_in_line(
    env,
    objects: list[str],
    axis: str | None = None,
    tolerance: float = 0.05,
    min_spacing: float = 0.02,
    env_id: int | None = None,
):
    """Checks if multiple objects are arranged in a line/row."""
    world = get_world(env)
    result = in_line(world, objects, axis, tolerance, min_spacing, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"objects_in_line: {objects} in a line (axis={axis}) -> {result}")
    return result

@atomic
def objects_stationary(
    env,
    object: str | list[str],
    linear_threshold: float = 0.01,
    angular_threshold: float = 0.1,
    check_angular: bool = True,
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects have stopped moving (velocity near zero)."""
    result = evaluate_spatial_condition(
        env, object,
        lambda world, obj, env_id=None: stationary(world, obj, linear_threshold, angular_threshold, check_angular, env_id=env_id),
        logical, K, env_id=env_id
    )
    if robolab.constants.DEBUG:
        print(f"objects_stationary: {object} stationary (logical={logical}) -> {result}")
    return result

@atomic
def object_center_of(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Check if the geometric centers of objects are aligned with reference_object (XY plane only)."""
    def condition(world, obj, env_id=None):
        result = center_of(world, obj, reference_object, tolerance, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_center_of: {object} centered on '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_above(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    z_margin: float = 0.0,
    mode: str = "bbox",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Check if objects are geometrically positioned above the top surface of reference_object."""
    def condition(world, obj, env_id=None):
        result = above_top(world, obj, reference_object, tolerance, z_margin, mode, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_above: {object} above '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_above_bottom(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    z_margin: float = 0.0,
    mode: str = "bbox",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Check if objects are positioned above the bottom surface of reference_object."""
    return object_on_bottom(
        env, object, reference_object, tolerance, z_margin, mode,
        require_contact_with, require_gripper_detached, gripper_name,
        logical, K, env_id
    )

#########################################################
# Special compound conditions
#########################################################

@atomic
def object_outside_of_and_on_surface(
    env,
    object: str | list[str],
    container: str,
    surface: str,
    tolerance: float = 0.01,
    require_gripper_detached: bool = False,
    gripper_name: str = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are outside of a container AND stably supported on a surface."""
    def condition(world, obj, env_id=None):
        result = _and(
            _not(in_opentop_container(world, obj, container, tolerance, env_id=env_id)),
            world.is_supported_on_surface(obj, surface, env_id=env_id)
        )
        if require_gripper_detached:
            result = _and(result, _not(in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_outside_of_and_on_surface: {object} outside '{container}' and on '{surface}' (logical={logical}) -> {result}")
    return result

@atomic
def object_groups_in_containers(
    env,
    groups: list[dict],
    env_id: int | None = None,
):
    """Checks multiple (object(s), container) groups; returns True only if all groups satisfy placement."""
    if groups is None or len(groups) == 0:
        if env_id is not None:
            return False
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    results = []
    for group in groups:
        objects = group.get("object", [])
        container = group.get("container")
        tolerance = group.get("tolerance", 0.01)
        logical = group.get("logical", "all")
        K = group.get("K", 1)
        require_contact_with = group.get("require_contact_with", False)
        require_gripper_detached = group.get("require_gripper_detached", True)

        r = object_in_container(
            env,
            object=objects,
            container=container,
            tolerance=tolerance,
            require_contact_with=require_contact_with,
            require_gripper_detached=require_gripper_detached,
            logical=logical,
            K=K,
            env_id=env_id,
        )
        results.append(r)

    if env_id is not None:
        return all(results)
    else:
        return torch.stack([r if isinstance(r, torch.Tensor) else torch.tensor(r) for r in results]).all(dim=0)


#########################################################
# Not conditions
#########################################################
@atomic
def wrong_object_grabbed(
    env,
    object: str | list[str],
    gripper_name: str = "gripper",
    ignore_objects: list[str] = ["table"],
    env_id: int | None = None,
):
    """Check if gripper is holding any object other than the specified target object(s)."""
    if isinstance(object, str):
        intended_set = {object}
    else:
        intended_set = set(object)
    ignore_set = set(ignore_objects)

    candidates = [obj for obj in env.cfg.contact_object_list if obj not in ignore_set]

    # This function returns a list and is inherently per-env
    if env_id is None:
        # Vectorized: check each env
        results = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for eid in range(env.num_envs):
            world = get_world(env)
            objects_in_contact = world.get_objects_in_contact_with(gripper_name, candidates, env_id=eid)
            for obj_name in objects_in_contact:
                if obj_name not in intended_set:
                    results[eid] = True
                    break
        return results
    else:
        world = get_world(env)
        objects_in_contact = world.get_objects_in_contact_with(gripper_name, candidates, env_id=env_id)
        for obj_name in objects_in_contact:
            if obj_name not in intended_set:
                if robolab.constants.DEBUG:
                    print(f"wrong_object_grabbed: Gripper holding '{obj_name}' instead of {object} -> True")
                return True
        if robolab.constants.DEBUG:
            print(f"wrong_object_grabbed: No wrong object grabbed -> False")
        return False


def get_wrong_object_grabbed(
    env,
    intended_objects: str | list[str],
    gripper_name: str = "gripper",
    ignore_objects: list[str] = ["table"],
    env_id: int | None = None,
) -> str | None:
    """Get the name of a wrong object that is currently grabbed.
    Note: inherently per-env, defaults to env_id=0 when None."""
    if env_id is None:
        env_id = 0

    if not gripper_slightly_closed(env, env_id=env_id):
        return None

    if isinstance(intended_objects, str):
        intended_set = {intended_objects}
    else:
        intended_set = set(intended_objects)
    ignore_set = set(ignore_objects)

    candidates = [obj for obj in env.cfg.contact_object_list if obj not in ignore_set]

    world = get_world(env)
    objects_in_contact = world.get_objects_in_contact_with(gripper_name, candidates, env_id=env_id)

    for obj in objects_in_contact:
        if obj not in intended_set:
            return obj
    return None


def gripper_hit_table(
    env,
    gripper_name: str = "gripper",
    table_name: str = "table",
    env_id: int | None = None,
):
    """Check if the gripper is in contact with the table."""
    world = get_world(env)
    result = in_contact(world, gripper_name, table_name, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"gripper_hit_table: '{gripper_name}' in contact with '{table_name}' -> {result}")
    return result


def gripper_fully_closed(
    env,
    robot_name: str = "robot",
    gripper_joint_name: str = "finger_joint",
    closed_threshold: float = 0.75,
    env_id: int | None = None,
):
    """Check if the gripper is fully closed (or nearly closed)."""
    import math as _math

    world = get_world(env)
    joint_positions = world.get_joint_positions(robot_name, env_id=env_id)
    joint_names = world.get_joint_names(robot_name)

    if gripper_joint_name not in joint_names:
        if env_id is not None:
            return False
        return torch.zeros(world.env.num_envs, dtype=torch.bool, device=world.env.device)

    joint_idx = joint_names.index(gripper_joint_name)
    max_closed_position = _math.pi / 4

    if env_id is not None:
        gripper_pos = joint_positions[joint_idx].item()
        normalized_pos = gripper_pos / max_closed_position
        return normalized_pos >= closed_threshold
    else:
        # joint_positions: (N, num_joints)
        gripper_pos = joint_positions[:, joint_idx]
        normalized_pos = gripper_pos / max_closed_position
        return normalized_pos >= closed_threshold


def gripper_slightly_closed(
    env,
    robot_name: str = "robot",
    gripper_joint_name: str = "finger_joint",
    closed_threshold: float = 0.30,
    env_id: int | None = None,
):
    """Check if the gripper is slightly closed (at least 30% closed by default)."""
    return gripper_fully_closed(env, robot_name, gripper_joint_name, closed_threshold, env_id=env_id)


#########################################################
# Ordering conditions
#########################################################

@atomic
def stacked(
    env,
    objects: list[str],
    order: str | None = None,
    tolerance: float = 0.01,
    env_id: int | None = None,
):
    """Checks if the objects are stacked in the given order."""
    if order == "None":
        order = None

    if robolab.constants.DEBUG:
        print(f"Checking stacked({objects}, order={order})")

    world = get_world(env)
    result = check_stacked(world, objects, order, tolerance, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"stacked: {objects} stacked (order={order}, tol={tolerance}) -> {result}")
    return result
