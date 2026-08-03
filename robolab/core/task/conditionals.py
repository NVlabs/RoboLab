# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from robolab.core.task.predicate_logic import _and, _not, _or
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

def _gripper_in_contact(world, object: str, gripper_name: str | list[str], env_id: int | None = None):
    """in_contact against one or more grippers; True if ANY of them touches `object`.

    `in_contact` itself treats a list as "all pairs must hold", which is the wrong
    logical for multi-end-effector robots (e.g. a dual-arm gripper pair) where either
    gripper grabbing the object should count.
    """
    names = [gripper_name] if isinstance(gripper_name, str) else list(gripper_name)
    result = in_contact(world, object, names[0], env_id=env_id)
    for name in names[1:]:
        result = _or(result, in_contact(world, object, name, env_id=env_id))
    return result

@atomic
def object_grabbed(
    env,
    object: str,
    gripper_name: str | list[str] = "gripper",
    env_id: int | None = None,
):
    """Check if an object is currently being grabbed by the gripper (in contact with gripper).

    `gripper_name` may be a list (e.g. for dual-arm robots) — the object counts as
    grabbed if ANY of the listed grippers is touching it.
    """
    world = get_world(env)
    result = _gripper_in_contact(world, object, gripper_name, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_grabbed: '{object}' in contact with '{gripper_name}' -> {result}")
    return result

@atomic
def object_dropped(
    env,
    object: str,
    gripper_name: str | list[str] = "gripper",
    env_id: int | None = None,
):
    """Check if an object has been dropped (not in contact with any of gripper_name)."""
    world = get_world(env)
    result = _not(_gripper_in_contact(world, object, gripper_name, env_id=env_id))
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
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are in an open-top container.

    Geometric check: the object's centroid is transformed into the container's local
    frame and bounds-checked against the container's local AABB (with one
    container-height of open-top slack along the container's local +z). Because the
    check is performed in the container's coordinates, the predicate is invariant to
    container orientation — a flipped or tipped container correctly fails.
    """
    def condition(world, obj, env_id=None):
        result = in_opentop_container(
            world, obj, container,
            tolerance=tolerance,
            env_id=env_id,
        )
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    tolerance: float = 0.01,
    require_contact_with: Union[str, list[str]] = None,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are stably supported on the top surface of reference_object.

    The check is the AND of:
      - is_supported_on_surface: contact-force from surface on obj is non-trivial,
        upward, and within a 45° cone of vertical.
      - centroid_in_footprint: obj's centroid xy lies within the surface's AABB
        (with ``tolerance`` slack). z is intentionally not bounded — concave
        surfaces (plates with wells, tilted/overhanging objects) make any
        all-corners-above-top rule too brittle.
    """
    def condition(world, obj, env_id=None):
        result = world.is_supported_on_surface(obj, reference_object, env_id=env_id)
        result = _and(result, centroid_in_footprint(
            world, obj, reference_object, tolerance=tolerance, env_id=env_id
        ))
        if require_contact_with is not None:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are outside the container.

    Symmetric with ``object_in_container``: the object is "outside" iff fewer
    than half of its hull vertices fall in the container's open-top hull
    (``frac_inside < 0.5``). Equivalent to ``not in_opentop_container``.
    """
    def condition(world, obj, env_id=None):
        result = _not(in_opentop_container(world, obj, container, tolerance, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_outside_of: {object} outside '{container}' (logical={logical}) -> {result}")
    return result

@atomic
def object_moved_to_container(
    env,
    object: str | list[str],
    target_container: str,
    source_container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects have been moved out of source_container and into target_container.

    Combines ``object_outside_of(source_container)`` and ``object_in_container(target_container)``
    into a single check, for pick-and-place tasks where the destination alone (e.g. an object
    that never left the source container, or was placed there directly) is not sufficient proof
    of a pick-and-place transfer.
    """
    def condition(world, obj, env_id=None):
        result = _and(
            _not(in_opentop_container(world, obj, source_container, tolerance, env_id=env_id)),
            in_opentop_container(world, obj, target_container, tolerance, env_id=env_id),
        )
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, target_container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_moved_to_container: {object} moved '{source_container}' -> '{target_container}' "
              f"(logical={logical}) -> {result}")
    return result

@atomic
def object_upright(
    env,
    object: str | list[str],
    tolerance: float = 0.1,
    up_axis: str = "z",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are standing upright (oriented correctly)."""
    def condition(world, obj, env_id=None):
        result = upright(world, obj, tolerance, up_axis, env_id=env_id)
        if require_contact_with is True:
            raise ValueError(
                "object_upright(require_contact_with=True) is invalid: object_upright "
                "has no reference_object. Pass a body name (str) or list of body names instead."
            )
        if require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
                at_pos = at_pos and not _gripper_in_contact(world, obj, gripper_name, env_id=env_id)
            return at_pos
        else:
            # pos: (N, 3)
            diff = torch.abs(pos - pos_target.unsqueeze(0))
            at_pos = (diff <= tolerance).all(dim=1)  # (N,)
            if require_gripper_detached:
                at_pos = at_pos & _not(_gripper_in_contact(world, obj, gripper_name, env_id=None))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] = "gripper",
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
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are outside of a container AND stably supported on a surface.

    Symmetric with ``object_in_container``: ``not in_opentop_container``
    (frac_inside < 0.5) for the container check; surface support unchanged.
    """
    def condition(world, obj, env_id=None):
        result = _and(
            _not(in_opentop_container(world, obj, container, tolerance, env_id=env_id)),
            world.is_supported_on_surface(obj, surface, env_id=env_id)
        )
        if require_gripper_detached:
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=env_id)))
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
    gripper_name: str | list[str] | None = None,
    table_name: str = "table",
    env_id: int | None = None,
):
    """Check if the gripper is in contact with the table."""
    world = get_world(env)
    if gripper_name is None:
        contact_gripper = getattr(getattr(env, "cfg", None), "contact_gripper", None)
        if isinstance(contact_gripper, dict) and contact_gripper:
            gripper_name = list(contact_gripper.keys())
        else:
            gripper_name = "gripper"
    result = _gripper_in_contact(world, table_name, gripper_name, env_id=env_id)
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


@atomic
def objects_placed_in_container_in_order(
    env,
    objects: list[str],
    container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = True,
    require_gripper_detached: bool = True,
    gripper_name: str | list[str] = "gripper",
    strict: bool = False,
    key: str | None = None,
    env_id: int | None = None,
):
    """Stateful: checks that ``objects`` ended up in ``container`` in the listed order.

    Latches each object's first-observed placement step per env (using
    ``env.episode_length_buf``). Returns True for an env only when:
      1. Every object currently satisfies the same per-object placement check
         used by ``object_in_container`` (containment + optional contact +
         optional gripper-detached), AND
      2. The latched first-placement steps are monotonic in the given order
         (``<=`` by default; ``<`` if ``strict=True``).

    Latches survive object ejection (first-ever placement is sticky), so a policy
    that places the wrong object first cannot recover by knocking it out and
    re-placing later. Per-env latches are cleared on episode reset by
    ``RobolabEnv._reset_idx`` via ``WorldState.reset_predicate_state``.
    """
    world = get_world(env)
    objects = list(objects)
    state_key = key or f"placed_in_order::{container}::{','.join(objects)}"

    def _make_state():
        N, dev = env.num_envs, env.device
        first = {obj: torch.full((N,), -1, dtype=torch.long, device=dev) for obj in objects}

        def reset(env_ids: torch.Tensor) -> None:
            for t in first.values():
                t[env_ids] = -1

        return {"first_placed_step": first, "__reset__": reset}

    state = world.get_or_init_predicate_state(state_key, _make_state)
    first_placed = state["first_placed_step"]
    ep_step = env.episode_length_buf  # (N,) long

    # Build the same per-object placement predicate as object_in_container.
    def _currently_placed(obj: str, eid: int | None):
        result = in_opentop_container(world, obj, container, tolerance=tolerance, env_id=eid)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=eid))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=eid))
        if require_gripper_detached:
            result = _and(result, _not(_gripper_in_contact(world, obj, gripper_name, env_id=eid)))
        return result

    if env_id is None:
        # Batched path used by IsaacLab TerminationManager.
        currently = {obj: _currently_placed(obj, None) for obj in objects}

        # Latch first-ever placement per env. torch.where keeps existing latches.
        for obj in objects:
            newly = (first_placed[obj] < 0) & currently[obj]
            if newly.any():
                first_placed[obj] = torch.where(newly, ep_step.to(first_placed[obj].dtype), first_placed[obj])

        all_in = torch.stack([currently[obj] for obj in objects], dim=0).all(dim=0)
        ordered = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for a, b in zip(objects[:-1], objects[1:]):
            both_latched = (first_placed[a] >= 0) & (first_placed[b] >= 0)
            if strict:
                cmp = first_placed[a] < first_placed[b]
            else:
                cmp = first_placed[a] <= first_placed[b]
            ordered &= both_latched & cmp

        result = all_in & ordered
    else:
        currently = {obj: _currently_placed(obj, env_id) for obj in objects}

        for obj in objects:
            if first_placed[obj][env_id].item() < 0 and bool(currently[obj]):
                first_placed[obj][env_id] = int(ep_step[env_id].item())

        all_in = all(bool(currently[obj]) for obj in objects)
        ordered = True
        for a, b in zip(objects[:-1], objects[1:]):
            fa = int(first_placed[a][env_id].item())
            fb = int(first_placed[b][env_id].item())
            if fa < 0 or fb < 0:
                ordered = False
                break
            if strict:
                if not (fa < fb):
                    ordered = False
                    break
            else:
                if not (fa <= fb):
                    ordered = False
                    break
        result = bool(all_in and ordered)

    if robolab.constants.DEBUG:
        print(f"objects_placed_in_container_in_order: {objects} -> '{container}' (strict={strict}) -> {result}")
    return result
