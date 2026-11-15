# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

from isaaclab.managers import TerminationTermCfg as TerminationCfg
from isaaclab.scene import InteractiveSceneCfg

from robolab.core.task.instruction import normalize_instruction_fields


@dataclass
class Task:
    """Base task configuration with scene, terminations, and instruction.

    instruction can be a plain string or a dict mapping type keys to strings, e.g.:
        instruction: str = "Pick up the banana"
        instruction = {"default": "Pick up the banana", "vague": "Move it", "specific": "..."}
    When a dict is used, omit the type annotation so Python treats it as a class
    variable (avoids mutable-default dataclass errors).
    """

    # These will be set by subclasses
    scene: InteractiveSceneCfg | Any = None
    instruction: str | dict[str, str] = ""
    terminations: TerminationCfg | Any = None
    rewards: Any = None  # Optional reward terms for RL training
    events: Any = None  # Optional event terms (e.g., randomization)
    subtasks: Any = None  # Subtasks for completion tracking
    contact_object_list: list[str] | None = None  # list of objects that can be in contact with the robot and will be tracked for contact sensors
    # Optional: objects that keep their relative poses when the initial pose is randomized
    # (--init-pose-noise), e.g. [["fork", "plate", "knife"]]. Adds to the groups detected
    # automatically from objects that touch at their default poses.
    object_groups: list[list[str]] | None = None
    # Optional: objects the initial-pose randomization never moves, e.g. ["shelf"]. Objects linked to
    # them (touching, or in the same object_groups entry) stay too. Physics is unaffected: they can
    # still be pushed during the episode.
    fixed_objects: list[str] | None = None

    episode_length_s: int = 60*10 # 10 minutes
    attributes: list[str] = None
    task_name: str = None  # Optional: explicit task name (defaults to class name). Useful for grouping task variants.


def resolve_instruction(instruction: str | dict[str, str], instruction_type: str = "default") -> str:
    """Resolve an instruction field to a plain string.

    If instruction is already a str, return it unchanged (instruction_type is ignored).
    If instruction is a dict, look up instruction_type, falling back to "default".
    """
    if isinstance(instruction, str):
        return instruction
    if instruction_type in instruction:
        return instruction[instruction_type]
    if "default" in instruction:
        return instruction["default"]
    raise ValueError(
        f"Instruction type '{instruction_type}' not found. Available: {list(instruction.keys())}"
    )


def verify_task_valid(task_class: type[Task]) -> tuple[bool, str]:
    """
    Verify if a task file is valid.
    """
    if task_class is None:
        error = f"Task class {task_class.__name__} is None."
        return False, error

    try:
        normalize_instruction_fields(task_class.instruction)
    except ValueError as exc:
        return False, f"Task class {task_class.__name__} has invalid instruction fields: {exc}"

    if task_class.terminations is None:
        error = f"Task class {task_class.__name__} has no terminations."
        return False, error

    if task_class.contact_object_list is None:
        error = f"Task class {task_class.__name__} has no contact object list: {task_class.contact_object_list}"
        return False, error

    if task_class.episode_length_s <= 0:
        error = f"Task class {task_class.__name__} episode length is not set or is less than 0: {task_class.episode_length_s}"
        return False, error

    # Check if contact_object_list matches scene objects
    contact_list_valid, error = verify_contact_objects_in_scene(task_class)
    if not contact_list_valid:
        error = f"Contact list is not valid: {error}"
        return False, error

    init_pose_fields_valid, error = verify_init_pose_fields(task_class)
    if not init_pose_fields_valid:
        return False, error

    # Check terminations
    terminations = task_class.terminations()
    termination_success_func = getattr(terminations, 'success', None)
    if termination_success_func is not None:
        func = termination_success_func.func
        params = termination_success_func.params
        from robolab.core.utils.function_loader import verify_callable_args_supplied
        valid, error = verify_callable_args_supplied(func, params)
        if not valid:
            error = task_class.__name__ + " is not a valid task class. " + error
            return False, error

        for obj_args in ["object", "container", "objects", "reference_object", "surface"]:
            if obj_args in params:
                objects = params.get(obj_args, [])
                if objects is None:
                    # Count conditionals accept object=None (= all contact objects).
                    continue
                if isinstance(objects, str):
                    objects = [objects]
                elif isinstance(objects, list):
                    pass
                else:
                    error = f"Object argument used in termination '{obj_args}' is not a string or list: {objects}"
                    return False, error
                for obj in objects:
                    if obj not in task_class.contact_object_list:
                        error = f"Object used in termination '{obj}' under argument '{obj_args}' is not in contact object list: {task_class.contact_object_list}"
                        return False, error

    return True, None


def verify_init_pose_fields(task_class: type[Task]) -> tuple[bool, str | None]:
    """Verify ``object_groups`` and ``fixed_objects``: the right shape, and every name is a scene object."""
    name = task_class.__name__
    groups = getattr(task_class, "object_groups", None)
    fixed = getattr(task_class, "fixed_objects", None)
    if groups is not None and (not isinstance(groups, list) or not all(
            isinstance(group, list) and group and all(isinstance(n, str) for n in group) for group in groups)):
        return False, f"Task class {name} object_groups must be a list of non-empty lists of names: {groups}"
    if fixed is not None and (not isinstance(fixed, list) or not all(isinstance(n, str) for n in fixed)):
        return False, f"Task class {name} fixed_objects must be a list of names: {fixed}"
    names = sorted({n for group in groups or [] for n in group} | set(fixed or []))
    usd_path = getattr(getattr(getattr(task_class.scene, "scene", None), "spawn", None), "usd_path", None)
    if not names or usd_path is None:
        return True, None
    from robolab.core.scenes.utils import verify_objects_in_scene
    valid, error = verify_objects_in_scene(names, usd_path)
    if not valid:
        return False, f"Task class {name} object_groups/fixed_objects: {error}"
    return True, None


def verify_contact_objects_in_scene(task_class: type[Task]) -> tuple[bool, str]:
    """
    Verify that all objects in the task's contact_object_list are present in the scene.

    Args:
        task_class: The task class to verify

    Returns:
        tuple[bool, list[str]]: (is_valid, list_of_error_messages)
    """


    # Try to extract the scene USD path
    try:
        scene = getattr(task_class, 'scene', None)
        if scene is None:
            error = "Task has no scene attribute"
            return False, error

        # Get the scene USD path by traversing scene.scene.spawn.usd_path
        usd_path = getattr(getattr(getattr(scene, 'scene', None), 'spawn', None), 'usd_path', None)

        # If no USD path found, check if scene has individual object attributes
        if usd_path is None:
            # Get all attributes from the scene class (not instance methods)
            # Use __annotations__ if available, otherwise check class __dict__
            if hasattr(scene, '__annotations__'):
                scene_attributes = list(scene.__annotations__.keys())
            else:
                scene_attributes = [attr for attr in vars(scene) if not attr.startswith('_') and attr != 'scene']

            # Check if all contact objects are present as scene attributes
            missing_objects = [obj for obj in task_class.contact_object_list if obj not in scene_attributes]

            if missing_objects:
                error = f"Contact objects not found in scene attributes: {missing_objects}. Scene has: {scene_attributes}"
                return False, error

            # All objects found in scene attributes
            return True, None

    except Exception as e:
        error = f"Error extracting scene path: {e}"
        return False, error

    from robolab.core.scenes.utils import verify_objects_in_scene
    # Verify that the contact objects are in the scene USD file
    contact_list_valid, error = verify_objects_in_scene(task_class.contact_object_list, usd_path)
    if not contact_list_valid:
        return False, error

    return True, None
