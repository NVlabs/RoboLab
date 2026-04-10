# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import TYPE_CHECKING, Any, Callable, cast

import robolab.constants
from robolab.core.task.predicate_logic import get_task_conditional_func
from robolab.core.utils.function_loader import func_as_str, get_callable_info

if TYPE_CHECKING:
    from robolab.core.task.subtask import Subtask

def process_subtasks_as_str(subtasks: list['Subtask'] | None) -> str:
    """
    Process a list of Subtask objects into a human-readable string representation.

    Args:
        subtasks: List of Subtask objects or None

    Returns:
        A string representation of all subtasks, formatted as:
        "subtask_name(objects=[...], logical=...) score=X"
        Multiple subtasks are separated by semicolons.
    """
    if not subtasks:
        return ""

    # Import here to avoid circular dependency
    from robolab.core.task.subtask import Subtask

    subtask_strs = []
    for subtask in subtasks:
        if not isinstance(subtask, Subtask):
            # Handle case where subtask might not be a Subtask object
            subtask_strs.append(str(subtask))
            continue

        # Extract object names from conditions dict
        # After __post_init__, conditions is always a dict[str, list[tuple[Callable, float]]]
        conditions_dict = cast(dict[str, list[tuple[Callable, float]]], subtask.conditions)
        group_names = list(conditions_dict.keys())

        # Format the subtask string
        if subtask.logical == "choose":
            subtask_str = f"{subtask.name}(groups={group_names}, logical={subtask.logical}, k={subtask.K}) score={subtask.score}"
        else:
            subtask_str = f"{subtask.name}(groups={group_names}, logical={subtask.logical}) score={subtask.score}"

        subtask_strs.append(subtask_str)

    return "; ".join(subtask_strs)

def count_stages_and_conditions(subtasks: list['Subtask'] | None) -> tuple[int, int]:
    """
    Count the number of sequential stages and atomic conditions.

    Args:
        subtasks: List of Subtask objects or None

    Returns:
        Tuple of (num_sequential_stages, num_atomic_conditions):
        - num_sequential_stages: Number of Subtask objects (sequential phases) in the list
        - num_atomic_conditions: Total count of atomic condition checks
          - For "all" logical: sum of all conditions across all objects
          - For "any" logical: count from one representative object (since only one needs to complete)
          - For "choose" logical: count from k objects (since k need to complete)
    """
    if not subtasks:
        return 0, 0

    # Import here to avoid circular dependency
    from robolab.core.task.subtask import Subtask

    num_sequential_stages = len(subtasks)
    num_atomic_conditions = 0

    for subtask in subtasks:
        if not isinstance(subtask, Subtask):
            # If it's not a Subtask object, skip counting
            continue

        # After __post_init__, conditions is always a dict[str, list[tuple[Callable, float]]]
        conditions_dict = cast(dict[str, list[tuple[Callable, float]]], subtask.conditions)

        # Count conditions based on logical mode
        if subtask.logical == "all":
            # All objects must complete, so count all conditions
            for object_subtask_list in conditions_dict.values():
                num_atomic_conditions += len(object_subtask_list)

        elif subtask.logical == "any":
            # Only one object needs to complete, so count conditions from first object
            if conditions_dict:
                first_object = list(conditions_dict.keys())[0]
                num_atomic_conditions += len(conditions_dict[first_object])

        elif subtask.logical == "choose":
            # k objects need to complete
            if conditions_dict and subtask.K is not None:
                # Count conditions from first object and multiply by k
                first_object = list(conditions_dict.keys())[0]
                conditions_per_object = len(conditions_dict[first_object])
                num_atomic_conditions += conditions_per_object * subtask.K

    return num_sequential_stages, num_atomic_conditions


def count_subtasks(subtasks: list['Subtask'] | None) -> int:
    """
    Count the total number of subtasks (manipulation actions) required to
    complete all sequential stages, accounting for logical mode.

    For each stage, the number of subtasks is determined by the logical mode
    and the number of object groups in its conditions dict:
        - "all":    every object group must complete -> num_object_groups
        - "any":    only one object group must complete -> 1
        - "choose": exactly K object groups must complete -> K

    The total is summed across all sequential stages.

    Args:
        subtasks: List of Subtask objects or None

    Returns:
        Total number of subtasks across all sequential stages.
    """
    if not subtasks:
        return 0

    from robolab.core.task.subtask import Subtask

    total = 0
    for subtask in subtasks:
        if not isinstance(subtask, Subtask):
            continue

        conditions_dict = cast(dict[str, list[tuple[Callable, float]]], subtask.conditions)
        num_groups = len(conditions_dict)

        if subtask.logical == "all":
            total += num_groups
        elif subtask.logical == "any":
            total += 1
        elif subtask.logical == "choose":
            total += subtask.K if subtask.K is not None else num_groups

    return total


from robolab.constants import DIFFICULTY_THRESHOLDS, SKILL_WEIGHTS


def compute_difficulty_score(
    num_subtasks: int,
    attributes: list[str],
) -> tuple[int, str]:
    """
    Compute a difficulty score and label from the number of subtasks and task attributes.

    The score combines manipulation volume (num_subtasks) with the hardest
    required skill (max skill weight from attributes):

        difficulty_score = num_subtasks + max(skill_weight for each attribute)

    The label is derived from thresholds defined in DIFFICULTY_THRESHOLDS:
        - simple:   score <= DIFFICULTY_THRESHOLDS[0]
        - moderate: score <= DIFFICULTY_THRESHOLDS[1]
        - complex:  score >  DIFFICULTY_THRESHOLDS[1]

    Args:
        num_subtasks: Number of subtask actions (from count_subtasks)
        attributes: List of task attribute strings (e.g., ['semantics', 'sorting', 'color']).
                    Difficulty-level attributes ('simple', 'moderate', 'complex') are ignored.

    Returns:
        Tuple of (difficulty_score, difficulty_label)
    """
    non_diff = [a for a in attributes if a not in ('simple', 'moderate', 'complex')]
    skill_weight = max((SKILL_WEIGHTS.get(a, 0) for a in non_diff), default=0)
    score = num_subtasks + skill_weight

    simple_max, moderate_max = DIFFICULTY_THRESHOLDS
    if score <= simple_max:
        label = 'simple'
    elif score <= moderate_max:
        label = 'moderate'
    else:
        label = 'complex'

    return score, label


def sanitize_subtask_conditions(conditions) -> dict[str, list[tuple[Callable, float]]]:
    """
    Sanitize the conditions structure.
    Valid conditions structures:
    Single condition (atomic subtask)
    1. conditions = func ---- single task

    These conditions (in set or list notation) is processed as a single group, and is subject to logical ('all', 'any', 'choose')

    2. conditions = [func1, func2] ---- list notation, assumed to have the same score.
    3. conditions = {func1, func2} ---- set notation, assumed to have the same score.
    4. conditions = [(func1, score), (func2, score)] ---- list of tuples, the scores are specified
    5. conditions = {(func1, score), (func2, score)} ---- set of tuples, the scores are specified

    If the conditions are provided in the form of a dictionary, then:
    - within each group, the conditions are checked sequentially
    - between groups, logical is applied

    4. conditions = {group1: func1, group2: func2}
    5. conditions = {group1: [func1, func2], group2: [func3, func4]} ---- groups are subject to logical. funcs are evaluated sequentially, and assumed to have the same score.
    6. conditions = {group1: {func1, func2}, group2: {func3, func4}} ---- set notation. groups are subject to logical. funcs are evaluated sequentially, and assumed to have the same score.
    7. conditions = {group1: {(func1, score), (func2, score)}, group2: {(func3, score), (func4, score)}} ---- set notation. groups are subject to logical. funcs are evaluated sequentially, and the scores are specified.
    8. conditions = {group1: [(func1, score), (func2, score)], group2: [(func3, score), (func4, score)]} ---- list notation. groups are subject to logical. funcs are evaluated sequentially, and the scores are specified.

    This function unifies all of the different above structures into a single type, which is the 8th type.

    Returns:
        A dictionary mapping group names to lists of (function, score) tuples.
    """
    # Case 4-8: dict - validate and potentially convert
    if isinstance(conditions, dict):
        if len(conditions) == 0:
            raise ValueError("conditions dict cannot be empty")

        sanitized_dict = {}
        for group, obj_conditions in conditions.items():

            # Case 4: single callable
            if callable(obj_conditions):
                sanitized_dict[group] = [(obj_conditions, 1.0)]

                if robolab.constants.DEBUG:
                    print(f"[Sanitize Subtask Conditions] Converting single callable to list for group '{group}': {obj_conditions} -> {sanitized_dict[group]}")

            # Case 5 or 8: list
            elif isinstance(obj_conditions, list):
                if len(obj_conditions) == 0:
                    raise ValueError(f"conditions for group '{group}' cannot be empty")

                first_elem = obj_conditions[0]

                # Check if it's case 8 (list of tuples) or case 5 (list of callables)
                if isinstance(first_elem, tuple):
                    # Case 8: list of (func, score) tuples - validate
                    for i, condition in enumerate(obj_conditions):
                        if not isinstance(condition, tuple):
                            raise TypeError(f"condition at index {i} for group '{group}' must be a tuple, got {type(condition)}")
                        if len(condition) != 2:
                            raise ValueError(f"condition tuple at index {i} for group '{group}' must have 2 elements (func, score), got {len(condition)}")
                        if not callable(condition[0]):
                            raise TypeError(f"first element of condition tuple at index {i} for group '{group}' must be callable")
                        if not isinstance(condition[1], (int, float)):
                            raise TypeError(f"second element of condition tuple at index {i} for group '{group}' must be a number (score)")
                    sanitized_dict[group] = obj_conditions

                elif callable(first_elem):
                    # Case 5: list of callables - convert to list of tuples with equal scores
                    for i, condition in enumerate(obj_conditions):
                        if not callable(condition):
                            raise TypeError(f"condition at index {i} for group '{group}' must be callable, got {type(condition)}")
                    equal_score = 1.0 / len(obj_conditions)
                    sanitized_dict[group] = [(func, equal_score) for func in obj_conditions]

                    if robolab.constants.DEBUG:
                        print(f"[Sanitize Subtask Conditions] Converting list of callables to tuples for group '{group}': {obj_conditions} -> {sanitized_dict[group]}")

                else:
                    raise ValueError(f"list elements for group '{group}' must be either tuples (func, score) or callables, got {type(first_elem)}")

            # Case 6 or 7: set
            elif isinstance(obj_conditions, set):
                if len(obj_conditions) == 0:
                    raise ValueError(f"conditions for group '{group}' cannot be empty")

                # Convert set to list for iteration
                obj_conditions_list = list(obj_conditions)
                first_elem = obj_conditions_list[0]

                # Check if it's case 7 (set of tuples) or case 6 (set of callables)
                if isinstance(first_elem, tuple):
                    # Case 7: set of (func, score) tuples - validate and convert to list
                    for i, condition in enumerate(obj_conditions_list):
                        if not isinstance(condition, tuple):
                            raise TypeError(f"condition at index {i} for group '{group}' must be a tuple, got {type(condition)}")
                        if len(condition) != 2:
                            raise ValueError(f"condition tuple at index {i} for group '{group}' must have 2 elements (func, score), got {len(condition)}")
                        if not callable(condition[0]):
                            raise TypeError(f"first element of condition tuple at index {i} for group '{group}' must be callable")
                        if not isinstance(condition[1], (int, float)):
                            raise TypeError(f"second element of condition tuple at index {i} for group '{group}' must be a number (score)")
                    sanitized_dict[group] = obj_conditions_list
                    if robolab.constants.DEBUG:
                        print(f"[Sanitize Subtask Conditions] Converting set of tuples to list for group '{group}'")

                elif callable(first_elem):
                    # Case 6: set of callables - convert to list of tuples with equal scores
                    for i, condition in enumerate(obj_conditions_list):
                        if not callable(condition):
                            raise TypeError(f"condition at index {i} for group '{group}' must be callable, got {type(condition)}")
                    equal_score = 1.0 / len(obj_conditions_list)
                    sanitized_dict[group] = [(func, equal_score) for func in obj_conditions_list]

                    if robolab.constants.DEBUG:
                        print(f"[Sanitize Subtask Conditions] Converting set of callables to tuples for group '{group}'")

                else:
                    raise ValueError(f"set elements for group '{group}' must be either tuples (func, score) or callables, got {type(first_elem)}")

            else:
                raise TypeError(f"conditions for group '{group}' must be a callable, list, or set, got {type(obj_conditions)}")

        return sanitized_dict

    # Case 2 or 3: list - convert to dict format
    elif isinstance(conditions, list):
        if len(conditions) == 0:
            raise ValueError("conditions list cannot be empty")

        # Check if it's case 2 (list of tuples) or case 3 (list of callables)
        first_elem = conditions[0]

        if isinstance(first_elem, tuple):
            sanitized_conditions = {}
            # Case 2: list of (func, score) tuples
            for i, condition in enumerate(conditions):
                if not isinstance(condition, tuple):
                    raise TypeError(f"condition at index {i} must be a tuple, got {type(condition)}")
                if len(condition) != 2:
                    raise ValueError(f"condition tuple at index {i} must have 2 elements (func, score), got {len(condition)}")
                if not callable(condition[0]):
                    raise TypeError(f"first element of condition tuple at index {i} must be callable")
                if not isinstance(condition[1], (int, float)):
                    raise TypeError(f"second element of condition tuple at index {i} must be a number (score)")
                sanitized_conditions[f"group{i+1}"] = [condition]

            if robolab.constants.DEBUG:
                print(f"[Sanitize Subtask Conditions] Converting list of tuples to dict: {conditions} converted to {sanitized_conditions}")

            return sanitized_conditions

        elif callable(first_elem):
            # Convert to dict format with equal scores
            equal_score = 1.0 / len(conditions)
            sanitized_conditions = {}
            # Case 3: list of callables - assign equal scores
            for i, condition in enumerate(conditions):
                if not callable(condition):
                    raise TypeError(f"condition at index {i} must be callable, got {type(condition)}")
                sanitized_conditions[f"group{i+1}"] = [(condition, equal_score)]

            if robolab.constants.DEBUG:
                print(f"[Sanitize Subtask Conditions] Converting list of callables to dict: {conditions} converted to {sanitized_conditions}")

            return sanitized_conditions

        else:
            raise ValueError(f"list elements must be either tuples (func, score) or callables, got {type(first_elem)}")

    # Case 4 or 5: set - convert to dict format
    elif isinstance(conditions, set):
        if len(conditions) == 0:
            raise ValueError("conditions set cannot be empty")

        # Convert set to list for iteration
        conditions_list = list(conditions)
        first_elem = conditions_list[0]

        # Check if it's case 5 (set of tuples) or case 4 (set of callables)
        if isinstance(first_elem, tuple):
            sanitized_conditions = {}
            # Case 5: set of (func, score) tuples
            for i, condition in enumerate(conditions_list):
                if not isinstance(condition, tuple):
                    raise TypeError(f"condition at index {i} must be a tuple, got {type(condition)}")
                if len(condition) != 2:
                    raise ValueError(f"condition tuple at index {i} must have 2 elements (func, score), got {len(condition)}")
                if not callable(condition[0]):
                    raise TypeError(f"first element of condition tuple at index {i} must be callable")
                if not isinstance(condition[1], (int, float)):
                    raise TypeError(f"second element of condition tuple at index {i} must be a number (score)")
                sanitized_conditions[f"group{i+1}"] = [condition]

            if robolab.constants.DEBUG:
                print(f"[Sanitize Subtask Conditions] Converting set of tuples to dict: {conditions} converted to {sanitized_conditions}")

            return sanitized_conditions

        elif callable(first_elem):
            # Convert to dict format with equal scores
            equal_score = 1.0 / len(conditions_list)
            sanitized_conditions = {}
            # Case 4: set of callables - assign equal scores
            for i, condition in enumerate(conditions_list):
                if not callable(condition):
                    raise TypeError(f"condition at index {i} must be callable, got {type(condition)}")
                sanitized_conditions[f"group{i+1}"] = [(condition, equal_score)]

            if robolab.constants.DEBUG:
                print(f"[Sanitize Subtask Conditions] Converting set of callables to dict: {conditions} converted to {sanitized_conditions}")

            return sanitized_conditions

        else:
            raise ValueError(f"set elements must be either tuples (func, score) or callables, got {type(first_elem)}")

    # Case 1: single callable - convert to dict format
    elif callable(conditions):
        sanitized_conditions = {
            "conditions": [(conditions, 1.0)]
        }

        if robolab.constants.DEBUG:
            print(f"[Sanitize Subtask Conditions] Converting single callable to dict: {conditions} converted to {sanitized_conditions}")

        return sanitized_conditions

    else:
        raise ValueError(f"conditions must be a callable, list, set, or dict, got {type(conditions)}")


def normalize_conditions_scores(conditions: dict[str, list[tuple[Callable, float]]]) -> dict[str, list[tuple[Callable, float]]]:
    """
    Normalize the scores of the conditions.
    """
    normalized_conditions = {}
    for group_name, list_of_conditions_func_tuple in conditions.items():
        total_score = sum(score for (condition, score) in list_of_conditions_func_tuple)
        total_score = max(total_score, 1e-9) # avoid division by zero
        normalized_conditions[group_name] = [(condition, score / total_score) for (condition, score) in list_of_conditions_func_tuple]
    return normalized_conditions
