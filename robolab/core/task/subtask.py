# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union, cast

from robolab.core.task.subtask_utils import normalize_conditions_scores, sanitize_subtask_conditions
from robolab.core.utils.function_loader import func_as_str


@dataclass
class Subtask:
    """
    A self-documenting container for a group of parallel conditions.

    This dataclass encapsulates conditions that execute in parallel across multiple objects,
    along with the completion logic and scoring information. It provides type safety and
    clarity when defining task structures.

    Attributes:
        conditions: Can be one of the following formats:
            1. conditions = func ---- Single condition (atomic subtask)
            2. conditions = [func1, func2] ---- list notation, assumed to have the same score, processed as a single group, and is subject to logical ('all', 'any', 'choose'), funcs are evaluated in parallel.
            3. conditions = {func1, func2} ---- set notation, assumed to have the same score, and is subject to logical ('all', 'any', 'choose'). funcs evaluated in parallel.
            4. conditions = [(func1, score), (func2, score)] ---- list of tuples, the scores are specified, and is subject to logical ('all', 'any', 'choose'). funcs evaluated in parallel.
            5. conditions = {(func1, score), (func2, score)} ---- set of tuples, the scores are specified, and is subject to logical ('all', 'any', 'choose'). funcs evaluated in parallel.

            If the conditions are provided in the form of a dictionary, then:
            - within each group, the conditions are checked sequentially
            - between groups, logical is applied

            4. conditions = {group1: func1, group2: func2}
            5. conditions = {group1: [func1, func2], group2: [func3, func4]} ---- groups are subject to logical. funcs are evaluated sequentially, and assumed to have the same score.
            6. conditions = {group1: {func1, func2}, group2: {func3, func4}} ---- set notation. groups are subject to logical. funcs are evaluated sequentially, and assumed to have the same score.
            7. conditions = {group1: {(func1, score), (func2, score)}, group2: {(func3, score), (func4, score)}} ---- set notation. groups are subject to logical. funcs are evaluated sequentially, and the scores are specified.
            8. conditions = {group1: [(func1, score), (func2, score)], group2: [(func3, score), (func4, score)]} ---- list notation. groups are subject to logical. funcs are evaluated sequentially, and the scores are specified.

            All conditions will be sanitized to follow the 8th format.

        logical: Completion mode determining when this group is considered complete:
            - "all": All objects must complete all their conditions (default)
            - "any": Success when any single object completes all its conditions
            - "choose": Success when exactly k objects complete all their conditions

        score: Overall score weight for this subtask group (0.0 to 1.0)

        k: Number of objects that must complete when logical="choose" (required for choose mode)

    # To check the valid formats for `conditions`, see the docstring of `sanitize_subtask_conditions` in `subtask_utils.py`
    """
    conditions: Union[
        Callable,
        list[tuple[Callable, float]],
        list[Callable],
        dict[str, list[tuple[Callable, float]]]
    ]
    score: float = 1.0
    logical: Literal["all", "any", "choose"] = "all"
    K: Optional[int] = None  # Required when logical="choose"
    name: str = "unnamed_subtask"

    def __post_init__(self):
        """Validate the subtask group configuration."""
        # Sanitize conditions structure to unified dict format
        self.conditions = cast(
            dict[str, list[tuple[Callable, float]]],
            sanitize_subtask_conditions(self.conditions)
        )

        # Normalize the scores of the conditions within each group to sum to 1.0.
        self.conditions = normalize_conditions_scores(self.conditions)

        # Validate score range
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")

        # Validate logical mode and k parameter
        if self.logical == "choose":
            if self.K is None:
                raise ValueError("Parameter 'K' is required when logical='choose'")
            # At this point, type checker knows k is not None
            if self.K < 1 or self.K > len(self.conditions):
                raise ValueError(f"Invalid K={self.K}: must be between 1 and {len(self.conditions)} (number of objects)")

    def print_conditions(self, status: str = " ", verbose: bool = False) -> None:
        """Print the conditions in this group."""
        if self.logical == "choose":
            print(f"Subtask: '{self.name}', logical='{self.logical}', score='{self.score}', K='{self.K}' {status}")
        else:
            print(f"Subtask: '{self.name}', logical='{self.logical}', score='{self.score}' {status}")

        if not verbose:
            return

        conditions_dict = cast(dict[str, list[tuple[Callable, float]]], self.conditions)
        if len(conditions_dict) > 1:
            for i, (group_names, conditions) in enumerate(conditions_dict.items()):
                    print(f"{i+1}. {group_names}:")
                    for condition in conditions:
                        # each subtask is a tuple of (function, score)
                        print(f"  {func_as_str(condition[0])}")
        else:
            conditions = list(conditions_dict.values())[0]
            print(f"  {func_as_str(conditions[0][0])}")

    def __repr__(self) -> str:
        """Provide a clear string representation for debugging."""
        # After __post_init__, conditions is always a dict
        conditions_dict = cast(dict[str, list[tuple[Callable, float]]], self.conditions)
        group_names = list(conditions_dict.keys())
        if self.logical == "choose":
            return f"Subtask(groups={group_names}, logical='{self.logical}', K={self.K}"
        return f"Subtask(groups={group_names}, logical='{self.logical}'"

    @property
    def group_names(self) -> list[str]:
        """Get the names of the groups in the subtask."""
        return list(self.conditions.keys())

    def get_group(self, group_name) -> list[tuple[Callable, float]]:
        """Get the conditions for a specific group in the subtask."""
        return self.conditions.get(group_name)

########################################################
# Validation functions
########################################################

def make_subtask_title(obj, action, container=None):
    if container:
        return f"{obj}_{action}_in_{container}"
    else:
        return f"{obj}_{action}"
