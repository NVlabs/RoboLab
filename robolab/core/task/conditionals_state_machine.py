# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Any, Callable

import robolab.constants
from robolab.core.task.status import StatusCode
from robolab.core.task.subtask import Subtask
from robolab.core.task.subtask_utils import *
from robolab.core.utils.function_loader import func_as_str, get_callable_info
from robolab.core.world.world_state import WorldState, get_world


class ConditionalsStateMachine:
    """
    A state machine that manages parallel execution of subtasks across multiple objects.

    This class handles a single subtask group where multiple objects can independently progress
    through their own conditional sequences in parallel. Each object maintains its own state and
    progress tracker, allowing for order-independent completion while supporting regression
    checking to ensure task validity.

    Attributes:
        env (Any): Environment object containing simulation state
        env_id (int): Environment instance identifier
        world (WorldState): World state interface for querying object states
        logical (str): Completion mode - "all", "any", or "choose"
        subtask_group_id (int): Identifier for this subtask group
        objects_in_scene (list[str]): Available objects in the scene
        subtasks (Subtask): Processed subtasks organized by object name
        object_tracker (dict[str, int]): Current subtask index for each object
        object_completed_table (dict[str, list[bool]]): Completion status per object per subtask
    """

    def __init__(self,
                 env: Any,
                 env_id: int = 0,
                 subtask: Subtask = None,
                 objects_in_scene: list[str] | None = None,
                 subtask_group_id: int = 0) -> None:
        """
        Initialize the flexible subtask state manager.

        Args:
            env: The environment object
            env_id: Environment ID
            subtasks: Single subtask dictionary
            objects_in_scene: List of object names in the scene
        """
        self.env = env
        self.env_id = env_id
        self.world = get_world(env)
        self.subtask_group_id = subtask_group_id

        self.objects_in_scene = objects_in_scene if objects_in_scene else []
        self.subtask = subtask

        # Validate k parameter for "choose" mode
        if self.subtask.logical == "choose":
            if self.subtask.K is None:
                raise ValueError("Parameter 'k' is required when logical='choose'")
            if self.subtask.K < 1 or self.subtask.K > len(self.subtask.conditions):
                raise ValueError(f"Invalid k={self.subtask.K}: must be between 1 and {len(self.subtask.conditions)} (number of objects)")

        # Subtasks is a dict of object names to their subtasks
        # self.subtasks = {"apple": ([function, score], [function, score], ...), "banana": ([function, score], [function, score], ...)}

        # Track score for each object independently
        self.object_tracker = {} # [object_name] -> int showing the current subtask index
        self.object_completed_table = {} # [object_name][subtask_index] -> bool showing which subtasks are completed

        self._prev_object_tracker_state = {} # the last object tracker state

        # Per-step condition cache to avoid duplicate evaluations
        self._condition_cache: dict[int, tuple[bool, str]] = {}

        # Initialize score tracking
        self._initialize_tracker()

    def reset(self) -> None:
        """Reset all score to initial state."""
        self._initialize_tracker()
        self._condition_cache.clear()

    def _calculate_object_score(self, obj_name: str) -> float:
        """Calculate the total score of completed subtasks."""
        object_score = 0.0

        conditions = self.subtask.get_group(obj_name)
        if not conditions:
            return object_score

        # conditions is a list of (function, score) tuples
        for i, (conditional_func, score) in enumerate(conditions):

            if self.object_completed_table[obj_name][i]:
                object_score += score

        object_score = max(object_score, 0.0) # cap at 0
        return object_score

    def _initialize_tracker(self) -> None:
        """Initialize score tracking for all objects."""
        for group_name, subtasks in self.subtask.conditions.items():
            self.object_tracker[group_name] = 0
            self.object_completed_table[group_name] = [False] * len(subtasks)

        self._prev_object_tracker_state = dict(self.object_tracker)
        if robolab.constants.VERBOSE:
            self._print_current_state()

    def check_condition_satisfied(self, conditional_func: Callable) -> tuple[bool, str]:
        """
        Check if a single subtask is completed.

        Uses per-step caching to avoid duplicate evaluations of the same condition
        within a single step.

        Returns:
            Tuple of (success: bool, info_string: str)
        """
        # Use function id as cache key
        cache_key = id(conditional_func)
        cached = self._condition_cache.get(cache_key)
        if cached is not None:
            result, info = cached
            return result, info

        # Add environment parameters and evaluate
        params_with_env = {'env': self.env, 'env_id': self.env_id}
        result = conditional_func(**params_with_env)

        # Create info string only once per condition per step
        succ_fail = "success" if result else "failed"
        info = f"{succ_fail}: {func_as_str(conditional_func)}."

        # Cache the result for this step
        self._condition_cache[cache_key] = (result, info)

        return result, info

    def _check_previous_subtasks(self, obj_name: str, current_idx: int, k: int = 1) -> tuple[bool, int, str]:
        """
        Check k previous subtasks for an object and find the earliest failure point.

        Returns:
            Tuple of (all_previous_satisfied: bool, regress_to_idx: int, info: str)
        """
        conditions_list = self.subtask.get_group(obj_name)
        # subtask_keys = list(subtasks.keys())

        # Check all subtasks from current_idx to k previous subtasks, stopping at 0 (inclusive)
        for i in range(current_idx, max(current_idx-k-1, -1), -1):
            conditional_func, _ = conditions_list[i]
            result, info = self.check_condition_satisfied(conditional_func)
            if not result:
                return False, i, info

        return True, current_idx, f"Previous subtask {current_idx-k} satisfied."

    def _find_target_condition_iteratively(self, obj_name: str, current_idx: int) -> tuple[int, str, bool, list[int]]:
        """
        Iteratively find the target condition index by checking forward for advancement
        or backward for regression.

        Starting from current_idx:
        1. Check forward to find the latest satisfied condition (for advancement)
        2. Check backward to find the earliest unsatisfied condition (for regression)

        Args:
            obj_name: Name of the object to check
            current_idx: Current condition index for this object

        Returns:
            Tuple of (target_idx: int, info: str, is_advancement: bool, satisfied_indices: list[int])
            - target_idx: The index to move to (could be forward or backward from current)
            - info: Description of what was found
            - is_advancement: True if advancing forward, False if regressing backward
            - satisfied_indices: List of all condition indices that were satisfied during advancement
        """
        conditions_list = self.subtask.get_group(obj_name)
        num_conditions = len(conditions_list)

        # Check forward: scan ALL remaining conditions from current_idx onward.
        # Do NOT break on unsatisfied conditions — a later condition may be
        # satisfied even if an earlier transient one is not (e.g., object_grabbed
        # is False after release but object_in_container is True). If the later
        # condition is met, the earlier one must have been satisfied at some point.
        latest_satisfied_idx = current_idx
        satisfied_indices = []
        for i in range(current_idx, num_conditions):
            conditional_func, _ = conditions_list[i]
            result, info = self.check_condition_satisfied(conditional_func)
            if result:
                latest_satisfied_idx = i
                satisfied_indices.append(i)

        # Return advancement if any forward condition is satisfied
        if satisfied_indices:
            conditional_func, _ = conditions_list[latest_satisfied_idx]
            _, info = self.check_condition_satisfied(conditional_func)
            return latest_satisfied_idx, info, True, satisfied_indices

        # No forward progress — check backward for regression.
        # Only regress when the current condition is NOT satisfied and no later
        # condition is satisfied either (checked above).
        earliest_unsatisfied_idx = current_idx
        for i in range(current_idx - 1, -1, -1):
            conditional_func, _ = conditions_list[i]
            result, info = self.check_condition_satisfied(conditional_func)
            if not result:
                earliest_unsatisfied_idx = i
            else:
                break

        if earliest_unsatisfied_idx < current_idx:
            conditional_func, _ = conditions_list[earliest_unsatisfied_idx]
            _, info = self.check_condition_satisfied(conditional_func)
            return earliest_unsatisfied_idx, info, False, []

        return current_idx, f"Condition {current_idx} status unchanged", False, []

    def step(self, env_events: list[tuple[str, StatusCode]] | None = None) -> tuple[bool, str, StatusCode, list[tuple[str, StatusCode]]]:
        """
        Step all objects and return their status.

        Args:
            env_events: Pre-computed events for this env from the shared EventTracker.
                       If provided, these are appended to all_status_codes.

        Returns:
            Tuple of (done, info, status_code, all_status_codes)
            - done: Whether all objects have completed based on the logical mode
            - info: Description of what happened
            - status_code: Primary status code
            - all_status_codes: List of (info, status_code) for ALL satisfied conditions
        """
        # Clear per-step condition cache to ensure fresh evaluations
        self._condition_cache.clear()

        # Process each object independently
        info = ""
        error_code = StatusCode.OK
        all_status_codes = []

        object_status = {}
        for group_name in self.subtask.group_names:
            obj_info, obj_error_code, obj_all_status_codes = self._step_object(group_name)
            object_status[group_name] = {
                "info": obj_info,
                "error_code": obj_error_code,
                "all_status_codes": obj_all_status_codes
            }

        # Update last object score
        if not self._prev_object_tracker_state:
            self._prev_object_tracker_state = dict(self.object_tracker)
            changed = True
            info = ""
            error_code = StatusCode.OK
        else:
            changed_objects = [
                obj for obj in self.object_tracker
                if self.object_tracker.get(obj) != self._prev_object_tracker_state.get(obj)
            ]
            changed = bool(changed_objects)
            info = ""
            error_code = StatusCode.OK
            for obj in changed_objects:
                info += object_status[obj]["info"]
                error_code = object_status[obj]["error_code"] if object_status[obj]["error_code"] != StatusCode.OK else error_code
                # Collect all status codes from all changed objects
                all_status_codes.extend(object_status[obj]["all_status_codes"])

        # Append pre-computed events from the shared EventTracker
        if env_events:
            all_status_codes.extend(env_events)

        # Store a copy of the current score for next step comparison
        self._prev_object_tracker_state = dict(self.object_tracker)

        if robolab.constants.VERBOSE and changed:
            self._print_current_state()

        if self.subtask.logical == "all":
            done = self.check_all_objects_completed()
        elif self.subtask.logical == "any":
            done = self.check_any_object_completed()
        elif self.subtask.logical == "choose":
            done = self.check_choose_objects_completed()
        else:
            raise ValueError(f"Invalid completion mode: {self.subtask.logical}")

        if done and robolab.constants.DEBUG:
            print(f"[ConditionalStateMachine] '{self.subtask.name}' DONE. {info}, status_code: {error_code.name}")

        return done, info, error_code, all_status_codes

    def _step_object(self, obj_name: str) -> tuple[str, StatusCode, list[tuple[str, StatusCode]]]:
        """
        Step a single object's subtask progression using iterative condition checking.

        Returns:
            Tuple of (info, status_code, all_status_codes)
            - info: Description of what happened
            - status_code: Primary status code (first satisfied condition for advancement)
            - all_status_codes: List of (info, status_code) for ALL satisfied conditions
        """
        conditionals_list = self.subtask.get_group(obj_name)
        num_conditions = len(conditionals_list)
        current_idx = self.object_tracker[obj_name]

        info = ""
        status_code = StatusCode.OK
        all_status_codes = []

        # Check if all subtasks are completed
        if current_idx >= num_conditions:
            status_code = StatusCode.OK
            info = f"All conditions satisfied for {obj_name}."
            return info, status_code, all_status_codes

        # Use iterative function to find target condition (regression or advancement)
        target_idx, target_info, is_advancement, satisfied_indices = self._find_target_condition_iteratively(
            obj_name, current_idx
        )

        if target_idx < current_idx:
            # REGRESSION: Need to regress to an earlier condition
            # Clear completion status for all conditions from target_idx to current_idx
            for i in range(target_idx, current_idx):
                self.object_completed_table[obj_name][i] = False

            self.object_tracker[obj_name] = target_idx

            if robolab.constants.DEBUG:
                prev_function = conditionals_list[target_idx]
                print(f"REGRESSION: {obj_name} from step {current_idx} → {target_idx} {prev_function[0]}")
                print(f"Reason: {target_info}")

            current_func_name, _ = get_callable_info(conditionals_list[target_idx][0])
            info = f"{target_info} regressing to step {target_idx} for {obj_name}."
            status_code = StatusCode.subtask_to_error(current_func_name)

        elif target_idx >= current_idx and is_advancement:
            # ADVANCEMENT: One or more conditions satisfied, advance past target
            # Mark all conditions from current_idx to target_idx as completed
            for i in range(current_idx, target_idx + 1):
                self.object_completed_table[obj_name][i] = True

            self.object_tracker[obj_name] = target_idx + 1

            if robolab.constants.DEBUG:
                print(f"[DEBUG][{self.__class__.__name__}] Advancing: '{obj_name}' from step {current_idx} → {self.object_tracker[obj_name]} (jumped {target_idx - current_idx + 1} steps)")

            # Build status codes for ALL satisfied conditions (for logging)
            for satisfied_idx in satisfied_indices:
                func_name, _ = get_callable_info(conditionals_list[satisfied_idx][0])
                cond_status = StatusCode.subtask_to_success(func_name)
                _, cond_info = self.check_condition_satisfied(conditionals_list[satisfied_idx][0])
                all_status_codes.append((cond_info, cond_status))

            # Use the FIRST satisfied condition's status code as the primary status
            # This ensures all intermediate conditions get logged properly
            first_func_name, _ = get_callable_info(conditionals_list[current_idx][0])
            status_code = StatusCode.subtask_to_success(first_func_name)

            steps_advanced = target_idx - current_idx + 1
            info = f"{target_info} advanced {steps_advanced} step(s) to step {self.object_tracker[obj_name]} for {obj_name}."

            # Check if all subtasks are now completed
            if self.object_tracker[obj_name] >= num_conditions:
                info = f"{info} All conditions satisfied for {obj_name}."

        else:
            # target_idx == current_idx and not advancement: No change
            if robolab.constants.DEBUG:
                print(f"[DEBUG][{self.__class__.__name__}] Working on {obj_name} step {current_idx}")

            info = target_info
            status_code = StatusCode.OK

        return info, status_code, all_status_codes

    def check_all_objects_completed(self) -> bool:
        """Check if all objects have completed all their subtasks."""
        for obj_name, conditionals_list in self.subtask.conditions.items():
            if self.object_tracker[obj_name] < len(conditionals_list):
                return False
        return True

    def check_any_object_completed(self) -> bool:
        """Check if any object has completed all its subtasks."""
        for obj_name, conditionals_list in self.subtask.conditions.items():
            if self.check_object_completed(obj_name):
                return True
        return False

    def check_choose_objects_completed(self) -> bool:
        """
        Check if exactly k objects have completed all their subtasks.

        Returns:
            bool: True if at least k objects have completed, False otherwise
        """
        if self.subtask.K is None:
            raise ValueError("k must be set for 'choose' mode")

        completed_count = sum(
            1 for obj_name in self.subtask.group_names
            if self.check_object_completed(obj_name)
        )

        return completed_count >= self.subtask.K

    def check_object_completed(self, obj_name: str) -> bool:
        """Check if an object has completed all its subtasks."""
        return self.object_tracker[obj_name] == len(self.subtask.get_group(obj_name))

    def get_object_score(self, obj_name: str) -> float:
        return self._calculate_object_score(obj_name)

    @property
    def total_score(self) -> float:
        """Get overall score across all objects, based on the completion mode"""
        if not self.subtask.conditions:
            return 1.0

        if self.subtask.logical == "all":
            # For "all" mode, normalize total score to 1.0
            completed_score = 0.0

            for obj_name in self.subtask.group_names:
                completed_score += self._calculate_object_score(obj_name)
            completed_score = completed_score / len(self.subtask.group_names)
            return completed_score

        elif self.subtask.logical == "any":
            # For "any" mode, find the object with highest score
            max_score = 0.0
            for obj_name in self.subtask.group_names:
                obj_score = self.get_object_score(obj_name)
                max_score = max(max_score, obj_score)
            return max_score

        elif self.subtask.logical == "choose":
            # For "choose" mode, track progress toward k completions
            # Score based on top k objects by completion percentage
            if self.subtask.K is None:
                raise ValueError("k must be set for 'choose' mode")

            # Get completion percentage for each object
            object_scores = []
            for obj_name in self.subtask.group_names:
                obj_score = self.get_object_score(obj_name)
                object_scores.append(obj_score)

            # Sort in descending order and take top k
            object_scores.sort(reverse=True)
            top_k_scores = object_scores[:self.subtask.K]

            # Average of top k scores
            return sum(top_k_scores) / self.subtask.K if self.subtask.K > 0 else 0.0

        else:
            raise ValueError(f"Invalid completion mode: {self.subtask.logical}")

    @property
    def total_subtasks(self) -> int:
        """Get total number of subtasks."""
        total_count = 0
        for obj_name, subtasks in self.subtask.conditions.items():
            total_count += len(subtasks)
        return total_count

    @property
    def completed_subtasks(self) -> int:
        """Get number of completed subtasks."""
        completed_count = 0
        for obj_name, subtasks in self.subtask.conditions.items():
            for i in range(len(subtasks)):
                if self.object_completed_table[obj_name][i]:
                    completed_count += 1
        return completed_count

    def get_subtask_state(self) -> dict[str, int]:
        """Get current subtask score for all objects."""
        return dict(self.object_tracker)


    def get_final_error_code(self) -> tuple[str, StatusCode]:
        """
        Get error code for incomplete conditions at episode end.

        Called when episode terminates without completion to register
        an error code for the stalled condition.

        Returns:
            Tuple of (info: str, error_code: StatusCode)
        """
        # Collect incomplete objects and their states
        incomplete_objects = []
        for obj_name in self.subtask.group_names:
            conditions_list = self.subtask.get_group(obj_name)
            current_idx = self.object_tracker[obj_name]

            if current_idx < len(conditions_list):
                conditional_func, _ = conditions_list[current_idx]
                func_name, _ = get_callable_info(conditional_func)
                func_str = func_as_str(conditional_func)  # e.g., "object_dropped(obj='mug', target='bin')"
                error_code = StatusCode.subtask_to_error(func_name)
                incomplete_objects.append((obj_name, func_str, current_idx, len(conditions_list), error_code))

        if not incomplete_objects:
            # All conditions completed
            return "All conditions completed.", StatusCode.OK

        # Use the first incomplete object for error code
        obj_name, func_str, current_idx, total_steps, error_code = incomplete_objects[0]

        # Format message based on logical mode
        if self.subtask.logical == "any":
            # For "any" mode: none of the objects completed
            if len(incomplete_objects) == 1:
                info = f"Condition not satisfied: {func_str} (step {current_idx+1}/{total_steps})"
            else:
                info = f"No object completed. Best progress: {func_str} (step {current_idx+1}/{total_steps})"
        elif self.subtask.logical == "choose":
            # For "choose" mode: not enough objects completed
            completed_count = sum(1 for obj in self.subtask.group_names if self.check_object_completed(obj))
            info = f"Only {completed_count}/{self.subtask.K} objects completed. Failed: {func_str} (step {current_idx+1}/{total_steps})"
        else:
            # For "all" mode: report all incomplete objects
            if len(incomplete_objects) == 1:
                info = f"Condition not satisfied: {func_str} (step {current_idx+1}/{total_steps})"
            else:
                # List all incomplete conditions
                incomplete_strs = [f"{obj[1]} (step {obj[2]+1}/{obj[3]})" for obj in incomplete_objects]
                info = f"Conditions not satisfied: {'; '.join(incomplete_strs)}"

        return info, error_code

    def _print_current_state(self) -> None:
        """Print detailed current state of all objects and their subtasks."""

        def generate_subtask_string(obj_name: str, i: int, current_idx: int, conditional_func: Callable, score: float) -> str:
            subtask_str = func_as_str(conditional_func)
            status_symbol = "✅" if self.object_completed_table[obj_name][i] else "❌"
            # Mark current position
            if i == current_idx:
                marker = " <-- IN PROGRESS"
            elif i < current_idx:
                marker = " (completed)"
            else:
                marker = ""
            return f"{status_symbol} {subtask_str}{marker} (score={score:.2f})"


        object_str = ", ".join(f"{obj_name}: {self.object_tracker[obj_name]}/{len(subtasks)}(score={self.get_object_score(obj_name):.2f})" for obj_name, subtasks in self.subtask.conditions.items())

        if self.subtask.logical == "choose":
            logical_str = f"logical: {self.subtask.logical} (k={self.subtask.K})"
        else:
            logical_str = f"logical: {self.subtask.logical}"

        print(f"Subtask {self.subtask_group_id+1} '{self.subtask.name}': Overall: {self.completed_subtasks}/{self.total_subtasks} (score={self.total_score:.2f}), Object progress: [{object_str}], {logical_str}")

        for obj_name, subtasks in self.subtask.conditions.items():
            current_idx = self.object_tracker[obj_name]
            max_subtasks = len(subtasks)

            if max_subtasks > 1:
                print(f"  {obj_name} ({current_idx}/{max_subtasks}) (score={self.get_object_score(obj_name):.2f}):")
                for i, (conditional_func, score) in enumerate(subtasks):
                    subtask_str = generate_subtask_string(obj_name, i, current_idx, conditional_func, score)
                    print(f"    {i+1:2}. {subtask_str}")
            else:
                conditional_func, score = subtasks[0]
                subtask_str = generate_subtask_string(obj_name, 0, current_idx, conditional_func, score)
                print(f"  {subtask_str}")
