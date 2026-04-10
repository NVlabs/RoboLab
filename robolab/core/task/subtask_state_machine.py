# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Any

import robolab.constants
from robolab.core.task.conditionals_state_machine import ConditionalsStateMachine
from robolab.core.task.status import StatusCode
from robolab.core.task.subtask import Subtask


class SubtaskStateMachine:
    """
    A state machine that manages and executes a sequence of subtasks sequentially.

    This class orchestrates the execution of multiple subtask groups in order, where each
    subtask group is handled by a SubtaskGroupStateMachine. It tracks overall progress,
    manages state transitions between subtasks, and provides scoring mechanisms for task
    completion assessment.

    The state machine progresses through subtasks one at a time, only advancing to the next
    subtask when the current one is complete. Each subtask can be either an atomic function
    or a composite function that expands into multiple object-specific subtasks.
    """

    def __init__(self,
                 env: Any,
                 env_id: int = 0,
                 subtasks: list[dict[str, Any]] | None = None,
                 objects_in_scene: list[str] | None = None) -> None:
        """
        Initialize the sequential task state machine.

        Args:
            env: The environment object
            env_id: Environment ID
            subtasks: List of subtask dictionaries
            objects_in_scene: List of object names in the scene

        Example:
            subtasks = [
                (func(params), score),
                (func(params), score),
            ]
        """
        self.initialized = False
        self.env = env
        self.env_id = env_id
        self.objects_in_scene = objects_in_scene or []
        self.subtasks = subtasks or [] # A list of subtask groups.
        self.total_subtasks = len(self.subtasks)
        self.conditionals_state_machine: ConditionalsStateMachine | None = None

        # Current state tracking
        self.current_subtask_index = 0 # Index of current subtask (also count of completed subtasks)
        self.unnormalized_total_score = calculate_unnormalized_total_score(self.subtasks)

        # Initialize state machine
        self._initialize_state_machine()
        self.initialized = True
        if robolab.constants.VERBOSE:
            self.print_current_state()

    def reset(self) -> None:
        """Reset to the beginning of the sequential tasks."""
        self.initialized = True
        self.current_subtask_index = 0
        self.conditionals_state_machine = None
        self._initialize_state_machine()


    def _clear_state_machine(self) -> None:
        """Clear the subtask state machine."""
        self.conditionals_state_machine = None

    def _initialize_state_machine(self) -> None:
        """Create a new SubtaskGroupStateMachine for the current subtask."""
        if self.current_subtask_index < len(self.subtasks):
            current_subtask = self.subtasks[self.current_subtask_index]
            self.conditionals_state_machine = ConditionalsStateMachine(
                env=self.env,
                env_id=self.env_id,
                subtask=current_subtask,  # Pass the current subtask directly
                objects_in_scene=self.objects_in_scene,
                subtask_group_id=self.current_subtask_index
            )
        else:
            self.conditionals_state_machine = None

    def _advance_to_next_subtask(self) -> None:
        """Move to the next subtask in the sequence."""
        self.current_subtask_index += 1
        if self.current_subtask_index >= len(self.subtasks):
            if robolab.constants.DEBUG:
                print("[SubtaskStateMachine] Clearing state machine because we've completed the last subtask")
            self._clear_state_machine()
        else:
            if robolab.constants.DEBUG:
                print(f"[SubtaskStateMachine] Advancing to next subtask {self.current_subtask_index}, total subtasks: {len(self.subtasks)}")
            self._initialize_state_machine()

    def step(self, env_events: list[tuple[str, StatusCode]] | None = None) -> tuple[bool, str, StatusCode, list[tuple[str, StatusCode]]]:
        """
        Step the state machine.

        Args:
            env_events: Pre-computed events for this env from the shared EventTracker.

        Returns:
            Tuple of (all_sequential_complete: bool, info: str, status_code: StatusCode, all_status_codes: list)
            - all_sequential_complete: Whether all sequential subtasks are complete
            - info: Description of what happened
            - status_code: Primary status code
            - all_status_codes: List of (info, status_code) for ALL satisfied conditions
        """
        # Check if we're completely done
        if self.is_complete():
            info = "All tasks completed."
            return True, info, StatusCode.OK, []

        # Step the current state machine
        assert self.conditionals_state_machine is not None, "State machine should exist"

        complete, info, status_code, all_status_codes = self.conditionals_state_machine.step(env_events=env_events)


        if complete and not self.is_complete():
            # Current subtask completed, advance to next
            self._advance_to_next_subtask()
            if robolab.constants.VERBOSE:
                self.print_current_state()
            msg = f"Completed subtask '{self.subtasks[self.current_subtask_index-1].name}' {self.current_subtask_index}/{self.total_subtasks}"
            return False, msg, status_code, all_status_codes
        elif complete and self.is_complete():
            if robolab.constants.VERBOSE:
                self.print_current_state()
            return True, "All tasks completed.", status_code, all_status_codes

        # Still in progress
        return False, info, status_code, all_status_codes

    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return self.current_subtask_index >= len(self.subtasks)

    def get_subtask_state(self) -> dict[str, Any]:
        """Get current subtask progress."""
        if self.is_complete():
            return {
                "completed": self.total_subtasks,
                "total": self.total_subtasks,
                "score": self.total_scores
            }

        return {
            "completed": self.current_subtask_index,
            "total": self.total_subtasks,
            "score": self.total_scores
        }

    def get_total_score(self) -> float:
        """Get overall scores across all subtasks."""
        if self.is_complete():
            return 1.0

        # Calculate overall progress: completed subtasks + current progress
        completed_portion = 0.0
        # Total score = sum(s_i * w_i / sum(w_i)). For completed tasks, s_i is always 1.0.
        for i in range(self.current_subtask_index):
            completed_portion += self.subtasks[i].score / self.unnormalized_total_score

        if self.conditionals_state_machine is None:
            return completed_portion

        current_portion = self.conditionals_state_machine.total_score * self.conditionals_state_machine.subtask.score / self.unnormalized_total_score
        return completed_portion + current_portion

    @property
    def total_scores(self) -> float:
        """Get overall scores across all objects."""
        return self.get_total_score()

    def get_final_error_code(self) -> tuple[str, "StatusCode"]:
        """
        Get error code for incomplete tasks at episode end.

        Called when episode terminates without completion to register
        an error code for the stalled condition.

        Returns:
            Tuple of (info: str, error_code: StatusCode)
        """
        if self.is_complete():
            return "All tasks completed.", StatusCode.OK

        # Get error from current conditionals state machine
        if self.conditionals_state_machine is not None:
            return self.conditionals_state_machine.get_final_error_code()

        return "Task incomplete.", StatusCode.UNKNOWN_FAILURE

    def print_subtasks(self, verbose: bool = False) -> None:
        for i, subtask in enumerate(self.subtasks):
            status = "✅" if i < self.current_subtask_index else "<-- IN PROGRESS" if i == self.current_subtask_index else " "
            if isinstance(subtask, Subtask):
                subtask.print_conditions(status, verbose=verbose)

    def print_current_state(self) -> None:
        """Print detailed current state of the sequential state machine."""
        len_of_divider = 60
        print(f"\n{'='*len_of_divider}")
        print("SUBTASK STATE MACHINE")
        print(f"{'='*len_of_divider}")
        print(f"Overall progress: {self.get_total_score():.1%}")
        print(f"Completed subtasks: {self.current_subtask_index}/{self.total_subtasks}")
        print(f"{'-'*len_of_divider}")
        self.print_subtasks(verbose=robolab.constants.VERBOSE)
        if self.conditionals_state_machine is not None:
            print(f"{'-'*len_of_divider}")
            self.conditionals_state_machine._print_current_state()
        print(f"{'='*len_of_divider}")


def calculate_unnormalized_total_score(subtasks: list[Subtask]) -> float:
    """Calculate the normalized total score."""
    total_score = 0.0
    for subtask in subtasks:
        total_score += subtask.score
    return total_score
