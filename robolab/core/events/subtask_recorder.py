# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import copy
from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

import robolab.constants
from robolab.core.task.event_tracker import EventTracker
from robolab.core.task.subtask_state_machine import SubtaskStateMachine


class SubtaskCompletionRecorderTerm(RecorderTerm):
    subtasks: list[dict[str, Any]] | None = None

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.subtasks = getattr(env.cfg, 'subtasks', None)
        self._num_envs = getattr(env, 'num_envs', 1)

        try:
            device = env.scene.device
        except Exception:
            device = torch.device("cpu")
        self._device = device

        if self.subtasks is None:
            self.subtask_state_machines = []
            self._event_tracker = None
        else:
            # One SubtaskStateMachine per env, each tracking independently
            self.subtask_state_machines = [
                SubtaskStateMachine(
                    env, env_id=eid,
                    subtasks=self.subtasks,
                    objects_in_scene=env.cfg.contact_object_list
                )
                for eid in range(self._num_envs)
            ]
            self._event_tracker = EventTracker(
                num_envs=self._num_envs, device=device
            )

        # Per-env info tracking
        self.infos: list[dict] = [
            {"status": 0, "completed": 0, "total": 0, "info": "", "score": 0.0}
            for _ in range(self._num_envs)
        ]
        self._final_infos: list[dict | None] = [None] * self._num_envs
        self._last_error_infos: list[tuple[str, int] | None] = [None] * self._num_envs
        self._last_error_scores: list[float] = [0.0] * self._num_envs

    @property
    def subtask_state_machine(self) -> SubtaskStateMachine | None:
        """Backward-compat: return env 0's state machine."""
        if not self.subtask_state_machines:
            return None
        return self.subtask_state_machines[0]

    @property
    def info(self) -> dict:
        """Backward-compat: return env 0's info."""
        return self.infos[0]

    def record_post_step(self):
        if not self.subtask_state_machines:
            return None, None

        # Determine which envs are frozen (terminated)
        frozen_mask = getattr(self._env, '_frozen_envs', None)
        if frozen_mask is None:
            frozen_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        # Detect envs that were frozen THIS step (newly terminated) — they still
        # need one final SM step to capture the success condition on the termination frame.
        pre_step_frozen = getattr(self._env, '_pre_step_frozen', None)
        if pre_step_frozen is not None:
            newly_frozen = frozen_mask & ~pre_step_frozen
        else:
            newly_frozen = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        # For event tracking and SM stepping, treat newly-frozen envs as still active
        effective_frozen = frozen_mask & ~newly_frozen

        # Compute per-env intended objects from each env's current CSM
        per_env_intended: list[set[str]] = []
        for eid in range(self._num_envs):
            sm = self.subtask_state_machines[eid]
            if sm.conditionals_state_machine is not None:
                per_env_intended.append(set(sm.conditionals_state_machine.subtask.group_names))
            else:
                per_env_intended.append(set())

        # Batch-check events across all envs (newly-frozen envs are still active for this step)
        all_events = self._event_tracker.check_events(
            env=self._env,
            per_env_intended=per_env_intended,
            frozen_mask=effective_frozen,
            verbose=robolab.constants.VERBOSE,
        )

        # Step each non-frozen env's state machine with its filtered events
        status_list = []
        completed_list = []
        score_list = []
        any_info = False

        for eid in range(self._num_envs):
            if effective_frozen[eid]:
                # Use last known values for previously-frozen envs
                status_list.append(self.infos[eid]["status"])
                completed_list.append(self.infos[eid]["completed"])
                score_list.append(self.infos[eid]["score"])
                continue

            # Filter events for this env: keep (info, status) where env_mask[eid] is True
            env_events = [
                (info_str, status_code)
                for info_str, status_code, env_mask in all_events
                if env_mask[eid]
            ]

            sm = self.subtask_state_machines[eid]
            done, info, status_code, all_status_codes = sm.step(env_events=env_events)

            subtask_state = sm.get_subtask_state()

            self.infos[eid]["status"] = status_code
            self.infos[eid]["info"] = info
            self.infos[eid]["completed"] = subtask_state["completed"]
            self.infos[eid]["total"] = subtask_state["total"]
            self.infos[eid]["score"] = subtask_state["score"]

            if all_status_codes:
                self.infos[eid]["all_status_codes"] = [
                    (cond_info, int(cond_status)) for cond_info, cond_status in all_status_codes
                ]
            else:
                self.infos[eid]["all_status_codes"] = []

            # Capture error info before auto-reset can cause regression
            current_score = subtask_state["score"]
            if not sm.is_complete():
                if self._last_error_infos[eid] is None or current_score >= self._last_error_scores[eid]:
                    error_info, error_code = sm.get_final_error_code()
                    self._last_error_infos[eid] = (error_info, int(error_code))
                    self._last_error_scores[eid] = current_score

            if info:
                any_info = True

            status_list.append(status_code)
            completed_list.append(subtask_state["completed"])
            score_list.append(subtask_state["score"])

        if not any_info:
            return None, None

        # Build per-env tensors
        status_tensor = torch.tensor(
            [int(s) for s in status_list], dtype=torch.uint16, device=self._device
        )
        completed_tensor = torch.tensor(
            completed_list, dtype=torch.uint8, device=self._device
        )
        score_tensor = torch.tensor(
            score_list, dtype=torch.float32, device=self._device
        )

        return "subtask", {
            "status": status_tensor,
            "completed": completed_tensor,
            "score": score_tensor,
        }

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, Any]:
        if not self.subtask_state_machines:
            return None

        if env_ids is None:
            env_ids = list(range(self._num_envs))

        for eid in env_ids:
            self.subtask_state_machines[eid].reset()

        if self._event_tracker is not None:
            self._event_tracker.reset_envs(list(env_ids))

        # Return env 0's info for backward compat
        return self.infos[0]

    def record_final_status(self) -> tuple[str, dict] | tuple[None, None]:
        """Record final status for all envs when episode ends incomplete."""
        if not self.subtask_state_machines:
            return None, None

        status_list = []
        completed_list = []
        score_list = []
        has_any_error = False

        for eid in range(self._num_envs):
            sm = self.subtask_state_machines[eid]

            if sm.is_complete() and self._last_error_infos[eid] is None:
                self._final_infos[eid] = None
                status_list.append(0)
                completed_list.append(self.infos[eid]["completed"])
                score_list.append(self.infos[eid]["score"])
                continue

            has_any_error = True

            # Use preserved error info
            if self._last_error_infos[eid] is not None:
                final_info, final_error = self._last_error_infos[eid]
            else:
                final_info, final_error = sm.get_final_error_code()
                final_error = int(final_error)

            self.infos[eid]["status"] = int(final_error)
            self.infos[eid]["info"] = final_info
            self._final_infos[eid] = copy.deepcopy(self.infos[eid])

            status_list.append(int(final_error))
            completed_list.append(self.infos[eid]["completed"])
            score_list.append(self.infos[eid]["score"])

        if not has_any_error:
            return None, None

        return "subtask", {
            "status": torch.tensor([int(s) for s in status_list], dtype=torch.uint16, device=self._device),
            "completed": torch.tensor(completed_list, dtype=torch.uint8, device=self._device),
            "score": torch.tensor(score_list, dtype=torch.float32, device=self._device),
        }

    def get_final_info(self, env_id: int | None = None) -> dict | list[dict | None] | None:
        """Get final info for incomplete episodes.

        Args:
            env_id: If None, return list of all envs' final infos.
                   If int, return that env's final info.
        """
        if env_id is None:
            return self._final_infos
        return self._final_infos[env_id]

    def clear(self):
        """Clear recording buffers.

        Note: This is called by the recorder manager during auto-flush to free
        memory. It must NOT reset the live state machine state (infos, error
        tracking) because those are needed for the full episode duration.
        The SM state is only reset via reset() when a new episode begins.
        """
        pass


@configclass
class SubtaskCompletionRecorderCfg(RecorderTermCfg):
    """Configuration for the subtask completion recorder term."""

    class_type: type[RecorderTerm] = SubtaskCompletionRecorderTerm

    subtasks: list[dict[str, Any]] | None = None
