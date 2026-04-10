# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Any

import torch

from robolab.core.task.conditionals import (
    get_wrong_object_grabbed,
    gripper_fully_closed,
    gripper_hit_table,
    object_grabbed,
    object_upright,
)
from robolab.core.task.predicate_logic import in_contact
from robolab.core.task.status import StatusCode
from robolab.core.world.world_state import get_world


class EventTracker:
    """
    Tracks grasp-related events across multiple parallel environments.

    Uses batched WorldState queries (env_id=None) for efficient per-env event
    detection. All internal state is stored as (num_envs,) tensors.

    Events tracked:
    - WRONG_OBJECT_GRABBED: When gripper grabs an object not in the intended target list
    - GRIPPER_HIT_TABLE: When gripper makes contact with table
    - GRIPPER_FULLY_CLOSED: When gripper closes fully (potential failed grasp)
    - OBJECT_STARTED_MOVING: Non-target object transitioned from stationary to moving
    - OBJECT_BUMPED: When object stops after small movement (< move_threshold), minor collision
    - OBJECT_MOVED: When object stops after significant movement (>= move_threshold), knocked/pushed
    - OBJECT_OUT_OF_SCENE: Object moved outside the workspace bounding box (fell off table)
    - OBJECT_TIPPED_OVER: Object that should be upright has fallen over
    - TARGET_OBJECT_DROPPED: Target object was grabbed but dropped mid-transport
    - GRIPPER_HIT_OBJECT: Gripper collided with a non-target object
    - MULTIPLE_OBJECTS_GRABBED: Gripper is in contact with multiple objects simultaneously

    Each event is recorded only on first occurrence per env. The tracker resets when
    the condition clears, allowing the event to be recorded again if it reoccurs.
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: torch.device = None,
        bump_threshold: float = 0.05,
        move_threshold: float = 0.50,
        velocity_threshold: float = 0.05,
        workspace_center: tuple[float, float, float] = (0.55, 0.0, 0.5),
        workspace_size: tuple[float, float, float] = (2.0, 2.0, 2.0)
    ):
        self.num_envs = num_envs
        self.device = device or torch.device("cpu")
        self.bump_threshold = bump_threshold
        self.move_threshold = move_threshold
        self.velocity_threshold = velocity_threshold

        self.workspace_center = torch.tensor(workspace_center, device=self.device)
        self.workspace_half_size = torch.tensor(workspace_size, device=self.device) / 2.0
        self.reset()

    def reset(self) -> None:
        """Reset all event trackers to initial state for all envs."""
        N, dev = self.num_envs, self.device

        # Per-env wrong object grab tracking (string names, must be dict)
        self._recorded_wrong_object_grab: dict[int, str | None] = {i: None for i in range(N)}

        # Per-env bool tensors
        self._recorded_gripper_hit_table = torch.zeros(N, dtype=torch.bool, device=dev)
        self._recorded_gripper_fully_closed = torch.zeros(N, dtype=torch.bool, device=dev)
        self._recorded_multiple_grab = torch.zeros(N, dtype=torch.bool, device=dev)
        self._target_was_grabbed = torch.zeros(N, dtype=torch.bool, device=dev)
        self._recorded_target_dropped = torch.zeros(N, dtype=torch.bool, device=dev)

        # Per-object per-env state (populated lazily)
        self._object_is_moving: dict[str, torch.Tensor] = {}          # obj -> (N,) bool
        self._position_when_started_moving: dict[str, torch.Tensor] = {}  # obj -> (N, 3)
        self._started_moving_mask: dict[str, torch.Tensor] = {}       # obj -> (N,) bool: which envs have a start pos
        self._recorded_out_of_scene: dict[str, torch.Tensor] = {}     # obj -> (N,) bool
        self._recorded_tipped_objects: dict[str, torch.Tensor] = {}   # obj -> (N,) bool
        self._recorded_gripper_hit_objects: dict[str, torch.Tensor] = {}  # obj -> (N,) bool

    def reset_envs(self, env_ids: list[int]) -> None:
        """Reset event state for specific envs only."""
        for eid in env_ids:
            self._recorded_wrong_object_grab[eid] = None
        idx = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        self._recorded_gripper_hit_table[idx] = False
        self._recorded_gripper_fully_closed[idx] = False
        self._recorded_multiple_grab[idx] = False
        self._target_was_grabbed[idx] = False
        self._recorded_target_dropped[idx] = False
        for d in (self._object_is_moving, self._position_when_started_moving,
                  self._started_moving_mask, self._recorded_out_of_scene,
                  self._recorded_tipped_objects, self._recorded_gripper_hit_objects):
            for t in d.values():
                t[idx] = 0

    def _is_outside_workspace_batched(self, positions: torch.Tensor) -> torch.Tensor:
        """Check if positions are outside workspace. positions: (N, 3), returns (N,) bool."""
        diff = torch.abs(positions - self.workspace_center)
        return torch.any(diff > self.workspace_half_size, dim=-1)

    def _get_not_intended_mask(self, obj_name: str, per_env_intended: list[set[str]]) -> torch.Tensor:
        """Return (N,) bool mask: True where obj_name is NOT in that env's intended set."""
        return torch.tensor(
            [obj_name not in per_env_intended[eid] for eid in range(self.num_envs)],
            dtype=torch.bool, device=self.device
        )

    def check_events(
        self,
        env: Any,
        per_env_intended: list[set[str]],
        frozen_mask: torch.Tensor | None = None,
        ignore_objects: list[str] = None,
        upright_objects: list[str] = None,
        verbose: bool = False,
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        """
        Check for events across all envs using batched queries.

        Args:
            env: The environment object
            per_env_intended: Per-env sets of intended target object names
            frozen_mask: (num_envs,) bool tensor, True for frozen envs to skip
            ignore_objects: Objects to ignore (default: ["table"])
            upright_objects: Objects that should remain upright
            verbose: Whether to print event messages

        Returns:
            List of (info_string, StatusCode, env_mask) where env_mask is (num_envs,) bool
            indicating which envs the event applies to.
        """
        events = []
        if frozen_mask is None:
            frozen_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        active_mask = ~frozen_mask

        if ignore_objects is None:
            ignore_objects = ["table"]
        ignore_set = set(ignore_objects)

        world = get_world(env)

        # --- Wrong object grabbed (per-env loop, returns string) ---
        for eid in range(self.num_envs):
            if frozen_mask[eid]:
                continue
            wrong_obj = get_wrong_object_grabbed(env, list(per_env_intended[eid]), env_id=eid)
            if wrong_obj is not None:
                if self._recorded_wrong_object_grab[eid] != wrong_obj:
                    info = f"Wrong object grabbed: '{wrong_obj}' (target objects: {list(per_env_intended[eid])})"
                    mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    mask[eid] = True
                    events.append((info, StatusCode.WRONG_OBJECT_GRABBED, mask))
                    self._recorded_wrong_object_grab[eid] = wrong_obj
                    if verbose:
                        print(f"[EventTracker] env{eid}: {info}")
            else:
                if self._recorded_wrong_object_grab[eid] is not None:
                    info = f"Wrong object that was grabbed is now detached: '{self._recorded_wrong_object_grab[eid]}'"
                    mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    mask[eid] = True
                    events.append((info, StatusCode.OK, mask))
                    if verbose:
                        print(f"[EventTracker] env{eid}: {info}")
                self._recorded_wrong_object_grab[eid] = None

        # --- Gripper hit table (batched) ---
        hit_table = gripper_hit_table(env, env_id=None)  # (N,) bool
        new_hit = hit_table & ~self._recorded_gripper_hit_table & active_mask
        if new_hit.any():
            events.append(("Gripper hit table", StatusCode.GRIPPER_HIT_TABLE, new_hit.clone()))
            self._recorded_gripper_hit_table |= new_hit
            if verbose:
                envs = new_hit.nonzero(as_tuple=False).squeeze(-1).tolist()
                print(f"[EventTracker] envs {envs}: Gripper hit table")
        # Reset recording for envs where condition cleared
        cleared = ~hit_table & self._recorded_gripper_hit_table & active_mask
        self._recorded_gripper_hit_table &= ~cleared

        # --- Gripper fully closed (batched) ---
        fully_closed = gripper_fully_closed(env, env_id=None)  # (N,) bool
        new_closed = fully_closed & ~self._recorded_gripper_fully_closed & active_mask
        if new_closed.any():
            events.append(("Gripper fully closed", StatusCode.GRIPPER_FULLY_CLOSED, new_closed.clone()))
            self._recorded_gripper_fully_closed |= new_closed
            if verbose:
                envs = new_closed.nonzero(as_tuple=False).squeeze(-1).tolist()
                print(f"[EventTracker] envs {envs}: Gripper fully closed")
        cleared = ~fully_closed & self._recorded_gripper_fully_closed & active_mask
        self._recorded_gripper_fully_closed &= ~cleared

        # --- Movement transitions (batched per object) ---
        movement_events = self._check_movement_transitions_batched(
            env, per_env_intended, ignore_set, active_mask, verbose
        )
        events.extend(movement_events)

        # --- Out of scene (batched per object) ---
        out_events = self._check_out_of_scene_batched(
            env, per_env_intended, ignore_set, active_mask, verbose
        )
        events.extend(out_events)

        # --- Tipped objects (batched per object) ---
        if upright_objects:
            tipped_events = self._check_tipped_objects_batched(
                env, upright_objects, active_mask, verbose
            )
            events.extend(tipped_events)

        # --- Target dropped (batched) ---
        drop_events = self._check_target_dropped_batched(
            env, per_env_intended, active_mask, verbose
        )
        events.extend(drop_events)

        # --- Gripper-object collision (batched per object) ---
        collision_events = self._check_gripper_object_collision_batched(
            env, per_env_intended, ignore_set, active_mask, verbose
        )
        events.extend(collision_events)

        # --- Multiple objects grabbed (batched) ---
        multi_events = self._check_multiple_objects_grabbed_batched(
            env, ignore_set, active_mask, verbose
        )
        events.extend(multi_events)

        return events

    def _check_movement_transitions_batched(
        self, env, per_env_intended, ignore_set, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []
        world = get_world(env)

        objects_to_check = [
            obj for obj in world.objects.keys()
            if obj not in ignore_set
        ]

        for obj_name in objects_to_check:
            not_intended = self._get_not_intended_mask(obj_name, per_env_intended)
            eligible = not_intended & active_mask

            if not eligible.any():
                continue

            try:
                current_pos, _ = world.get_pose(obj_name, env_id=None)  # (N, 3)
                velocity = world.get_velocity(obj_name, env_id=None)    # (N, 6)
                linear_speed = torch.norm(velocity[:, :3], dim=-1)       # (N,)
                is_moving = linear_speed > self.velocity_threshold

                was_moving = self._object_is_moving.get(
                    obj_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                )

                # Started moving
                started = is_moving & ~was_moving & eligible
                if started.any():
                    if obj_name not in self._position_when_started_moving:
                        self._position_when_started_moving[obj_name] = torch.zeros(self.num_envs, 3, device=self.device)
                        self._started_moving_mask[obj_name] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._position_when_started_moving[obj_name][started] = current_pos[started]
                    self._started_moving_mask[obj_name] |= started
                    # Don't emit OBJECT_STARTED_MOVING as a separate event in the return;
                    # it's used internally for displacement tracking

                # Stopped moving
                stopped = ~is_moving & was_moving & eligible
                has_start = self._started_moving_mask.get(
                    obj_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                )
                stopped_with_start = stopped & has_start

                if stopped_with_start.any():
                    start_pos = self._position_when_started_moving[obj_name]
                    displacement = torch.norm(current_pos - start_pos, dim=-1)  # (N,)

                    moved_mask = stopped_with_start & (displacement >= self.move_threshold)
                    if moved_mask.any():
                        avg_disp = displacement[moved_mask].mean().item()
                        events.append((
                            f"Object moved: '{obj_name}' displaced {avg_disp:.3f}m",
                            StatusCode.OBJECT_MOVED,
                            moved_mask.clone()
                        ))
                        if verbose:
                            envs = moved_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                            print(f"[EventTracker] envs {envs}: Object moved: '{obj_name}'")

                    bumped_mask = stopped_with_start & (displacement >= self.bump_threshold) & (displacement < self.move_threshold)
                    if bumped_mask.any():
                        avg_disp = displacement[bumped_mask].mean().item()
                        events.append((
                            f"Object bumped: '{obj_name}' nudged {avg_disp:.3f}m",
                            StatusCode.OBJECT_BUMPED,
                            bumped_mask.clone()
                        ))
                        if verbose:
                            envs = bumped_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                            print(f"[EventTracker] envs {envs}: Object bumped: '{obj_name}'")

                    # Clear start positions for stopped envs
                    self._started_moving_mask[obj_name] &= ~stopped_with_start

                self._object_is_moving[obj_name] = is_moving

            except Exception:
                continue

        return events

    def _check_out_of_scene_batched(
        self, env, per_env_intended, ignore_set, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []
        world = get_world(env)

        for obj_name in world.objects.keys():
            if obj_name in ignore_set:
                continue

            not_intended = self._get_not_intended_mask(obj_name, per_env_intended)
            already_recorded = self._recorded_out_of_scene.get(
                obj_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            eligible = not_intended & active_mask & ~already_recorded

            if not eligible.any():
                continue

            try:
                current_pos, _ = world.get_pose(obj_name, env_id=None)  # (N, 3)
                outside = self._is_outside_workspace_batched(current_pos)
                new_outside = outside & eligible

                if new_outside.any():
                    events.append((
                        f"Object out of scene: '{obj_name}'",
                        StatusCode.OBJECT_OUT_OF_SCENE,
                        new_outside.clone()
                    ))
                    if obj_name not in self._recorded_out_of_scene:
                        self._recorded_out_of_scene[obj_name] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._recorded_out_of_scene[obj_name] |= new_outside
                    if verbose:
                        envs = new_outside.nonzero(as_tuple=False).squeeze(-1).tolist()
                        print(f"[EventTracker] envs {envs}: Object out of scene: '{obj_name}'")

            except Exception:
                continue

        return events

    def _check_tipped_objects_batched(
        self, env, upright_objects, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []

        for obj_name in upright_objects:
            already_recorded = self._recorded_tipped_objects.get(
                obj_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            eligible = active_mask & ~already_recorded
            if not eligible.any():
                continue

            try:
                # object_upright returns (N,) bool when env_id=None
                is_upright = object_upright(env, obj_name, tolerance=0.3, env_id=None)
                tipped = ~is_upright & eligible

                if tipped.any():
                    events.append((
                        f"Object tipped over: '{obj_name}'",
                        StatusCode.OBJECT_TIPPED_OVER,
                        tipped.clone()
                    ))
                    if obj_name not in self._recorded_tipped_objects:
                        self._recorded_tipped_objects[obj_name] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._recorded_tipped_objects[obj_name] |= tipped
                    if verbose:
                        envs = tipped.nonzero(as_tuple=False).squeeze(-1).tolist()
                        print(f"[EventTracker] envs {envs}: Object tipped over: '{obj_name}'")

            except Exception:
                continue

        return events

    def _check_target_dropped_batched(
        self, env, per_env_intended, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []

        # Check if any target is currently grabbed per env
        any_grabbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Collect all unique intended objects across envs
        all_intended = set()
        for s in per_env_intended:
            all_intended.update(s)

        for obj_name in all_intended:
            try:
                grabbed = object_grabbed(env, obj_name, env_id=None)  # (N,) bool
                # Only count for envs where this object IS intended
                is_intended = torch.tensor(
                    [obj_name in per_env_intended[eid] for eid in range(self.num_envs)],
                    dtype=torch.bool, device=self.device
                )
                any_grabbed |= (grabbed & is_intended)
            except Exception:
                continue

        # Detect drop: was grabbed -> now not grabbed
        dropped = self._target_was_grabbed & ~any_grabbed & active_mask & ~self._recorded_target_dropped
        if dropped.any():
            events.append((
                "Target object dropped during transport",
                StatusCode.TARGET_OBJECT_DROPPED,
                dropped.clone()
            ))
            self._recorded_target_dropped |= dropped
            if verbose:
                envs = dropped.nonzero(as_tuple=False).squeeze(-1).tolist()
                print(f"[EventTracker] envs {envs}: Target object dropped")

        self._target_was_grabbed = any_grabbed

        # Reset drop recording for envs that grab again
        re_grabbed = any_grabbed & self._recorded_target_dropped
        self._recorded_target_dropped &= ~re_grabbed

        return events

    def _check_gripper_object_collision_batched(
        self, env, per_env_intended, ignore_set, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []
        world = get_world(env)

        candidates = [
            obj for obj in world.objects.keys()
            if obj not in ignore_set
        ]

        for obj_name in candidates:
            not_intended = self._get_not_intended_mask(obj_name, per_env_intended)
            already_recorded = self._recorded_gripper_hit_objects.get(
                obj_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            eligible = not_intended & active_mask & ~already_recorded

            if not eligible.any():
                continue

            try:
                contact = in_contact(world, "gripper", obj_name, env_id=None)  # (N,) bool
                new_contact = contact & eligible

                if new_contact.any():
                    events.append((
                        f"Gripper hit object: '{obj_name}'",
                        StatusCode.GRIPPER_HIT_OBJECT,
                        new_contact.clone()
                    ))
                    if obj_name not in self._recorded_gripper_hit_objects:
                        self._recorded_gripper_hit_objects[obj_name] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._recorded_gripper_hit_objects[obj_name] |= new_contact
                    if verbose:
                        envs = new_contact.nonzero(as_tuple=False).squeeze(-1).tolist()
                        print(f"[EventTracker] envs {envs}: Gripper hit object: '{obj_name}'")

            except Exception:
                continue

        return events

    def _check_multiple_objects_grabbed_batched(
        self, env, ignore_set, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []
        eligible = active_mask & ~self._recorded_multiple_grab

        if not eligible.any():
            return events

        world = get_world(env)
        contact_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        for obj_name in world.objects.keys():
            if obj_name in ignore_set:
                continue
            try:
                contact = in_contact(world, "gripper", obj_name, env_id=None)  # (N,) bool
                contact_count += contact.int()
            except Exception:
                continue

        multi = (contact_count > 1) & eligible
        if multi.any():
            events.append((
                "Multiple objects grabbed",
                StatusCode.MULTIPLE_OBJECTS_GRABBED,
                multi.clone()
            ))
            self._recorded_multiple_grab |= multi
            if verbose:
                envs = multi.nonzero(as_tuple=False).squeeze(-1).tolist()
                print(f"[EventTracker] envs {envs}: Multiple objects grabbed")

        return events
