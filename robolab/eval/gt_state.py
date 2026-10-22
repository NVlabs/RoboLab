# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth simulator state export during policy evaluation.

Enabled with ``--enable-gt-state`` on any runner. Each step,
:func:`robolab.eval.episode.run_episode` calls :meth:`GroundTruthStateExporter.export_all`
for the active envs and stores the result on the observation dict as
``obs["gt_state"] = {env_id: state}``. Clients that want privileged state
(e.g. an inference proxy / orchestrator) read their own env's entry via
``InferenceClient._orchestrator_keys``; the policy itself never sees it.

The core exporter emits a *raw snapshot*: direct simulator readouts plus the
subtask recorder's tracking. Derived quantities (lift tracking, grasp
detection) are the consumer's business — the VoLo client computes its own in
``policies/volo/metadata.py`` from this per-step stream.

Frames and units — every pose in this payload uses the same convention:
positions are meters in the **env-local frame** (world position minus the
env origin; orientation of the env-local frame equals the world frame, envs
are translated copies), quaternions are **world-frame (w, x, y, z)**, and
velocities are world-frame (linear m/s, angular rad/s). Exports happen once
per **environment step** (policy rate, after decimated physics substeps),
not per physics substep.

Per-env state schema (all arrays are numpy, msgpack-serialisable):

``objects`` — one entry per manipulable scene object (from
``env_cfg.contact_object_list``, minus fixtures):
    ``pos`` (3,) float32; ``quat`` (4,) float32; ``vel`` (6,) float32
    linear+angular.

``robot`` — ``ee_pos`` (3,) float32; ``ee_quat`` (4,) float32;
``gripper_closedness`` float32 in [0, 1] (0 = open, 1 = closed);
``objects_in_contact`` list[str] — objects currently touching the gripper
per the contact sensor, regardless of gripper closure.

``subtask`` — completion status from the env's
:class:`SubtaskCompletionRecorderTerm`: ``completed`` / ``total`` int,
``score`` float32, ``info`` str, ``conditions`` list of per-object
condition rows, ``object_completed`` {object: bool} across all past+current
subtasks, ``all_subtask_conditions`` {"subtask_i": bool} re-evaluated
against the live sim each step (enables regression detection when a
previously satisfied subtask no longer holds).

``scene_objects`` — list of manipulable object names; ``step`` — steps
since construction/reset.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from robolab.core.utils.isaaclab_compat import as_torch, quat_isaaclab_to_wxyz

logger = logging.getLogger(__name__)


class GroundTruthStateExporter:
    """Export per-env ground-truth sim state for inference-time consumers.

    Construct once per episode (after ``env.reset()``); call
    :meth:`export_all` once per step. Robot-specific names default to the
    DROID configuration and are constructor parameters, not assumptions:

    Args:
        env: The ManagerBasedRLEnv (RobolabEnv) instance.
        env_cfg: Environment config (provides ``contact_object_list``).
        ee_body_name: Robot body whose pose is reported as the end-effector.
            Default ``"base_link"`` is the DROID gripper mount link.
        gripper_joint_name: Articulation joint used to derive gripper
            closedness. Default ``"finger_joint"`` (DROID Robotiq gripper).
        gripper_joint_closed_pos: Joint position (radians) at fully closed;
            closedness is ``joint_pos / gripper_joint_closed_pos``.
        gripper_contact_body: Contact-sensor body name used for the
            gripper-contact query.
    """

    # Objects that are scene fixtures, not manipulable.
    FIXTURE_NAMES = {"table", "robot"}

    def __init__(
        self,
        env: Any,
        env_cfg: Any,
        *,
        ee_body_name: str = "base_link",
        gripper_joint_name: str = "finger_joint",
        gripper_joint_closed_pos: float = math.pi / 4,
        gripper_contact_body: str = "gripper",
    ) -> None:
        self.env = env
        self.env_cfg = env_cfg
        self.ee_body_name = ee_body_name
        self.gripper_joint_name = gripper_joint_name
        self.gripper_joint_closed_pos = gripper_joint_closed_pos
        self.gripper_contact_body = gripper_contact_body
        contact_list: list[str] = getattr(env_cfg, "contact_object_list", []) or []
        self._object_names: list[str] = [n for n in contact_list if n not in self.FIXTURE_NAMES]
        self._step = 0
        self._subtask_recorder = self._find_subtask_recorder()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear per-episode tracking. Call when reusing across episodes."""
        self._step = 0
        self._subtask_recorder = self._find_subtask_recorder()

    def export_all(self, env_ids: list[int]) -> dict[int, dict]:
        """Export state for the given envs. Call once per sim step."""
        self._step += 1
        return {env_id: self._export_env(env_id) for env_id in env_ids}

    # ------------------------------------------------------------------
    # Per-env export
    # ------------------------------------------------------------------

    def _export_env(self, env_id: int) -> dict:
        return {
            "objects": self._export_objects(env_id),
            "robot": self._export_robot(env_id),
            "subtask": self._export_subtask(env_id),
            "scene_objects": self._object_names,
            "step": self._step,
        }

    def _export_objects(self, env_id: int) -> dict:
        """Per-object pose and velocity, read directly from the sim.

        ``pos`` is env-local meters, ``quat`` world-frame (w, x, y, z),
        ``vel`` world-frame linear+angular.
        """
        from robolab.core.world.world_state import get_world

        world = get_world(self.env)
        result = {}
        for name in self._object_names:
            try:
                world.get_body(name)
            except (ValueError, KeyError):
                continue
            pos, quat = world.get_pose(name, is_relative=True, env_id=env_id)
            try:
                vel_np = world.get_velocity(name, env_id=env_id).cpu().numpy().astype(np.float32)
            except (ValueError, KeyError, AttributeError):
                vel_np = np.zeros(6, dtype=np.float32)
            result[name] = {
                "pos": pos.cpu().numpy().astype(np.float32)[:3],
                "quat": quat.cpu().numpy().astype(np.float32),
                "vel": vel_np,
            }
        return result

    def _export_robot(self, env_id: int) -> dict:
        """EE pose (env-local pos, world-frame wxyz quat), gripper state, contacts."""
        from robolab.core.world.world_state import get_world

        world = get_world(self.env)
        robot = world.get_articulation("robot")
        body_idx = robot.data.body_names.index(self.ee_body_name)
        env_origin = self.env.scene.env_origins[env_id].cpu().numpy().astype(np.float32)
        ee_pos = as_torch(robot.data.body_pos_w)[env_id, body_idx, :].cpu().numpy().astype(np.float32) - env_origin
        ee_quat = quat_isaaclab_to_wxyz(as_torch(robot.data.body_quat_w)[env_id, body_idx, :])
        ee_quat = ee_quat.cpu().numpy().astype(np.float32)
        joint_names = robot.data.joint_names
        if self.gripper_joint_name in joint_names:
            joint_idx = joint_names.index(self.gripper_joint_name)
            joint_pos = as_torch(robot.data.joint_pos)[env_id, joint_idx].item()
            gripper_closedness = float(joint_pos / self.gripper_joint_closed_pos)
        else:
            gripper_closedness = 0.0
        return {
            "ee_pos": ee_pos[:3],
            "ee_quat": ee_quat,
            "gripper_closedness": np.float32(gripper_closedness),
            "objects_in_contact": self._gripper_contacts(world, env_id),
        }

    def _gripper_contacts(self, world: Any, env_id: int) -> list[str]:
        """Objects currently in contact with the gripper (contact sensor)."""
        try:
            return world.get_objects_in_contact_with(
                self.gripper_contact_body, self._object_names, env_id=env_id,
            )
        except (ValueError, KeyError, AttributeError) as err:
            logger.debug("Gripper contact query unavailable for env %d: %s", env_id, err)
            return []

    # ------------------------------------------------------------------
    # Subtask state
    # ------------------------------------------------------------------

    def _export_subtask(self, env_id: int) -> dict:
        """Subtask completion status from this env's SubtaskCompletionRecorderTerm."""
        recorder = self._subtask_recorder
        state_machines = getattr(recorder, "subtask_state_machines", None) if recorder else None
        if not state_machines or env_id >= len(state_machines):
            return {"completed": 0, "total": 0, "score": np.float32(0.0), "info": "", "conditions": []}
        info = recorder.infos[env_id]
        sm = state_machines[env_id]
        return {
            "completed": int(info.get("completed", 0)),
            "total": int(info.get("total", 0)),
            "score": np.float32(float(info.get("score", 0.0))),
            "info": str(info.get("info", "")),
            "conditions": self._export_conditions(sm),
            "object_completed": self._export_object_completed(sm, env_id),
            "all_subtask_conditions": self._check_all_subtask_conditions(sm, env_id),
        }

    def _check_all_subtask_conditions(self, sm: Any, env_id: int) -> dict[str, bool]:
        """Evaluate ALL subtask conditions against the live sim each step.

        Returns e.g. ``{"subtask_0": True, "subtask_1": False}`` where each
        entry reflects whether that subtask's conditions hold *right now*,
        honoring the subtask's ``logical`` mode ("all" / "any" / "choose").
        A consumer that considers subtask 0 done can detect regression (e.g.
        a stacked block fell) when its entry flips back to False.
        """
        result = {}
        for i, subtask in enumerate(sm.subtasks):
            key = f"subtask_{i}"
            try:
                group_results = [
                    all(cond_func(env=self.env, env_id=env_id) for cond_func, _score in cond_list)
                    for cond_list in subtask.conditions.values()
                ]
                logical = getattr(subtask, "logical", "all")
                n_satisfied = sum(group_results)
                if logical == "any":
                    result[key] = n_satisfied >= 1
                elif logical == "choose":
                    result[key] = n_satisfied >= (getattr(subtask, "K", None) or 1)
                else:
                    result[key] = all(group_results)
            except Exception as err:
                logger.debug("GT subtask condition check failed (env %d, subtask %d): %s", env_id, i, err)
                result[key] = False
        return result

    def _export_object_completed(self, sm: Any, env_id: int) -> dict[str, bool]:
        """Per-object completion across the current and all past subtasks.

        The current subtask uses robolab's tracked CSM state (authoritative);
        past subtasks re-evaluate their *terminal* condition against the sim,
        since condition lists are sequential steps (grabbed → lifted →
        placed) and earlier transients are no longer true once the object is
        placed. Future subtasks are skipped: under the sequential state
        machine they cannot be legitimately complete yet, and the initial
        scene can spuriously satisfy them. ORed across condition groups so
        an object involved in several groups counts as done if any group is.
        Lets a consumer whose subgoal order differs from robolab's internal
        ordering still match per-object completion.
        """
        result: dict[str, bool] = {}
        csm = sm.conditionals_state_machine
        cur_idx = sm.current_subtask_index
        if csm is not None:
            try:
                cur_subtask = sm.subtasks[cur_idx] if cur_idx < len(sm.subtasks) else None
                for group_name in csm.subtask.group_names:
                    done = csm.check_object_completed(group_name)
                    cond_list = cur_subtask.conditions.get(group_name, []) if cur_subtask else []
                    for key in self._group_object_keys(group_name, cond_list, fallback=group_name):
                        result[key] = result.get(key, False) or done
            except (AttributeError, TypeError) as err:
                logger.debug("GT current-subtask completion failed (env %d): %s", env_id, err)
        for i, subtask in enumerate(sm.subtasks):
            if i >= cur_idx:
                continue
            try:
                for group_name, cond_list in subtask.conditions.items():
                    if not cond_list:
                        continue
                    terminal_cond, _score = cond_list[-1]
                    done = terminal_cond(env=self.env, env_id=env_id)
                    keys = self._group_object_keys(group_name, cond_list, fallback=f"subtask_{i}:{group_name}")
                    for key in keys:
                        result[key] = result.get(key, False) or done
            except Exception as err:
                logger.debug("GT cross-subtask completion failed (env %d, subtask %d): %s", env_id, i, err)
        return result

    def _export_conditions(self, sm: Any) -> list[dict]:
        """Per-object condition satisfaction rows for the current subtask."""
        csm = sm.conditionals_state_machine
        if csm is None:
            return []
        conditions = []
        try:
            cur_idx = sm.current_subtask_index
            cur_subtask = sm.subtasks[cur_idx] if cur_idx < len(sm.subtasks) else None
            for group_name, completed_list in getattr(csm, "object_completed_table", {}).items():
                cond_list = cur_subtask.conditions.get(group_name, []) if cur_subtask else []
                for key in self._group_object_keys(group_name, cond_list, fallback=group_name):
                    for i, satisfied in enumerate(completed_list):
                        conditions.append({
                            "object": str(key),
                            "condition_idx": i,
                            "info": "",
                            "satisfied": bool(satisfied),
                        })
        except (AttributeError, TypeError) as err:
            logger.debug("GT condition export failed: %s", err)
        return conditions

    def _group_object_keys(self, group_name: str, cond_list: list, *, fallback: str) -> list[str]:
        """Resolve a CSM condition-group name to real scene-object names.

        For pick-and-place subtasks the group name IS the object name. For
        atomic compound predicates (e.g. ``partial(stacked, objects=[...])``)
        the group name is a generic placeholder like ``"conditions"`` or a
        descriptive label like ``"red_blue"``; recover the actual objects
        from the terminal condition's partial keywords so consumers see
        consistent per-object names regardless of subtask authoring style.
        """
        if group_name in self._object_names:
            return [group_name]
        if cond_list:
            terminal_cond, _score = cond_list[-1]
            objs = getattr(terminal_cond, "keywords", {}).get("objects") or \
                getattr(terminal_cond, "keywords", {}).get("object")
            if isinstance(objs, list):
                return [str(o) for o in objs]
            if isinstance(objs, str):
                return [objs]
        return [fallback]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_subtask_recorder(self):
        """Locate the SubtaskCompletionRecorderTerm on the env's recorder manager."""
        from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm

        rm = getattr(self.env, "recorder_manager", None)
        if rm is None or not hasattr(rm, "get_term"):
            logger.warning("GT-state export: no recorder manager; subtask state will be empty.")
            return None
        term = rm.get_term(SubtaskCompletionRecorderTerm)
        if term is None:
            logger.warning("GT-state export: SubtaskCompletionRecorderTerm not found; subtask state will be empty.")
        return term
