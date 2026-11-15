# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge one Astra worker to RoboLab's per-environment inference loop.

RoboLab owns reset, stepping, termination, videos, and result export. The
worker blocks while the evaluator executes its approved action chunk, then
receives fresh observations before requesting the next Cosmos proposal.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .agent.run import CodexPolicy
from .control.action_edit_kinematics import RobotFK
from .control.eef_controller import build_eef_proposal
from .cosmos.observation import pack_cosmos3_observation
from .cosmos.student import Cosmos3Student
from .cosmos3_rollout import (
    ASTRA_EFFORT,
    ASTRA_MODEL,
    Cosmos3RolloutTools,
    handlers,
    prepare_prompt_source,
    tool_specs,
    workspace_factory,
)
from .io import write_json


def resize_with_pad(image, height, width):
    """Load RoboLab's image helper only inside a RoboLab runtime."""
    from robolab.core.utils.image_utils import resize_with_pad as resize
    return resize(image, height, width)


@dataclass
class PendingChunk:
    actions: np.ndarray
    index: int = 0
    trace: list[dict] = field(default_factory=list)
    result: dict | None = None


@dataclass
class EpisodeState:
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tick: int = 0
    source: str = "student"
    pending: PendingChunk | None = None
    stopped: bool = False
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    stop_requested: bool = False
    stop_reason: str | None = None
    error: str | None = None
    worker: threading.Thread | None = None


class LocalRolloutRPC:
    """In-process robot services for the episode worker."""

    def __init__(self, owner: "EpisodeBridge"):
        self.owner = owner
        self.closed = False

    def request(self, op, **kwargs):
        if self.closed:
            raise RuntimeError("Rollout RPC is closed")
        return self.owner.request(op, **kwargs)

    def close(self):
        self.closed = True


class EpisodeBridge:
    """Duck-typed RoboLab inference client for one environment and episode.

    The native runner calls :meth:`infer` before each physical step.
    The controller resumes when its approved prefix is exhausted; reasoning
    does not advance simulation.
    """

    def __init__(self, env, env_cfg, output, codex, policy_uri, *, token=None,
                 robolab_revision, max_decisions=3000, timeout=120,
                 student_action_horizon=15,
                 require_full_student_chunk=False, seed_base=2040,
                 transport_factory=None,
                 model=ASTRA_MODEL, effort=ASTRA_EFFORT, provider="openai"):
        self.model, self.effort, self.provider = model, effort, provider
        self.transport_factory = transport_factory
        self.instruction = None
        self.env = env
        self.task = getattr(env_cfg, "task_name", getattr(env_cfg, "_task_name", ""))
        if not self.task:
            raise ValueError("RoboLab environment config has no task identity")
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.codex = codex
        self.policy_uri = policy_uri
        self.token = token
        self.robolab_revision = robolab_revision
        self.max_decisions = max_decisions
        self.timeout = timeout
        if type(seed_base) is not int or seed_base < 0:
            raise ValueError("seed_base must be a nonnegative integer")
        self.seed_base = seed_base
        if int(env.num_envs) != 1:
            raise ValueError('Cosmos+Astra supports one environment per job')
        if type(student_action_horizon) is not int or not 1 <= student_action_horizon <= 32:
            raise ValueError("student_action_horizon must be an integer in 1..32")
        if require_full_student_chunk and student_action_horizon != 32:
            raise ValueError("full-chunk Cosmos acceptance currently requires horizon 32")
        self.student_action_horizon = student_action_horizon
        self.require_full_student_chunk = bool(require_full_student_chunk)
        self.resources = []
        self.condition = threading.Condition()
        self.state = EpisodeState()
        self.raw_obs = None
        self.last_action: np.ndarray | None = None
        self.started = False
        self.closed = False
        self.run_index = 0
        self.prompt_source = prepare_prompt_source(self.output,
            student_action_horizon=self.student_action_horizon,
            require_full_student_chunk=self.require_full_student_chunk)
        self.fk = None

    def begin_episode(self, episode_idx: int) -> None:
        if self.started:
            raise RuntimeError("Cosmos+Astra client supports one native run per process")
        self.run_index = int(episode_idx)

    def _start_worker(self) -> None:
        if self.started:
            return
        self.started = True
        self.state.worker = threading.Thread(
            target=self._worker, name="cosmos-astra-episode", daemon=True)
        self.state.worker.start()

    def _own_resource(self, resource):
        with self.condition:
            if self.closed:
                resource.close()
                raise RuntimeError("Orchestration closed during worker startup")
            self.resources.append(resource)
        return resource

    def _cosmos_transport(self, *args, **kwargs):
        from .cosmos.transport import MsgPackWebSocketTransport
        return self._own_resource(MsgPackWebSocketTransport(*args, **kwargs))

    def _codex_transport(self, *args, **kwargs):
        from .agent.transport import StdioAppServer
        factory = self.transport_factory or StdioAppServer
        return self._own_resource(factory(*args, **kwargs))

    def _worker(self) -> None:
        root = self.output / "episode"
        student = worker = rollout = None
        try:
            root.mkdir(parents=True, exist_ok=False)
            (root / "controller").mkdir()
            student = Cosmos3Student(self.policy_uri, self.token,
                robolab_revision=self.robolab_revision,
                action_horizon=self.student_action_horizon,
                transport_cls=self._cosmos_transport)
            worker = CodexPolicy(root / "codex_workspace", self.codex,
                timeout=self.timeout, method="cosmos3_plus_gpt", provider=self.provider,
                skill_root=self.prompt_source,
                specs=tool_specs(student_action_horizon=self.student_action_horizon),
                workspace_factory=workspace_factory, tool_handlers=handlers,
                proposal_tool="policy_infer", model=self.model, effort=self.effort,
                bounded_memory=True,
                transport_factory=self._codex_transport)
            def rpc_factory(*_args, **_kwargs):
                return LocalRolloutRPC(self)
            rollout = Cosmos3RolloutTools(root / "controller", self.task,
                student, seed=self.seed_base + self.run_index,
                max_decisions=self.max_decisions,
                prompt_sha256=worker.prompt_sha256, teacher_model=self.model, teacher_effort=self.effort,
                student_action_horizon=self.student_action_horizon,
                require_full_student_chunk=self.require_full_student_chunk,
                rpc_factory=rpc_factory)
            worker.run(rollout)
        except BaseException:
            error = traceback.format_exc()
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / "failure.json", dict(error=error))
            with self.condition:
                self.state.error = error
                self.condition.notify_all()
        finally:
            if rollout is not None:
                try:
                    rollout.close()
                except Exception:
                    pass
            if student is not None:
                try:
                    student.close()
                except Exception:
                    pass
            if worker is not None:
                try:
                    worker.close()
                except Exception:
                    pass

    @staticmethod
    def _array(value):
        value = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        return np.asarray(value[0]).copy()

    def _observation(self) -> dict:
        if self.raw_obs is None:
            raise RuntimeError("Native RoboLab observation is not available")
        raw = self.raw_obs
        proprio = raw["proprio_obs"]
        state = np.concatenate((self._array(proprio["arm_joint_pos"]),
                                self._array(proprio["gripper_pos"]).reshape(-1))).astype(np.float32)
        main = resize_with_pad(self._array(
            raw["image_obs"]["over_shoulder_left_camera"])[..., :3], 224, 224)
        wrist = resize_with_pad(self._array(raw["image_obs"]["wrist_cam"])[..., :3], 224, 224)
        cosmos = pack_cosmos3_observation(raw, self.instruction,
            env_id=0, episode_idx=self.run_index)
        return dict(main_images=main, wrist_images=wrist, states=state,
            cosmos_image=cosmos["observation/image"],
            task_description=cosmos["prompt"])

    def _teacher_observation(self) -> dict:
        state = self.state
        obs = self._observation()
        image_obs = self.raw_obs["image_obs"]
        # Encode/thumbnail in the rollout client, preserving the source aspect
        # ratio like Agent-EEF's 640-pixel JPEG observations.
        main_rgb = self._array(image_obs["over_shoulder_left_camera"])[..., :3]
        right_rgb = self._array(image_obs["over_shoulder_right_camera"])[..., :3]
        wrist_rgb = self._array(image_obs["wrist_cam"])[..., :3]
        proprio = self.raw_obs["proprio_obs"]
        return dict(episode_id=state.episode_id, step_id=state.tick,
            remaining_steps=max(0, int(self.env.max_episode_length) - state.tick),
            instruction=obs["task_description"], states=obs["states"],
            main_images=obs["main_images"], wrist_images=obs["wrist_images"],
            main_rgb=main_rgb, right_rgb=right_rgb, wrist_rgb=wrist_rgb,
            cosmos_image=obs["cosmos_image"],
            ee_pos=self._array(proprio["ee_pos"]),
            ee_quat_wxyz=self._array(proprio["ee_quat"]),
            eef_pos=self._array(proprio["eef_pos"]),
            eef_quat_wxyz=self._array(proprio["eef_quat"]),
            frame="robot_root", ee_frame="base_link_in_robot_root",
            eef_frame="eef_frame_in_robot_root", position_units="meters",
            teacher_inputs="RGB_proprio_no_native_success_criteria_no_object_poses")

    def _metadata(self) -> dict:
        return dict(task=self.task, instruction=self.instruction,
            max_episode_steps=int(self.env.max_episode_length), control_dt=float(self.env.step_dt),
            action_dim=8, action_semantics="absolute_arm_joint_positions_7_and_binary_gripper_gt_0.5",
            policy_identity_source="remote_cosmos3_owner_declared_model_endpoint_metadata_unverified",
            num_envs=1)

    def _ensure_fk(self):
        if self.fk is None:
            import omni.usd
            self.fk = RobotFK.from_stage(omni.usd.get_context().get_stage())
        return self.fk

    def _fk_preview(self, actions) -> dict:
        actions = np.asarray(actions, np.float32)
        fk = self._ensure_fk()
        current = self._teacher_observation()
        predicted = fk.matrix(current["states"][:7])
        from .control.action_edit_kinematics import transform
        measured = transform(current["ee_pos"], current["ee_quat_wxyz"])
        from scipy.spatial.transform import Rotation
        distance = float(np.linalg.norm(predicted[:3, 3] - measured[:3, 3]))
        angle = float(Rotation.from_matrix(predicted[:3, :3] @ measured[:3, :3].T).magnitude())
        check = dict(step_id=self.state.tick, position_error_m=distance,
            rotation_error_rad=angle, physical_steps=0,
            passed=distance < .002 and angle < .01)
        if not check["passed"]:
            raise ValueError(f"FK does not match native measured flange for the episode: {check}")
        return dict(episode_id=self.state.episode_id,
            step_id=self.state.tick, physical_steps=0,
            frame="base_link_in_robot_root", units="meters_and_wxyz",
            measured_fk_check=check, trajectory=fk.trajectory(actions),
            interpretation="kinematic_target_preview_not_physics_or_object_future")

    def _record_decision(self, arguments: dict) -> dict:
        state = self.state
        directory = self.output / "episode" / "sim" / "codex_decisions"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{int(arguments["decision"]):03d}.json'
        record = dict(schema="hybrid_rollout.codex_decision.v1", env_id=0,
            episode_id=state.episode_id, step_id=state.tick,
            physical_steps=0, **arguments)
        write_json(path, record)
        return dict(record_path=str(path), physical_steps=0)

    def _mark_stopped(self) -> None:
        state = self.state
        if state.stopped:
            return
        already_frozen = bool(self.env._frozen_envs[0].item())
        if not already_frozen:
            # Controller stops bypass RoboLab's native _reset_idx termination
            # path, so finalize this environment's pending recorder demo here,
            # on the simulator thread, before freezing it.
            # Use the named native success term, not generic termination, for
            # both JSON/RPC and HDF5. Other termination reasons are not success.
            state.success = self._native_success()
            state.terminated = bool(state.success)
            self.env._env_results[0] = bool(state.success)
            self.env._env_term_step.setdefault(0, state.tick)
            recorder = getattr(self.env, "recorder_manager", None)
            if recorder is not None:
                try:
                    import torch
                    success = torch.tensor([state.success], dtype=torch.bool,
                                           device=self.env.device)
                    recorder.set_success_to_episodes([0], success)
                    recorder.export_episodes(env_ids=[0])
                except Exception:
                    state.error = traceback.format_exc()
                    self.condition.notify_all()
                    raise
            self.env._frozen_envs[0] = True
        state.stopped = True
        self.condition.notify_all()

    def request(self, op: str, **arguments):
        state = self.state
        with self.condition:
            if state.error:
                raise RuntimeError(state.error)
            if op not in ("metadata", "reset"):
                if (arguments.pop("episode_id", None) != state.episode_id
                        or arguments.pop("step_id", None) != state.tick):
                    raise ValueError("Request identity differs from the episode")
            if op == "metadata":
                return self._metadata()
            if op == "reset":
                return dict(obs=self._observation(), episode_id=state.episode_id,
                    step_id=state.tick, initial_state_hash=None,
                    metadata=dict(self._metadata(), env_id=0,
                                  reset_mode="native_reset"))
            if op == "teacher_observation":
                return self._teacher_observation()
            if op == "begin_combination":
                return dict(episode_id=state.episode_id, step_id=state.tick,
                            source=state.source, changed=True)
            if op == "switch_control_source":
                state.source = arguments["source"]
                return dict(episode_id=state.episode_id, step_id=state.tick,
                            source=state.source, changed=True)
            if op == "fk_preview":
                return self._fk_preview(arguments["actions"])
            if op == "record_codex_decision":
                return self._record_decision(arguments)
            if op == "eef_joint_target":
                proposal = build_eef_proposal(self.env, arguments.get("position"),
                    arguments.get("quaternion_wxyz"), arguments.get("gripper_closed"),
                    env_id=0, damping=arguments.get("damping", .05),
                    max_joint_delta_rad=arguments.get("max_joint_delta_rad", .05),
                    max_translation_step_m=arguments.get("max_translation_step_m", .02),
                    max_rotation_step_rad=arguments.get("max_rotation_step_rad", .1),
                    joint_limit_margin_rad=arguments.get("joint_limit_margin_rad", 0.0))
                return dict(proposal, episode_id=state.episode_id, step_id=state.tick)
            if op == "finish_pilot":
                # The Astra worker must not touch the simulator recorder. Queue
                # finalization for infer(), which runs on the simulator
                # thread, and wait until that work is complete.
                state.stop_requested = True
                state.stop_reason = arguments["reason"]
                self.condition.notify_all()
                while not state.stopped and state.error is None and not self.closed:
                    self.condition.wait()
                if state.error:
                    raise RuntimeError(state.error)
                if not state.stopped:
                    raise RuntimeError("Rollout closed before stop finalization")
                return dict(episode_id=state.episode_id, step_id=state.tick,
                    success=state.success, terminated=state.terminated,
                    truncated=state.truncated, video_frames=state.tick,
                    control_dt=float(self.env.step_dt), reason=state.stop_reason)
            if op != "chunk_step":
                raise ValueError(f"Unsupported rollout operation: {op}")
            actions = np.asarray(arguments["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 8 or not 1 <= len(actions) <= 64:
                raise ValueError("Actions must be finite [H,8], 1 <= H <= 64")
            if not np.isfinite(actions).all() or state.pending is not None or state.stopped:
                raise ValueError("Invalid, overlapping, or stopped action chunk")
            actions = actions.copy()
            actions[:, 7] = (actions[:, 7] > .5).astype(np.float32)
            pending = PendingChunk(actions)
            state.pending = pending
            self.condition.notify_all()
            while pending.result is None and state.error is None and not self.closed:
                self.condition.wait()
            if state.error:
                raise RuntimeError(state.error)
            if pending.result is None:
                raise RuntimeError("Rollout closed before action acknowledgement")
            return pending.result

    def _native_success(self) -> bool:
        try:
            return bool(self.env.termination_manager.get_term("success")[0].item())
        except Exception:
            return bool(self.env._env_results.get(0, False))

    def _complete_previous_step(self, obs) -> None:
        self.raw_obs = obs
        if self.last_action is None:
            return
        state = self.state
        pending = state.pending
        if pending is None:
            raise RuntimeError("Episode lost its pending action acknowledgement")
        state.tick += 1
        frozen = bool(self.env._frozen_envs[0].item())
        state.success = self._native_success() if frozen else False
        state.terminated = frozen and state.success
        state.truncated = frozen and not state.success
        pending.trace.append(dict(obs=self._observation(), reward=0.0,
            terminated=state.terminated, truncated=state.truncated,
            valid=True, success=state.success, episode_id=state.episode_id,
            step_id=state.tick, executed_action=self.last_action.copy(),
            action_source=state.source))
        pending.index += 1
        if frozen or pending.index >= len(pending.actions):
            pending.result = dict(step_id=state.tick, steps=pending.trace)
            state.pending = None
        self.last_action = None
        self.condition.notify_all()

    def _raise_worker_error(self):
        if self.state.error:
            raise RuntimeError("Cosmos+Astra worker failed:\n" + self.state.error)

    def infer(self, obs) -> dict:
        with self.condition:
            self._complete_previous_step(obs)
            self._start_worker()
            state = self.state
            while True:
                self._raise_worker_error()
                if self.closed:
                    raise RuntimeError("Orchestration closed while waiting for an action")
                if state.stop_requested or state.stopped:
                    raise RuntimeError("Controller stopped before native episode termination")
                if state.pending is not None:
                    break
                # Codex refreshes its inactivity deadline as tool calls progress.
                self.condition.wait(self.timeout)
            action = state.pending.actions[state.pending.index].copy()
            self.last_action = action
            return dict(action=action, viz=None)

    def reset(self) -> None:
        with self.condition:
            # The evaluator calls reset after the terminal step without another
            # infer call. Acknowledge that action without advancing physics.
            if self.last_action is not None:
                try:
                    obs = self.env.observation_manager.compute()
                    self._complete_previous_step(obs)
                except Exception as error:
                    self.state.error = traceback.format_exc()
                    self.last_action = None
                    self.condition.notify_all()
                    raise RuntimeError("Failed to acknowledge the final action") from error
            self._mark_stopped()
            self.condition.notify_all()
        if self.state.worker is not None:
            self.state.worker.join(timeout=30)

    def check_worker(self, timeout: float = 120) -> None:
        """Reject an episode whose controller did not finish cleanly."""
        with self.condition:
            self._raise_worker_error()
        worker = self.state.worker
        if worker is not None:
            worker.join(timeout=timeout)
        with self.condition:
            self._raise_worker_error()
        if worker is not None and worker.is_alive():
            raise RuntimeError("Cosmos+Astra worker did not exit")

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
            resources = list(self.resources)
        # Closing I/O wakes a worker even when it is not waiting on our condition.
        for resource in reversed(resources):
            try:
                resource.close()
            except Exception:
                pass
        if self.state.worker is not None:
            self.state.worker.join(timeout=35)
            if self.state.worker.is_alive():
                raise RuntimeError("Orchestration worker did not exit after cancellation")
