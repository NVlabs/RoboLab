# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos-specific identity and proposal boundary over the public gate harness."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .agent.schema import _obj, response_schema
from .agent.workspace import prepare_workspace
from .control.client import RolloutTools
from .io import write_json

PUBLIC_SKILL = Path(__file__).resolve().parent / "agent"
from .settings import EFFORT as ASTRA_EFFORT
from .settings import MODEL as ASTRA_MODEL


def prepare_prompt_source(output: Path, *, student_action_horizon=15,
                          require_full_student_chunk=False) -> Path:
    """Render the policy instructions for this run's action horizon."""
    if type(student_action_horizon) is not int or not 1 <= student_action_horizon <= 32:
        raise ValueError("student_action_horizon must be an integer in 1..32")
    if require_full_student_chunk and student_action_horizon != 32:
        raise ValueError("Full-chunk acceptance requires horizon 32")
    target = output / "prompt_source"
    target.mkdir()
    (target / "context").mkdir()
    student_rule = (
        "`student`: accept and execute the entire fresh 32-step proposal; set `steps` to 32. Partial student prefixes are not allowed."
        if require_full_student_chunk else
        f"`student`: execute a prefix of 1–{student_action_horizon} steps from this fresh proposal."
    )
    hashes = {}
    for relative in ("SKILL.md", "gate_prompt.md", "context/teacher_context.md",
                     "context/eef_control.md", "context/cosmos_geometry.md"):
        source = PUBLIC_SKILL / relative
        text = source.read_text().replace("{{student_action_horizon}}", str(student_action_horizon))
        text = text.replace("{{student_rule}}", student_rule)
        eef_rule = ("`student`: accept all 32 rows of the latest student proposal by setting `steps` to 32. No targets change and partial prefixes are not allowed."
            if require_full_student_chunk else f"`student`: select the first 1–{student_action_horizon} rows of the latest student proposal. No\n  targets change; the unused suffix is discarded.")
        text = text.replace("{{student_eef_rule}}", eef_rule)
        destination = target / relative
        destination.write_text(text)
        hashes[relative] = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                               rendered_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    write_json(target / "manifest.json", dict(
        student_action_horizon=student_action_horizon,
        require_full_student_chunk=require_full_student_chunk, files=hashes))
    return target


def workspace_factory(audit, skill_root):
    return prepare_workspace(audit, skill_root)


def tool_specs(*, student_action_horizon=15):
    string = {"type": "string"}
    return [
        dict(type="function", name="robolab_start",
            description="Start one RoboLab episode and return RGB/proprio.",
            inputSchema=_obj(dict(task=string, output_dir=string))),
        dict(type="function", name="policy_infer",
            description="Infer one fresh Cosmos student-policy proposal from the latest observation.",
            inputSchema=_obj(dict(observation_path=string, output_dir=string))),
        dict(type="function", name="robolab_execute",
            description="Validate and execute a reviewed proposal prefix or bounded EEF correction.",
            inputSchema=_obj(dict(proposal_path=string,
                response=response_schema(student_step_max=student_action_horizon),
                output_dir=string))),
    ]


def handlers(rollout):
    return dict(robolab_start=rollout.start, policy_infer=rollout.infer,
                robolab_execute=rollout.execute)


class Cosmos3RolloutTools(RolloutTools):
    student_tool = "policy_infer"
    camera_keys = ('main_rgb', 'right_rgb', 'wrist_rgb')
    image_size = 640
    image_format = 'JPEG'

    def __init__(self, *args, teacher_effort=ASTRA_EFFORT, max_decisions=3000, **kwargs):
        super().__init__(*args, teacher_effort=teacher_effort,
                         max_decisions=max_decisions, **kwargs)

    def drain_decision_budget(self):
        """No more model/proposal requests; measured holds until native termination."""
        if self.phase != 'infer' or not self.decision_budget_exhausted():
            raise RuntimeError('Decision-budget holds require an exhausted action budget')
        if self.source != 'gpt_eef':
            self._rpc('switch_control_source', source='gpt_eef', reason='decision_budget_hold')
            self.source = 'gpt_eef'
        start_tick = self.tick
        trace = []
        while True:
            observation = self._rpc('teacher_observation')
            action = np.asarray(observation['states'], np.float32).copy()
            action[7] = float(action[7] > .5)
            result = self._rpc('chunk_step', actions=action[None, :])
            valid = [row for row in result['steps'] if row['valid']]
            if not valid or result['step_id'] <= self.tick:
                raise RuntimeError('Native simulation did not advance a budget hold')
            trace.extend(dict(executed_action=np.asarray(row["executed_action"]).copy(),
                              states=np.asarray(row["obs"]["states"]).copy())
                         for row in valid)
            self.tick = result['step_id']
            if valid[-1]['terminated'] or valid[-1]['truncated']:
                break
        self.counters['budget_hold_steps'] = len(trace)
        record = dict(decision=len(self.history), start_tick=start_tick, end_tick=self.tick,
            response=dict(mode='hold', steps=len(trace), reason='decision_budget'),
            executed_steps=len(trace), source=self.source, source_detail='decision_budget_hold',
            terminal=True, native_success=bool(valid[-1]['success']))
        write_json(self.output/'decision_budget_hold.json', record)
        np.savez_compressed(self.output/'decision_budget_hold.npz',
            actions=np.asarray([row['executed_action'] for row in trace], np.float32),
            states=np.asarray([row['states'] for row in trace], np.float32))
        return self.finish('terminal')

    def student_spec(self):
        identity = self.student.metadata["identity_sha256"]
        version = f'nvidia/Cosmos3-Nano-Policy-DROID/{identity[:16]}'
        return version, identity, dict(evaluation_method="cosmos3_plus_gpt",
            prompt_geometry='gripper_camera', action_execution='blocking_tool_reply',
            config="Cosmos3-Nano-Policy-DROID", checkpoint=None,
            student_backend="remote_cosmos3_msgpack_websocket", student_identity_sha256=identity,
            student_server_metadata=self.student.metadata)

    def record_student_identity(self):
        write_json(self.output / "student_policy_identity.json", self.student.metadata)

    def hold_after_invalid_action(self, *, error, **arguments):
        """Consume one native hold step after three invalid action attempts.

        No malformed target is executed. Discard the stale Cosmos proposal and
        return fresh images so the agent can request a new proposal.
        """
        if self.phase != 'execute':
            raise RuntimeError('Invalid-action recovery requires a current proposal')
        decision = len(self.history)
        directory = Path(self.next_call()['output_dir'])
        held_target = dict(self.observation_packet['current_eef'],
                           gripper_closed=bool(self.obs['states'][7] >= .5))
        if self.source != 'gpt_eef':
            self._rpc('switch_control_source', source='gpt_eef', reason='invalid_action_hold')
            self.source = 'gpt_eef'
        action = np.asarray(self.obs['states'], dtype=np.float32).copy()
        action[7] = float(action[7] >= .5)
        start_tick = self.tick
        result = self._rpc('chunk_step', actions=action[None, :])
        valid = [row for row in result['steps'] if row['valid']]
        self.tick = result['step_id']
        done = bool(valid and (valid[-1]['terminated'] or valid[-1]['truncated']))
        self.counters['recovery_steps'] += len(valid)
        record = dict(decision=decision, start_tick=start_tick, end_tick=self.tick,
            response=dict(mode='hold', steps=1, reason=error), invalid_action=True,
            executed_steps=len(valid), source=self.source, source_detail='invalid_action_hold',
            prediction_id=self.prediction['prediction_id'], selected_student_steps=0,
            discarded_student_steps=len(self.actions), commanded_eef_target=held_target,
            executed_gripper_closed=[bool(row['executed_action'][7] > .5) for row in valid],
            terminal=done, native_success=bool(valid and valid[-1]['success']))
        write_json(self.output/f'invalid_action_hold_{decision:03d}.json', record)
        np.savez_compressed(self.output/f'execution_{decision:03d}.npz',
            actions=np.asarray([row['executed_action'] for row in valid], np.float32),
            states=np.asarray([row['obs']['states'] for row in valid], np.float32))
        self.history.append(record)
        self.preceding = self.request
        write_json(self.output/'history.json', self.history)
        self.phase = 'infer'
        final = self.finish('terminal') if done else None
        packet = self._observation(directory, result=final)
        packet['feedback'] = f'The previous action was rejected: {error}. '
        packet['feedback'] += 'One hold step was executed. Request a fresh proposal and corrected decision.'
        return packet

    def add_observation_feedback(self, packet):
        packet['decision_budget'] = dict(limit=self.max_decisions,
            decisions=self.decisions,
            remaining=max(0, self.max_decisions-self.decisions) if self.max_decisions else None)
        if not self.history or not self.history[-1]['executed_steps']:
            return
        record = self.history[-1]
        response = record['response']
        mode = response['mode']
        if 'commanded_eef_target' in record:
            target = record['commanded_eef_target']
        elif mode == 'eef':
            target = response['target']
        elif mode == 'edit':
            from .control.action_edit_kinematics import edited_targets
            target = edited_targets(self.fk['trajectory'], response['steps'], response['edit'])[
                record['executed_steps'] - 1]
        else:
            # Cosmos emits joint targets: use robot-only FK, not object truth.
            target = self.fk['trajectory'][record['executed_steps'] - 1]
        actual = packet['current_eef']
        record['commanded_eef_target'] = target
        dot = abs(float(np.dot(target['quaternion_wxyz'], actual['quaternion_wxyz'])))
        tracking = dict(
            translation_error_m=float(np.linalg.norm(
                np.asarray(target['position']) - actual['position'])),
            rotation_error_rad=float(2*np.arccos(np.clip(dot, 0, 1))),
            measured_gripper_closed=bool(packet['current_state'][7] >= .5),
            commanded_gripper_closed=bool(target['gripper_closed']))
        packet['post_chunk_tracking'] = tracking
        record['post_chunk_tracking'] = tracking
        write_json(self.output/'history.json', self.history)
