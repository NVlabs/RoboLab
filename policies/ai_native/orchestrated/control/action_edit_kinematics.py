# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Adapted from GPT-as-Policy (https://github.com/anonymous-report-421/GPT-as-Policy),
# hybrid_rollout/robolab/robolab_server/action_edit_kinematics.py.

"""Robot-only FK and bounded edits; never reads objects or advances physics."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))


def transform(position, quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, float)
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = position
    return t


def pose(t):
    q = Rotation.from_matrix(t[:3, :3]).as_quat()
    return dict(position=t[:3, 3].tolist(), quaternion_wxyz=q[[3, 0, 1, 2]].tolist())


class RobotFK:
    def __init__(self, chain):
        self.chain = chain
        names = [j['name'] for j in chain if j['kind'] == 'revolute']
        if set(names) != set(JOINTS) or len(names) != 7:
            raise ValueError(f'Expected exactly seven Panda joints, got {names}')

    @classmethod
    def from_stage(cls, stage, robot_path='/World/envs/env_0/robot'):
        from pxr import Usd, UsdPhysics
        root = stage.GetPrimAtPath(robot_path)
        edges = {}
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            b0, b1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(b0) != 1 or len(b1) != 1:
                continue
            if not all(str(b).startswith(robot_path + '/') for b in (b0[0], b1[0])):
                continue
            def local(index):
                q = getattr(joint, f'GetLocalRot{index}Attr')().Get()
                return transform(list(getattr(joint, f'GetLocalPos{index}Attr')().Get()),
                                 [q.GetReal(), *q.GetImaginary()]).tolist()
            if prim.IsA(UsdPhysics.RevoluteJoint):
                axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
                kind = 'revolute'
            elif prim.IsA(UsdPhysics.FixedJoint):
                axis, kind = 'X', 'fixed'
            else:
                continue
            rec = dict(name=prim.GetName(), kind=kind, axis=axis, local0=local(0), local1=local(1))
            edges.setdefault(str(b0[0]), []).append((str(b1[0]), dict(rec, reverse=False)))
            edges.setdefault(str(b1[0]), []).append((str(b0[0]), dict(rec, reverse=True)))
        start = robot_path + '/panda_link0'
        target = robot_path + '/Gripper/Robotiq_2F_85/base_link'
        queue, seen = [(start, [])], {start}
        for node, chain in queue:
            if node == target:
                return cls(chain)
            for nxt, edge in edges.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, chain + [edge]))
        raise ValueError('No robot-only joint chain from Panda root to Robotiq base_link')

    def matrix(self, q):
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('FK requires seven finite joint positions')
        positions, t = dict(zip(JOINTS, q)), np.eye(4)
        for j in self.chain:
            motion = np.eye(4)
            if j['kind'] == 'revolute':
                v = np.zeros(3); v['XYZ'.index(j['axis'])] = positions[j['name']]
                motion[:3, :3] = Rotation.from_rotvec(v).as_matrix()
            edge = np.asarray(j['local0']) @ motion @ np.linalg.inv(j['local1'])
            t = t @ (np.linalg.inv(edge) if j['reverse'] else edge)
        return t

    def trajectory(self, actions):
        a = np.asarray(actions)
        if a.ndim != 2 or a.shape[1] != 8 or not 1 <= len(a) <= 32 or not np.isfinite(a).all():
            raise ValueError('Expected finite H1..32 x 8 action proposal')
        return [dict(index=i, **pose(self.matrix(row[:7])), gripper_closed=bool(row[7] > .5))
                for i, row in enumerate(a)]


def edited_targets(trajectory, steps, edit):
    """Smooth root-frame offset relative to the selected student prefix.

    Gripper overrides apply explicitly from the first step. Closure after an
    approach must be issued in a later decision using a fresh observation.
    """
    if type(steps) is not int or not 1 <= steps <= min(5, len(trajectory)):
        raise ValueError('Edit requires 1..5 steps within the available proposal')
    if not isinstance(edit, dict) or set(edit) != {'delta_position', 'delta_rotation_vector', 'gripper'}:
        raise ValueError('Edit requires delta_position, delta_rotation_vector, gripper')
    dp, dr = np.asarray(edit['delta_position'], float), np.asarray(edit['delta_rotation_vector'], float)
    if dp.shape != (3,) or dr.shape != (3,) or not np.isfinite([dp, dr]).all():
        raise ValueError('Finite 3D edit vectors required')
    if np.linalg.norm(dp) > .05 + 1e-10 or np.linalg.norm(dr) > .35 + 1e-10:
        raise ValueError('Edit exceeds 5 cm or 0.35 rad short-correction limit')
    if edit['gripper'] not in ('keep', 'open', 'closed'):
        raise ValueError('Unknown gripper override')
    targets = []
    for i, original in enumerate(trajectory[:steps]):
        t = transform(original['position'], original['quaternion_wxyz'])
        alpha = (i + 1) / steps
        t[:3, 3] += alpha * dp
        t[:3, :3] = Rotation.from_rotvec(alpha * dr).as_matrix() @ t[:3, :3]
        grip = original['gripper_closed'] if edit['gripper'] == 'keep' else edit['gripper'] == 'closed'
        targets.append(dict(index=i, **pose(t), gripper_closed=grip))
    return targets
