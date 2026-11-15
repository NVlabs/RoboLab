# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Adapted from GPT-as-Policy (https://github.com/anonymous-report-421/GPT-as-Policy),
# hybrid_rollout/robolab/robolab_server/validation.py.

"""Validate action bounds, observation identity, and intervention assessments."""
import numpy as np
from scipy.spatial.transform import Rotation

from .action_edit_kinematics import edited_targets
from .gate_assessment import validate_assessment


def angles(p):
    q = np.asarray(p['quaternion_wxyz'])
    return dict(p, rpy_deg=Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler('xyz', degrees=True).tolist())


def validate_response(response, request):
    if response.get('request_id') != request['request_id']:
        raise ValueError('Response is stale or belongs to another observation')
    if not isinstance(response.get('reason'), str) or not response['reason'].strip():
        raise ValueError('A visible observation/decision explanation is required')
    mode = response.get('mode')
    if mode not in ('student', 'edit', 'eef'):
        raise ValueError('Expected student/edit/eef; only native termination ends an episode')
    if request.get('gate_policy') == 'failure-or-intent':
        validate_assessment(response, request)
    n = response.get('steps')
    if mode == 'student':
        cap = request.get('student_action_horizon', 15)
        if type(cap) is not int or not 1 <= cap <= 32:
            raise ValueError('Invalid student action horizon')
        if request.get('require_full_student_chunk'):
            if type(n) is not int or n != cap:
                raise ValueError(f'Full-chunk acceptance requires exactly {cap} student steps')
        elif type(n) is not int or not 1 <= n <= cap:
            raise ValueError(f'Expected 1..{cap} execution steps')
    elif type(n) is not int or not 1 <= n <= 5:
        raise ValueError('Expected 1..5 execution steps')
    if mode == 'edit':
        edited_targets(request['student_eef_trajectory'], n, response['edit'])
    elif mode == 'eef':
        target = response['target']
        p, q = np.asarray(target['position'], float), np.asarray(target['quaternion_wxyz'], float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError('Finite EEF position/quaternion required')
        if abs(np.linalg.norm(q)-1) > 1e-4 or type(target['gripper_closed']) is not bool:
            raise ValueError('Unit quaternion and explicit boolean gripper required')
        if np.linalg.norm(p-np.asarray(request['current_eef']['position'])) > .05+1e-9:
            raise ValueError('Recovery EEF target exceeds 5 cm')
        old = request['current_eef']['quaternion_wxyz']
        delta = Rotation.from_quat(q[[1, 2, 3, 0]]) * Rotation.from_quat(np.asarray(old)[[1, 2, 3, 0]]).inv()
        if delta.magnitude() > .35+1e-9:
            raise ValueError('Recovery EEF target exceeds 0.35 rad')
