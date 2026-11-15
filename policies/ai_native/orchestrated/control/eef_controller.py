# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Adapted from GPT-as-Policy (https://github.com/anonymous-report-421/GPT-as-Policy),
# hybrid_rollout/robolab/robolab_server/eef_controller.py.

"""One bounded DLS joint-target proposal for RoboLab DROID's base_link.

Commands are absolute base_link poses in the robot-root frame, quaternion wxyz.
This is NOT DroidCfg's rotated eef_frame. The module reads robot kinematics only,
does not advance physics or replace the action controller, and has no action
queue. Call again after each real control step while tracking a fixed target.

The generic DLS solver runs on small CPU lists, requiring no Isaac/Torch import.
Jacobian indexing/frame conventions follow IsaacLab 2.2 task_space_actions.py:
fixed-base body_id - 1; floating-base joint_id + 6; rotate both Jacobian blocks
from world to root. Actual tracking and contacts require simulator validation.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any

ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
BODY_NAME = "base_link"
FRAME = "base_link_in_robot_root"
PROPOSAL_SCHEMA = "rlinf_robolab.eef_joint_target.v1"
ACTION_SEMANTICS = "absolute_arm_joint_positions_7_and_binary_gripper_gt_0.5"
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class EEFControlError(ValueError):
    """Invalid command or robot kinematics; no simulator mutation occurred."""


def _plain(value: Any):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _vector(value, size, name):
    value = _plain(value)
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise EEFControlError(f"{name} must have shape [{size}]")
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not math.isfinite(item) for item in value):
        raise EEFControlError(f"{name} must contain finite real numbers")
    return [float(item) for item in value]


def _positive(value, name, *, zero_allowed=False):
    number = _vector([value], 1, name)[0]
    if number < 0 or (number == 0 and not zero_allowed):
        raise EEFControlError(f"{name} must be {'nonnegative' if zero_allowed else 'positive'}")
    return number


def _quat(value, name="quaternion_wxyz"):
    values = _vector(value, 4, name)
    norm = math.hypot(*values)
    if norm < 1e-12 or not math.isfinite(norm):
        raise EEFControlError(f"{name} must have a finite nonzero norm")
    return [item / norm for item in values]


def _conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]


def _multiply(one, two):
    w, x, y, z = one
    a, b, c, d = two
    return [w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
            w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a]


def _rotate(q, vector):
    return _multiply(_multiply(q, [0.0, *vector]), _conjugate(q))[1:]


def _limit_norm(vector, limit):
    norm = math.hypot(*vector)
    scale = min(1.0, limit / norm) if norm else 1.0
    return [item * scale for item in vector]


def _rotation_error(current, target):
    error = _quat(_multiply(target, _conjugate(current)))
    # Canonical shortest rotation, including deterministic sign at exactly pi.
    first = next((item for item in error if abs(item) > 1e-15), 1.0)
    if first < 0:
        error = [-item for item in error]
    norm = math.hypot(*error[1:])
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2 * math.atan2(norm, max(0.0, error[0]))
    return [item * angle / norm for item in error[1:]]


def _solve(matrix, rhs):
    """Solve the six-dimensional regularized system with partial pivoting."""
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for column in range(6):
        pivot = max(range(column, 6), key=lambda index: abs(rows[index][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        if abs(divisor) < 1e-18 or not math.isfinite(divisor):
            raise EEFControlError("DLS system is numerically singular")
        rows[column] = [value / divisor for value in rows[column]]
        for index in range(6):
            if index != column:
                factor = rows[index][column]
                rows[index] = [x - factor*y for x, y in zip(rows[index], rows[column])]
    return [row[-1] for row in rows]


def compute_joint_target(joint_positions, joint_limits, jacobian_root,
                         current_position, current_quaternion_wxyz,
                         position=None, quaternion_wxyz=None, *, damping=0.05,
                         max_joint_delta_rad=0.05, max_translation_step_m=0.02,
                         max_rotation_step_rad=0.1, joint_limit_margin_rad=0.0,
                         return_diagnostics=False):
    """Pure seven-joint DLS proposal; targets/poses/Jacobian all use root frame.

    Applies dq = J.T (J J.T + damping**2 I)^-1 dx, then bounds each joint
    target by both measured-position delta and native soft joint limits.
    It is a local kinematic update, without collision avoidance or reachability
    guarantees. A zero command holds measured joints if they are within limits.
    """
    joint_positions = _vector(joint_positions, 7, "joint_positions")
    current_position = _vector(current_position, 3, "current_position")
    current_quaternion = _quat(current_quaternion_wxyz, "current_quaternion_wxyz")
    position = current_position if position is None else _vector(position, 3, "position")
    target_quaternion = current_quaternion if quaternion_wxyz is None else _quat(quaternion_wxyz)
    damping = _positive(damping, "damping")
    delta_limit = _positive(max_joint_delta_rad, "max_joint_delta_rad")
    translation_limit = _positive(max_translation_step_m, "max_translation_step_m")
    rotation_limit = _positive(max_rotation_step_rad, "max_rotation_step_rad")
    margin = _positive(joint_limit_margin_rad, "joint_limit_margin_rad", zero_allowed=True)
    jacobian_root, joint_limits = _plain(jacobian_root), _plain(joint_limits)
    if not isinstance(jacobian_root, (list, tuple)) or len(jacobian_root) != 6:
        raise EEFControlError("jacobian_root must have shape [6,7]")
    jacobian = [_vector(row, 7, "jacobian_root row") for row in jacobian_root]
    if not isinstance(joint_limits, (list, tuple)) or len(joint_limits) != 7:
        raise EEFControlError("joint_limits must have shape [7,2]")
    intervals = []
    for measured, pair in zip(joint_positions, joint_limits):
        lower, upper = _vector(pair, 2, "joint limit")
        lower, upper = lower + margin, upper - margin
        if lower >= upper:
            raise EEFControlError("Joint limit interval is empty or reversed")
        lower, upper = max(lower, measured-delta_limit), min(upper, measured+delta_limit)
        if lower > upper:
            raise EEFControlError("Measured joint is too far outside limits for the bounded update")
        intervals.append((lower, upper))
    translation = _limit_norm([goal-now for goal, now in zip(position, current_position)], translation_limit)
    rotation = _limit_norm(_rotation_error(current_quaternion, target_quaternion), rotation_limit)
    error = translation + rotation
    system = [[sum(x*y for x, y in zip(one, two)) + (damping*damping if i == j else 0.0)
               for j, two in enumerate(jacobian)] for i, one in enumerate(jacobian)]
    solved = _solve(system, error)
    delta = [sum(jacobian[row][column] * solved[row] for row in range(6)) for column in range(7)]
    result = [min(upper, max(lower, current + change))
              for current, change, (lower, upper) in zip(joint_positions, delta, intervals)]
    result = _vector(result, 7, "computed joint target")
    if not return_diagnostics:
        return result
    return result, dict(
        measured_arm_joint_positions=joint_positions,
        soft_joint_position_limits=[_vector(pair, 2, "joint limit") for pair in joint_limits],
        target_position_error_m=math.hypot(*[goal-now for goal, now in zip(position, current_position)]),
        target_rotation_error_rad=math.hypot(*_rotation_error(current_quaternion, target_quaternion)),
        bounded_task_delta=error, jacobian_root=jacobian,
        requested_joint_delta=delta,
        joint_clipped=[abs(current+change-value) > 1e-12
                       for current, change, value in zip(joint_positions, delta, result)],
    )


def _row(value, name, env_id=0):
    value = _plain(value)
    if (not isinstance(value, (list, tuple)) or type(env_id) is not int
            or not 0 <= env_id < len(value)):
        raise EEFControlError(f"{name} has no environment row {env_id}")
    return value[env_id]


def _robot_state(env, asset_name, env_id=0):
    if type(env_id) is not int or not 0 <= env_id < int(env.num_envs):
        raise EEFControlError(f"env_id must be in [0,{int(env.num_envs)})")
    robot = env.scene[asset_name]
    joint_names = list(robot.joint_names if hasattr(robot, "joint_names") else robot.data.joint_names)
    body_names = list(robot.body_names if hasattr(robot, "body_names") else robot.data.body_names)
    for name in (*ARM_JOINT_NAMES, "finger_joint"):
        if joint_names.count(name) != 1:
            raise EEFControlError(f"Expected exactly one DROID joint named {name}")
    if body_names.count(BODY_NAME) != 1:
        raise EEFControlError("Expected exactly one DROID base_link body")
    body_id = body_names.index(BODY_NAME)
    joint_ids = [joint_names.index(name) for name in ARM_JOINT_NAMES]
    positions = _vector(_row(robot.data.joint_pos, "joint_pos", env_id), len(joint_names), "joint_pos")
    root_position = _vector(_row(robot.data.root_pos_w, "root_pos_w", env_id), 3, "root_pos_w")
    root_quaternion = _quat(_row(robot.data.root_quat_w, "root_quat_w", env_id))
    body_position = _vector(_row(robot.data.body_pos_w, "body_pos_w", env_id)[body_id], 3, "body_pos_w")
    body_quaternion = _quat(_row(robot.data.body_quat_w, "body_quat_w", env_id)[body_id])
    inverse_root = _conjugate(root_quaternion)
    position = _rotate(inverse_root, [body-root for body, root in zip(body_position, root_position)])
    quaternion = _quat(_multiply(inverse_root, body_quaternion))
    gripper_closed = positions[joint_names.index("finger_joint")] > math.pi / 8
    return robot, joint_ids, body_id, inverse_root, positions, position, quaternion, gripper_closed


def current_eef_pose(env, *, asset_name="robot", env_id=0):
    """Read base_link pose in robot-root, plus measured binary gripper state."""
    _, _, _, _, _, position, quaternion, closed = _robot_state(env, asset_name, env_id)
    return dict(position=position, quaternion_wxyz=quaternion, gripper_closed=closed,
                frame=FRAME)


def next_joint_target(env, position=None, quaternion_wxyz=None, gripper_closed=None, *,
                      asset_name="robot", env_id=0, damping=0.05, max_joint_delta_rad=0.05,
                      max_translation_step_m=0.02, max_rotation_step_rad=0.1,
                      joint_limit_margin_rad=0.0):
    """Return one [panda_joint1..7, binary_gripper] absolute action as a list.

    None position/orientation hold the currently measured pose. None gripper
    thresholds the measured finger position; pass a bool to retain an explicit
    open/closed command during contact. Feed the result through the existing
    joint-target action path, then observe again. This function never env.step's.
    """
    return build_eef_proposal(env, position, quaternion_wxyz, gripper_closed,
        asset_name=asset_name, env_id=env_id, damping=damping, max_joint_delta_rad=max_joint_delta_rad,
        max_translation_step_m=max_translation_step_m,
        max_rotation_step_rad=max_rotation_step_rad,
        joint_limit_margin_rad=joint_limit_margin_rad)["action"]


def build_eef_proposal(env, position=None, quaternion_wxyz=None, gripper_closed=None, *,
                       asset_name="robot", env_id=0, damping=0.05, max_joint_delta_rad=0.05,
                       max_translation_step_m=0.02, max_rotation_step_rad=0.1,
                       joint_limit_margin_rad=0.0):
    """Return a JSON-safe, zero-step proposal using one robot-state read.

    No object state, collision planner or simulator mutation is used. Diagnostics
    are local linear kinematics, not evidence of physical motion or IK convergence.
    The action is rounded to the float32 values the jointpos RPC actually accepts.
    Identity and branch ownership must be checked by the caller before use.
    """
    if gripper_closed is not None and not isinstance(gripper_closed, bool):
        raise EEFControlError("gripper_closed must be bool or None")
    robot, ids, body_id, inverse_root, positions, current_position, current_quaternion, closed = _robot_state(
        env, asset_name, env_id)
    fixed = bool(robot.is_fixed_base)
    jacobian_body = body_id - 1 if fixed else body_id
    if jacobian_body < 0:
        raise EEFControlError("Cannot control the omitted fixed root body")
    joint_columns = ids if fixed else [index + 6 for index in ids]
    all_jacobians = _row(robot.root_physx_view.get_jacobians(), "jacobians", env_id)
    if jacobian_body >= len(all_jacobians):
        raise EEFControlError("PhysX Jacobian body count does not match the articulation")
    selected = all_jacobians[jacobian_body]
    if len(selected) != 6 or any(len(row) != len(positions) + (0 if fixed else 6) for row in selected):
        raise EEFControlError("Unexpected PhysX Jacobian shape")
    world = [_vector([row[index] for index in joint_columns], 7, "Jacobian row") for row in selected]
    root = [[0.0] * 7 for _ in range(6)]
    for column in range(7):
        for offset in (0, 3):
            rotated = _rotate(inverse_root, [world[offset + i][column] for i in range(3)])
            for i in range(3):
                root[offset + i][column] = rotated[i]
    limits = _row(robot.data.soft_joint_pos_limits, "soft_joint_pos_limits", env_id)
    if len(limits) != len(positions):
        raise EEFControlError("Joint limit count differs from joint positions")
    arm, diagnostics = compute_joint_target([positions[index] for index in ids], [limits[index] for index in ids], root,
                               current_position, current_quaternion, position, quaternion_wxyz,
                               damping=damping, max_joint_delta_rad=max_joint_delta_rad,
                               max_translation_step_m=max_translation_step_m,
                               max_rotation_step_rad=max_rotation_step_rad,
                               joint_limit_margin_rad=joint_limit_margin_rad,
                               return_diagnostics=True)
    # Match server.chunk_step's float32 casting without importing NumPy/Torch.
    unrounded_arm = arm
    arm = [struct.unpack("!f", struct.pack("!f", value))[0] for value in arm]
    actual_delta = [goal-positions[index] for index, goal in zip(ids, arm)]
    predicted_delta = [sum(a*b for a, b in zip(row, actual_delta)) for row in root]
    diagnostics.update(proposed_joint_delta=actual_delta,
        predicted_task_delta=predicted_delta,
        linearized_residual=[desired-predicted for desired, predicted
                             in zip(diagnostics["bounded_task_delta"], predicted_delta)],
        physical_tracking_verified=False,
        max_float32_rounding_error_rad=max(abs(a-b) for a, b in zip(arm, unrounded_arm)),
        gripper_command_source="measured_threshold" if gripper_closed is None else "explicit_boolean")
    diagnostics["robot_controller_targets"] = {name: _row(getattr(robot.data, name), name, env_id)
        for name in ("joint_pos_target", "joint_vel_target", "joint_effort_target")
        if hasattr(robot.data, name)}
    diagnostics["clocks"] = {}
    for name in ("episode_length_buf", "_sim_step_counter", "common_step_counter"):
        if not hasattr(env, name):
            continue
        value = _plain(getattr(env, name))
        if (isinstance(value, (list, tuple)) and len(value) == int(env.num_envs)):
            value = value[env_id]
        diagnostics["clocks"][name] = value
    # Jacobian is robot kinematics for local audit, not a measured trajectory
    # or an object pose. Teacher-facing text need only summarize pose errors.
    target_closed = closed if gripper_closed is None else gripper_closed
    return dict(schema=PROPOSAL_SCHEMA, physical_steps=0, frame=FRAME,
        position_units="meters", quaternion_order="wxyz", action_semantics=ACTION_SEMANTICS,
        action=arm + [float(target_closed)],
        current_pose=dict(frame=FRAME, position=current_position,
                          quaternion_wxyz=current_quaternion, gripper_closed=closed),
        target_pose=dict(frame=FRAME, position=current_position if position is None else _vector(position, 3, "position"),
                         quaternion_wxyz=current_quaternion if quaternion_wxyz is None else _quat(quaternion_wxyz),
                         gripper_closed=target_closed),
        controller=dict(method="dls", damping=float(damping),
            max_joint_delta_rad=float(max_joint_delta_rad), max_translation_step_m=float(max_translation_step_m),
            max_rotation_step_rad=float(max_rotation_step_rad), joint_limit_margin_rad=float(joint_limit_margin_rad),
            changes_native_action_controller=False), diagnostics=diagnostics,
        implementation_sha256=IMPLEMENTATION_SHA256)
