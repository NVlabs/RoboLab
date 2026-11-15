# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Adapted from GPT-as-Policy (https://github.com/anonymous-report-421/GPT-as-Policy),
# hybrid_rollout/robolab/skill/schema.py.

def _obj(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def response_schema(request_id=None, *, student_step_max=15):
    if type(student_step_max) is not int or not 1 <= student_step_max <= 32:
        raise ValueError("student_step_max must be an integer in 1..32")
    number = {"type": "number"}
    vector3 = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    vector4 = {"type": "array", "items": number, "minItems": 4, "maxItems": 4}
    string = {"type": "string"}
    progress = _obj(dict(
        verified_completed={"type": "array", "items": string},
        currently_attempting=string,
        remaining={"type": "array", "items": string},
    ))
    assessment = _obj(dict(
        task_progress=progress,
        current_subgoal=string,
        execution_status={"type": "string", "enum": [
            "not_started", "progressing", "failed", "uncertain", "recovered"]},
        execution_evidence=string,
        expected_next_intent=string,
        predicted_next_intent=string,
        intent_status={"type": "string", "enum": ["aligned", "misaligned", "uncertain"]},
        intent_evidence=string,
    ))
    identifier = string if request_id is None else {"type": "string", "enum": [request_id]}
    return _obj(dict(
        request_id=identifier,
        mode={"type": "string", "enum": ["student", "edit", "eef"]},
        steps={"type": "integer", "minimum": 1, "maximum": max(5, student_step_max)},
        reason=string,
        edit=_obj(dict(delta_position=vector3, delta_rotation_vector=vector3,
                       gripper={"type": "string", "enum": ["keep", "open", "closed"]})),
        target=_obj(dict(position=vector3, quaternion_wxyz=vector4,
                         gripper_closed={"type": "boolean"})),
        assessment=assessment,
    ))
