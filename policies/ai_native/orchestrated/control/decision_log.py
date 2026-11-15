# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Adapted from GPT-as-Policy (https://github.com/anonymous-report-421/GPT-as-Policy),
# hybrid_rollout/robolab/robolab_server/decision_log.py.

"""Public structured decisions only; no model reasoning stream or physics calls."""
import json
from datetime import datetime, timezone


def decision_event(*, episode_id, step_id, decision, prediction_id, response,
                   student_action_horizon=15, origin='robolab_server'):
    mode = response['mode']
    return dict(schema='hybrid_rollout.codex_decision.v1', event='codex_decision',
        recorded_at=datetime.now(timezone.utc).isoformat(), origin=origin,
        status='decision_received_not_execution_ack', episode_id=episode_id,
        step_id=step_id, decision=decision, prediction_id=prediction_id,
        request_id=response['request_id'], mode=mode, reason=response['reason'],
        requested_steps=response['steps'], codex_override=mode in ('edit', 'eef'),
        prefix_only=mode == 'student' and response['steps'] < student_action_horizon,
        numeric_action_changed=None, response=response)


def emit(event):
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
