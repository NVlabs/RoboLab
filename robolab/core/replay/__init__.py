# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Faithful playback of recorded episodes.

Everything replay needs beyond the recorded actions lives here, organized by
what is being restored or checked (see ``docs/replay.md`` for the user guide):

- :mod:`~robolab.core.replay.env_config` — load the ``env_cfg.json`` recorded
  next to an episode and overlay it onto a freshly built env config, so replay
  uses the exact recorded config values rather than the current repo's task
  definitions.
- :mod:`~robolab.core.replay.scene_state` — reset the scene to the recorded
  initial state, and validate per-step sim state against the recorded states.

For warnings when the current simulator stack or GPU differs from the recording,
see :func:`robolab.core.utils.version_utils.warn_on_recording_mismatch`.

Typical driver flow:

    env_cfg = parse_env_cfg(task, ...)
    recorded, path = load_recorded_env_cfg(hdf5_path)
    apply_recorded_env_cfg(env_cfg, recorded)
    env, _ = create_env(env_cfg, ...)
    restore_recorded_initial_state(env, hdf5_path, episode)
    # step recorded actions open-loop; StateValidator.check_step() per step
"""

from robolab.core.replay.env_config import apply_recorded_env_cfg, load_recorded_env_cfg
from robolab.core.replay.scene_state import StateValidator, restore_recorded_initial_state

__all__ = [
    "apply_recorded_env_cfg",
    "load_recorded_env_cfg",
    "restore_recorded_initial_state",
    "StateValidator",
]
