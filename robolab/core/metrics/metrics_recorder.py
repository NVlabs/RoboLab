# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from robolab.core.utils.isaaclab_compat import scene_state_from_isaaclab


class MetricsRecorderTerm(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def record_post_step(self):
        states = scene_state_from_isaaclab(self._env.scene.get_state(is_relative=True))

        return "metric", states



@configclass
class TrajectoryMetricsRecorderCfg(RecorderTermCfg):
    """Configuration for the subtask completion recorder term."""

    class_type: type[RecorderTerm] = MetricsRecorderTerm
