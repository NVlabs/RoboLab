# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from collections.abc import Sequence
from typing import Any

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class MetricsRecorderTerm(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def record_post_step(self):
        states = self._env.scene.get_state(is_relative=True)

        return "metric", states



@configclass
class TrajectoryMetricsRecorderCfg(RecorderTermCfg):
    """Configuration for the subtask completion recorder term."""

    class_type: type[RecorderTerm] = MetricsRecorderTerm
