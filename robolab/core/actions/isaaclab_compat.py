# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action terms that preserve RoboLab's public quaternion convention."""

import torch
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

from robolab.core.utils.isaaclab_compat import ISAACLAB_USES_XYZW


class RobolabDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Accept WXYZ absolute pose commands on both Isaac Lab 2 and 3."""

    def process_actions(self, actions: torch.Tensor):
        command_type = self.cfg.controller.command_type
        is_absolute_pose = command_type == "pose" and not self.cfg.controller.use_relative_mode
        if ISAACLAB_USES_XYZW and is_absolute_pose:
            external_actions = actions
            actions = actions.clone()
            actions[..., 3:7] = external_actions[..., [4, 5, 6, 3]]
            super().process_actions(actions)
            # Recordings and raw-action diagnostics remain version-independent.
            self._raw_actions[:] = external_actions
            return
        super().process_actions(actions)


@configclass
class RobolabDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[DifferentialInverseKinematicsAction] = RobolabDifferentialInverseKinematicsAction
