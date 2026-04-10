# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.task import Task


@configclass
class ClampInRightBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": "spring_clamp", "container": "right_bin", "require_gripper_detached": True})

@dataclass
class ClampInRightBinTask(Task):
    contact_object_list = ["left_bin", "right_bin", "red_hammer", "husky_hammer", "cordless_drill", "spring_clamp", "table"]
    scene = import_scene("tools_container.usda", contact_object_list)
    terminations = ClampInRightBinTerminations
    instruction = {
        "default": "Put the spring clamp in the right bin",
        "vague": "Put the clamp in the right bin",
        "specific": "Pick up the spring clamp from the table and place it in the bin on the right side",
    }
    episode_length_s: int = 60
    attributes = ['semantics', 'spatial']
    subtasks = [
        pick_and_place(
            object=["spring_clamp"],
            container="right_bin",
            logical="all",
            score=1.0
        )
    ]
