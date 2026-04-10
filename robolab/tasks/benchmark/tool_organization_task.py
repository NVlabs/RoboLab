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
class ToolOrganizationTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["red_hammer", "husky_hammer"], "container": "left_bin", "logical": "all", "tolerance": 0.0, "require_gripper_detached": True})

@dataclass
class ToolOrganizationTask(Task):
    contact_object_list = ["left_bin", "right_bin", "red_hammer", "husky_hammer", "cordless_drill", "spring_clamp", "table"]
    scene = import_scene("tools_container.usda", contact_object_list)
    terminations = ToolOrganizationTerminations
    instruction = {
        "default": "Put the red hammer and black hammer in the left bin",
        "vague": "Put hammers in the left bin",
        "specific": "Pick up the red-handled hammer and the black-handled hammer and place both in the left bin",
    }
    episode_length_s: int = 180
    attributes = ['semantics', 'spatial', 'color']
    subtasks = [
        pick_and_place(
            object=["red_hammer", "husky_hammer"],
            container="left_bin",
            logical="all",
            score=1.0
        )
    ]
