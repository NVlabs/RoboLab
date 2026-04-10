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
class ToolOrganizationBothTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["red_hammer", "husky_hammer"], "container": "right_bin", "logical": "all", "require_gripper_detached": True})

@dataclass
class ToolOrganizationBothTask(Task):
    contact_object_list = ["left_bin", "right_bin", "red_hammer", "husky_hammer", "cordless_drill", "spring_clamp", "table"]
    scene = import_scene("tools_container.usda", contact_object_list)
    terminations = ToolOrganizationBothTerminations
    instruction = {
        "default": "Put hammers in the right bin and do not touch anything else",
        "vague": "Put hammers in the right bin",
        "specific": "Pick up the red hammer and the black hammer and place them in the bin on the right side, leaving all other tools untouched",
    }
    episode_length_s: int = 180
    attributes = ['semantics', 'spatial']
    subtasks = [
        pick_and_place(
            object=["red_hammer", "husky_hammer"],
            container="right_bin",
            logical="all",
            score=0.5
        ),
        # pick_and_place(
        #     object=["cordless_drill", "spring_clamp"],
        #     container="left_bin",
        #     logical="all",
        #     score=0.5
        # )
    ]
