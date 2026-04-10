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
class DishesInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["mug", "mug_01", "bowl"], "container": "grey_bin", "logical": "all", "require_gripper_detached": True})

@dataclass
class DishesInBinTask(Task):
    contact_object_list = ["mug", "mug_01", "bowl", "grey_bin", "table"]
    scene = import_scene("mugs2_bananas2_ketchup_rubiks3_bin.usda", contact_object_list)
    terminations = DishesInBinTerminations
    instruction = {
        "default": "Put the dishware in the grey bin",
        "vague": "Put away dishes",
        "specific": "Pick up the two mugs and a bowl from the table and place them into the grey bin",
    }
    episode_length_s: int = 180
    attributes = ['vague', 'semantics']
    subtasks = [
        pick_and_place(
            object=["mug", "mug_01", "bowl"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
