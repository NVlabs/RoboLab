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
class ClearOrganicObjectsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "red_onion","avocado01", ], "container": "right_bin", "logical": "all", "require_gripper_detached": True})

@dataclass
class ClearOrganicObjectsTask(Task):
    contact_object_list = ["lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "red_onion", "whitepackerbottle_a01", "avocado01", "crabbypenholder", "milkjug_a01", "serving_bowl", "utilityjug_a03", "right_bin", "table"]
    scene = import_scene("clutter_fruit_bottle_bluebin.usda", contact_object_list)
    terminations = ClearOrganicObjectsTerminations
    instruction = {
        "default": "Clear away the organic objects",
        "vague": "Put away all the food items",
        "specific": "Identify every organic fruits and vegetables (lemon, lime, orange, pomegranate, pumpkin, red onion, avocado) on the table and place each into the bin",
    }
    episode_length_s: int = 240
    attributes = ['semantics']
    subtasks = [
        pick_and_place(
            object=["lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "red_onion","avocado01", ],
            container="right_bin",
            logical="all",
            score=1.0
        )
    ]
