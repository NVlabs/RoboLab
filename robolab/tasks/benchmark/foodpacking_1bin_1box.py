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
class FoodPacking1BoxesTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["cheez_it"], "container": "bin_a06", "logical": "all", "require_gripper_detached": True})

@dataclass
class FoodPacking1BoxesTask(Task):
    contact_object_list = ["bin_a06", "cheez_it", "mustard","tomato_soup_can", "table"]
    scene = import_scene("foodpacking_1bin_1box_1can.usda", contact_object_list)
    terminations = FoodPacking1BoxesTerminations
    instruction = {
        "default": "Pack boxed foods into the bin",
        "vague": "Pack boxes",
        "specific": "Pick up the boxed food item and place it inside the bin",
    }
    episode_length_s: int = 60
    attributes = ['semantics']
    subtasks = [
        pick_and_place(
            object=["cheez_it"],
            container="bin_a06",
            logical="all",
            score=1.0
        )
    ]
