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
class RecycleCartonTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["milk_carton", "orange_juice_carton"], "container": "container_a01", "logical": "all", "require_gripper_detached": True})

@dataclass
class RecycleCartonTask(Task):
    contact_object_list = ["container_a01", "milk_carton", "orange_juice_carton", "alphabet_soup_can",  "smartphone", "mayonnaise_bottle", "ketchup_bottle", "mug", "table"]
    scene = import_scene("cartons_in_crate.usda", contact_object_list)
    terminations = RecycleCartonTerminations
    instruction = {
        "default": "Put the recyclable cartons in the grey bin",
        "vague": "Recycle the cartons",
        "specific": "Pick up the food cartons that can be recycled and place them into the grey bin in the center of the table",
    }
    episode_length_s: int = 90
    attributes = ['semantics', 'vague']
    subtasks = [
        pick_and_place(
            object=["milk_carton", "orange_juice_carton"],
            container="container_a01",
            logical="all",
            score=1.0
        )
    ]
