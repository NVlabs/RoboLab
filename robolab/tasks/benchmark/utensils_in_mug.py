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
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["fork_big", "spoon_big"],
            "container": "ceramic_mug",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class UtensilsInMugTask(Task):
    """Task: Put the utensils on the mug."""
    contact_object_list = [
        "table", "bowl", "banana", "bagel_07", "coffee_can", "banana_01",
        "yogurt_cup", "coffee_pot", "ceramic_mug", "pitcher", "fork_big",
        "spoon_big", "apple_01", "orange2", "milk_carton",
        "orange_juice_carton", "bagel_01", "bagel_02", "plate_small",
        "plate_large"
    ]
    scene = import_scene("breakfast_table.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the fork and spoon in the ceramicmug",
        "vague": "Put the utensils in the mug",
        "specific": "Take the fork on the bagel plate and put it in the ceramic mug, and take the spoon from the red bowl and put it in the mug",
    }
    episode_length_s: int = 90
    attributes = ['semantics', 'affordance']
    subtasks = [
        pick_and_place(
            object=["fork_big", "spoon_big"],
            container="ceramic_mug",
            logical="all",
            score=1.0
        )
    ]
