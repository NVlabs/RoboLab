# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_on_top, pick_and_place_on_surface
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_on_top,
        params={
            "object": ["banana", "banana_01"],
            "reference_object": "plate_small",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class MoveBananaToBagelPlateTask(Task):
    """Task: Move the bananas to the small plate."""
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
        "default": "Move the bananas to the bagel plate",
        "vague": "Put the bananas with the bagels",
        "specific": "Pick up the yellow bananas and transfer them onto the plate containing the bagels",
    }
    episode_length_s: int = 90
    attributes = ['semantics']
    subtasks = [
        pick_and_place_on_surface(
            object=["banana", "banana_01"],
            surface="plate_small",
            logical="all",
            score=1.0
        )
    ]
