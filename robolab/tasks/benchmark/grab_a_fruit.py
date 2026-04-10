# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_above, object_grabbed, object_picked_up
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success_banana = DoneTerm(func=object_picked_up, params={"object": "banana", "surface": "table", "distance": 0.05})
    success_banana_01 = DoneTerm(func=object_picked_up, params={"object": "banana_01", "surface": "table", "distance": 0.05})
    success_apple = DoneTerm(func=object_picked_up, params={"object": "apple_01", "surface": "table", "distance": 0.05})
    success_orange = DoneTerm(func=object_picked_up, params={"object": "orange2", "surface": "table", "distance": 0.05})


@dataclass
class GrabAFruitTask(Task):
    """Task: Grab a fruit."""
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
        "default": "Pick up a fruit",
        "vague": "Grab a fruit",
        "specific": "Reach for and grasp any one of the fruits on the table and lift it off the surface",
    }
    episode_length_s: int = 30
    attributes = ['semantics']
    subtasks = [
        Subtask(
            name="grab_a_fruit",
            conditions=[
                partial(object_grabbed, object="banana"),
                partial(object_grabbed, object="banana_01"),
                partial(object_grabbed, object="apple_01"),
                partial(object_grabbed, object="orange2"),
            ],
            logical="any",
            score=1.0
        )
    ]
