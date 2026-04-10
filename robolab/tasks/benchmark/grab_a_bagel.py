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
    success_bagel_02 = DoneTerm(func=object_picked_up, params={"object": "bagel_02", "surface": "table", "distance": 0.05})
    success_bagel_01 = DoneTerm(func=object_picked_up, params={"object": "bagel_01", "surface": "table", "distance": 0.05})
    success_bagel_07 = DoneTerm(func=object_picked_up, params={"object": "bagel_07", "surface": "table", "distance": 0.05})


@dataclass
class GrabABagelTask(Task):
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
        "default": "Grab a bagel",
        "vague": "Grab some bread",
        "specific": "Reach for one of the bagels on the table and pick it up",
    }
    episode_length_s: int = 30
    attributes = ['semantics']
    subtasks = [
        Subtask(
            name="grab_a_bagel",
            conditions=[
                partial(object_grabbed, object="bagel_02"),
                partial(object_grabbed, object="bagel_01"),
                partial(object_grabbed, object="bagel_07"),
            ],
            logical="any",
            score=1.0
        )
    ]
