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
class PickOrangeObjectTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": "measuring_cups_1", "surface": "table", "distance": 0.05}
    )

@dataclass
class PickOrangeObjectTask(Task):
    contact_object_list = ["table", "redonion", "serving_bowl", "clay_plates", "wooden_spoons", "spatula", "storage_box", "tomato_sauce_can", "measuring_cups_1", "pink_spaghetti_spoon", "spoon_1", "green_serving_spoon", "storage_box_01", "ladle", "wooden_bowl", "potato_masher"]
    scene = import_scene("cooking_table.usda", contact_object_list)
    terminations = PickOrangeObjectTerminations
    instruction = {
        "default": "Pick up the orange measuring cup",
        "vague": "Pick up orange-colored thing",
        "specific": "Identify and pick up the orange-colored object from the cooking area",
    }
    episode_length_s: int = 60
    attributes = ['color']
    subtasks = [
        Subtask(
            name="pick_orange_object",
            conditions=[
                partial(object_grabbed, object="measuring_cups_1"),
                partial(object_above, object="measuring_cups_1", reference_object="table", z_margin=0.05),
            ],
            logical="all",
            score=1.0
        )
    ]
