# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_outside_of, pick_and_place_on_surface
from robolab.core.task.task import Task


@configclass
class CookingClearPlateTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_outside_of,
        params={"object": ["spoon_1", "measuring_cups_1"], "container": "clay_plates", "logical": "all", "tolerance": 0.0, "require_gripper_detached": True}
    )

@dataclass
class CookingClearPlateTask(Task):
    contact_object_list = ["table", "redonion", "serving_bowl", "clay_plates", "wooden_spoons", "spatula", "storage_box", "tomato_sauce_can", "measuring_cups_1", "pink_spaghetti_spoon", "spoon_1", "green_serving_spoon", "storage_box_01", "ladle", "wooden_bowl", "potato_masher"]
    scene = import_scene("cooking_table.usda", contact_object_list)
    terminations = CookingClearPlateTerminations
    instruction = {
        "default": "Put the two measuring cups outside of the plate",
        "vague": "Clear the plate",
        "specific": "Pick up the orange measuring cup and the blue measuring cup from the plate and place each on the table surface",
    }
    episode_length_s: int = 180
    attributes = ['color', 'sorting']

    subtasks = [
        pick_and_place_on_surface(
            object=["spoon_1", "measuring_cups_1"],
            surface="table",
            logical="all",
            score=1.0
        )
    ]
