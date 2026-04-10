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
class CookingPickPastaToolTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": ["pink_spaghetti_spoon"], "container": "storage_box_01", "logical": "all", "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class CookingPickPastaToolTask(Task):
    contact_object_list = ["table", "redonion", "serving_bowl", "clay_plates", "wooden_spoons", "spatula", "storage_box", "tomato_sauce_can", "measuring_cups_1", "pink_spaghetti_spoon", "spoon_1", "green_serving_spoon", "storage_box_01", "ladle", "wooden_bowl", "potato_masher"]
    scene = import_scene("cooking_table.usda", contact_object_list)
    terminations = CookingPickPastaToolTerminations
    instruction = {
        "default": "Move the pink tool from this utensil container to the other utensil holder",
        "vague": "Move the pink tool over to the other container",
        "specific": "Remove the pink pasta tool from its current utensil container and place it in the other utensil holder",
    }
    episode_length_s: int = 60
    attributes = ['spatial', 'vague', 'color']

    subtasks = [
        pick_and_place(
            object=["pink_spaghetti_spoon"],
            container="storage_box_01",
            logical="all",
            score=1.0
        )
    ]
