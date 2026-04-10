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
            "object": ["whitepackerbottle_a01", "whitepackerbottle_a02", "whitepackerbottle_a03"],
            "container": "squarepail_a01",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class PlasticBottlesInSquarePailTask(Task):
    contact_object_list = [
        "large_storage_rack", "whitepackerbottle_a01", "whitepackerbottle_a02",
        "utilityjug_a01", "utilityjug_a02", "squarepail_a01", "plasticpail_a02",
        "whitepackerbottle_a03", "table"
    ]
    scene = import_scene("shelf_with_cleaning_products.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put all the small plastic bottles in the square pail",
        "vague": "Clean up small bottles in the square pail",
        "specific": "Pick up the three small plastic bottles and place them inside the square shaped pail",
    }
    episode_length_s: int = 180
    attributes = ['semantics', 'sorting', 'size']
    subtasks = [
        pick_and_place(
            object=["whitepackerbottle_a01", "whitepackerbottle_a02", "whitepackerbottle_a03"],
            container="squarepail_a01",
            logical="all",
            score=1.0
        )
    ]
