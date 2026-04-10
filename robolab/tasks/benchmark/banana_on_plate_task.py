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
class BananaOnPlateTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_on_top, params={"object": ["banana"], "reference_object": "plate_large", "gripper_name": "gripper", "require_gripper_detached": True})

@dataclass
class BananaOnPlateTask(Task):
    contact_object_list = ["bagel_00", "bagel_06", "banana", "bowl", "plate_large", "table"]
    scene = import_scene("bagel_plate_banana_bowl.usda", contact_object_list)
    terminations = BananaOnPlateTerminations
    instruction = {
        "default": "Pick up the banana and put it on the plate",
        "vague": "Put fruit on the dish",
        "specific": "Pick up the yellow banana and place it flat on the white ceramic plate",
    }
    episode_length_s: int = 40
    attributes = []
    subtasks = [
        pick_and_place_on_surface(
            object=["banana"],
            surface="plate_large",
            logical="all",
            score=1.0
        )
    ]
