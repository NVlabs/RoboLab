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
class FruitsMovingOrangeOrLimeTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": ["orange_01", "lime01", "orange_02", "lime01_01"], "container": "wooden_bowl", "logical": "any", "tolerance": 0.0, "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class FruitsMovingOrangeOrLimeTask(Task):
    contact_object_list = ["table", "lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "redonion", "serving_bowl", "clay_plates", "wooden_bowl", "wooden_spoons", "spatula", "storage_box"]
    scene = import_scene("fruits_in_basket.usda", contact_object_list)
    terminations = FruitsMovingOrangeOrLimeTerminations
    instruction = {
        "default": "Move an orange or a lime to the wood bowl",
        "vague": "Put an orange or lime fruit in the wooden bowl",
        "specific": "Select either an orange or a green lime and place it inside the wooden bowl",
    }
    episode_length_s: int = 60
    attributes = ['conjunction']

    # Updated to use new clean API
    subtasks = [
        pick_and_place(
            object=["orange_01", "lime01", "orange_02", "lime01_01"],
            container="wooden_bowl",
            logical="any",
            score=1.0
        )
    ]
