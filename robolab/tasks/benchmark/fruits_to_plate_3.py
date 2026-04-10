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
class FruitsOnPlate3Terminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_on_top,
        params={"object": ["lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01"], "reference_object": "clay_plates", "logical": "choose", "K": 3, "require_gripper_detached": True}
    )

@dataclass
class FruitsOnPlate3Task(Task):
    contact_object_list = ["table", "lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "redonion", "serving_bowl", "clay_plates", "wooden_spoons", "spatula", "storage_box"]
    scene = import_scene("fruits_out_of_basket.usda", contact_object_list)
    terminations = FruitsOnPlate3Terminations
    instruction = {
        "default": "Put three (3) fruits on the plate",
        "vague": "Put three fruit on the plate",
        "specific": "Select exactly three fruits (lemon, lime, orange, pomegranate) from the table and place all of them on the plate",
    }
    episode_length_s: int = 200
    attributes = ['semantics', 'vague', 'counting']

    subtasks = [
        pick_and_place_on_surface(
            object=["lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01"],
            surface="clay_plates",
            logical="choose",
            K=3,
            score=1.0
        )
    ]
