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
class FruitsOnionToPlateTerminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_on_top,
        params={"object": ["redonion"], "reference_object": "clay_plates", "logical": "all", "require_gripper_detached": True}
    )

@dataclass
class FruitsOnionToPlateTask(Task):
    contact_object_list = ["table", "lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "redonion", "serving_bowl", "clay_plates", "wooden_bowl", "wooden_spoons", "spatula", "storage_box"]
    scene = import_scene("fruits_in_basket.usda", contact_object_list)
    terminations = FruitsOnionToPlateTerminations
    instruction = {
        "default": "Put the onion on the plate",
        "vague": "Plate the onion",
        "specific": "Pick up the round onion and place it on the flat plate",
    }
    episode_length_s: int = 60
    attributes = ['semantics', 'vague']

    subtasks = [
        pick_and_place_on_surface(
            object=["redonion"],
            surface="clay_plates",
            logical="all",
            score=1.0
        )
    ]
