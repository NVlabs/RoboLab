# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, object_on_top, pick_and_place_on_surface
from robolab.core.task.task import Task


@configclass
class BagelsOnPlateTerminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_on_top, params={"object": ["bagel_00", "bagel_06"], "reference_object": "plate_large", "gripper_name": "gripper", "logical": "all", "require_gripper_detached": True})

@dataclass
class BagelsOnPlateTask(Task):
    contact_object_list =  ["bagel_00", "bagel_06", "banana", "bowl", "plate_large", "table"]
    scene = import_scene("bagel_plate_banana_bowl.usda", contact_object_list)
    terminations = BagelsOnPlateTerminations
    instruction = {
        "default": "Put the bagels on the plate",
        "vague": "Plate some bagels",
        "specific": "Pick up the two bagels from the table and place them onto the plate",
    }
    episode_length_s: int = 60
    attributes = ['semantics']
    subtasks = [
        pick_and_place_on_surface(
            object=["bagel_00", "bagel_06"],
            surface="plate_large",
            logical="all",
            score=1.0
        )
    ]
