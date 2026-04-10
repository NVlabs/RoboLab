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
class RedItemsInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["mug", "bowl"], "container": "grey_bin", "logical": "all", "require_gripper_detached": True})

@dataclass
class RedItemsInBinTask(Task):
    contact_object_list = ["mug", "bowl", "grey_bin", "table"]
    scene = import_scene("rubiks_cube_banana_bowl_mug_bin.usda", contact_object_list)
    terminations = RedItemsInBinTerminations
    instruction = {
        "default": "Put all the red things in the grey bin",
        "vague": "Sort items by color",
        "specific": "Identify every red-colored object on the table and place each one into the grey bin",
    }
    episode_length_s: int = 60
    attributes = ['color', 'sorting']
    subtasks = [
        pick_and_place(
            object=["mug", "bowl"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
