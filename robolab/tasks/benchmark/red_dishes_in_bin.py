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
    success = DoneTerm(func=object_in_container, params={"object": ["mug", "bowl"], "container": "grey_bin", "logical": "all", "require_gripper_detached": True})

@dataclass
class RedDishesInBinTask(Task):
    contact_object_list = ["mug", "bowl", "grey_bin", "table"]
    scene = import_scene("rubiks_cube_banana_bowl_mug_bin.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the red dishware in the grey bin",
        "vague": "Clean up red dishware",
        "specific": "Pick up each red mug and bowl from the table and place them in the grey bin",
    }
    episode_length_s: int = 60
    attributes = ['color', 'semantics']
    subtasks = [
        pick_and_place(
            object=["mug", "bowl"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
