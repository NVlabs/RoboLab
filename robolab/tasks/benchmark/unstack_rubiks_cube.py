# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_in_container,
    object_outside_of_and_on_surface,
    pick_and_place,
    pick_and_place_on_surface,
)
from robolab.core.task.task import Task


@configclass
class UnstackRubiksCubeTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_outside_of_and_on_surface, params={"object": ["rubiks_cube_middle", "rubiks_cube_top", "rubiks_cube_bottom"], "container": "grey_bin", "surface": "table", "logical": "all", "require_gripper_detached": True})

@dataclass
class UnstackRubiksCubeTask(Task):
    contact_object_list = ["mug", "banana", "ketchup_bottle", "rubiks_cube_middle", "rubiks_cube_top", "rubiks_cube_bottom", "bowl", "grey_bin", "table"]
    scene = import_scene("mug_banana_ketchup_bowl_rubiks3_bin.usda", contact_object_list)
    terminations = UnstackRubiksCubeTerminations
    instruction = {
        "default": "Unstack the rubiks cube tower",
        "vague": "Unstack the tower",
        "specific": "Remove each rubiks cube from the stacked tower one at a time and place them separately on the table",
    }
    episode_length_s: int = 90
    attributes = ['stacking']
    subtasks = [
        pick_and_place_on_surface(
            object=["rubiks_cube_middle", "rubiks_cube_top"],
            surface="table",
            logical="all",
            score=1.0
        )
    ]
