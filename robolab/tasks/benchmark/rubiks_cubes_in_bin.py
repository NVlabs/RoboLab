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
            "object": ["rubiks_cube", "rubiks_cube_1", "rubiks_cube_2"],
            "container": "grey_bin",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class RubiksCubesInBinTask(Task):
    """Task: Sort all rubiks cubes into the bin."""
    contact_object_list = [
        "table", "rubiks_cube", "rubiks_cube_1", "rubiks_cube_2",
        "grey_bin", "lizard_figurine", "birdhouse", "yellow_block",
        "red_block", "green_block", "blue_block", "lizard_figurine_01"
    ]
    scene = import_scene("toys_cleanup.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Sort all rubiks cubes into the bin",
        "vague": "Put away rubiks cubes",
        "specific": "Pick up only the three rubiks cube from the table and place them all into the grey bin",
    }
    episode_length_s: int = 120
    attributes = ['sorting']
    subtasks = [
        pick_and_place(
            object=["rubiks_cube", "rubiks_cube_1", "rubiks_cube_2"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
