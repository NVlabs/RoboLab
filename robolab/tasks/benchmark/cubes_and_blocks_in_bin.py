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
            "object": [
                "rubiks_cube", "rubiks_cube_1", "rubiks_cube_2",
                "yellow_block", "red_block", "green_block", "blue_block"
            ],
            "container": "grey_bin",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class CubesAndBlocksInBinTask(Task):
    """Task: Put all the cubes and blocks in the bin."""
    contact_object_list = [
        "table", "rubiks_cube", "rubiks_cube_1", "rubiks_cube_2",
        "grey_bin", "lizard_figurine", "birdhouse", "yellow_block",
        "red_block", "green_block", "blue_block", "lizard_figurine_01"
    ]
    scene = import_scene("toys_cleanup.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put all the cubes and blocks in the bin",
        "vague": "Put blocks away",
        "specific": "Pick up every solid-colored cube and rubiks cubes from the table and place them all into the bin",
    }
    episode_length_s: int = 240
    attributes = ['sorting', 'conjunction']
    subtasks = [
        pick_and_place(
            object=["rubiks_cube", "rubiks_cube_1", "rubiks_cube_2"],
            container="grey_bin",
            logical="all",
            score=0.5
        ),
        pick_and_place(
            object=["yellow_block", "red_block", "green_block", "blue_block"],
            container="grey_bin",
            logical="all",
            score=0.5
        )
    ]
