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
            "object": ["lizard_figurine", "lizard_figurine_01"],
            "container": "grey_bin",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class AnimalsInBinTask(Task):
    """Task: Put the animals in the bin."""
    contact_object_list = [
        "table", "rubiks_cube", "rubiks_cube_1", "rubiks_cube_2",
        "grey_bin", "lizard_figurine", "birdhouse", "yellow_block",
        "red_block", "green_block", "blue_block", "lizard_figurine_01"
    ]
    scene = import_scene("toys_cleanup.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the lizards in the bin",
        "vague": "Put away the animals",
        "specific": "Pick up the two identical green lizard figurines and place them into the bin",
    }
    episode_length_s: int = 90
    attributes = ['semantics']
    subtasks = [
        pick_and_place(
            object=["lizard_figurine", "lizard_figurine_01"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
