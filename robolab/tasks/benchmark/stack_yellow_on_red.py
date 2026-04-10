# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, stacked
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=stacked,
        params={"objects": ["red_block", "yellow_block"], "order": "bottom_to_top"}
    )


@dataclass
class StackYellowOnRedTask(Task):
    """Task: Stack the yellow block on the red block."""
    contact_object_list = [
        "table", "rubiks_cube", "rubiks_cube_1", "rubiks_cube_2",
        "grey_bin", "lizard_figurine", "birdhouse", "yellow_block",
        "red_block", "green_block", "blue_block", "lizard_figurine_01"
    ]
    scene = import_scene("toys_cleanup.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Stack the yellow block on the red block",
        "vague": "Stack blocks: yellow on red",
        "specific": "Pick up the yellow block on the left and place it precisely on top of the red block in front of the birdhouse",
    }
    episode_length_s: int = 60
    attributes = ['stacking']
    subtasks = [
        Subtask(
            name="stack_yellow_on_red",
            conditions=[
                partial(object_grabbed, object="yellow_block"),
                partial(stacked, objects=["red_block", "yellow_block"], order="bottom_to_top"),
            ],
            logical="all",
            score=1.0
        )
    ]
