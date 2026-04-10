# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import stacked
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class BlockStackingOrderAgnosticTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=stacked, params={"objects": ["red_block", "blue_block", "green_block", "yellow_block"], "order": "None"})

@dataclass
class BlockStackingOrderAgnosticTask(Task):
    contact_object_list = ["red_block", "blue_block", "green_block", "yellow_block", "table"]
    scene = import_scene("colored_blocks.usda", contact_object_list)
    terminations = BlockStackingOrderAgnosticTerminations
    instruction = {
        "default": "Stack the blocks into a tower",
        "vague": "Build a tower",
        "specific": "Pick up the colored blocks and stack them vertically on top of each other, forming one tower",
    }
    episode_length_s: int = 90
    attributes = ['stacking']

    # Progressive stacking subtasks - track intermediate stack formations
    # Since order doesn't matter, we track any 2, then any 3, then all 4 blocks stacked
    subtasks = [
        Subtask(
            name="stack_any_2_blocks",
            conditions={
                "red_blue": partial(stacked, objects=["red_block", "blue_block"], order=None),
                "red_green": partial(stacked, objects=["red_block", "green_block"], order=None),
                "red_yellow": partial(stacked, objects=["red_block", "yellow_block"], order=None),
                "blue_green": partial(stacked, objects=["blue_block", "green_block"], order=None),
                "blue_yellow": partial(stacked, objects=["blue_block", "yellow_block"], order=None),
                "green_yellow": partial(stacked, objects=["green_block", "yellow_block"], order=None),
            },
            logical="any",  # Any pair stacked counts as progress
            score=0.33
        ),
        Subtask(
            name="stack_any_3_blocks",
            conditions={
                "red_blue_green": partial(stacked, objects=["red_block", "blue_block", "green_block"], order=None),
                "red_blue_yellow": partial(stacked, objects=["red_block", "blue_block", "yellow_block"], order=None),
                "red_green_yellow": partial(stacked, objects=["red_block", "green_block", "yellow_block"], order=None),
                "blue_green_yellow": partial(stacked, objects=["blue_block", "green_block", "yellow_block"], order=None),
            },
            logical="any",  # Any 3 blocks stacked counts as progress
            score=0.33
        ),
        Subtask(
            name="stack_all_4_blocks",
            conditions=partial(stacked, objects=["red_block", "blue_block", "green_block", "yellow_block"], order=None),
            score=0.34
        ),
    ]
