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
class BlockStackingSpecifiedOrderTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=stacked, params={"objects": ["red_block", "blue_block", "green_block", "yellow_block"], "order": "bottom_to_top"})

@dataclass
class BlockStackingSpecifiedOrderTask(Task):
    contact_object_list = ["red_block", "blue_block", "green_block", "yellow_block", "table"]
    scene = import_scene("colored_blocks.usda", contact_object_list)
    terminations = BlockStackingSpecifiedOrderTerminations
    instruction = {
        "default": "Stack the blocks in the order from bottom to top: red, blue, green, yellow",
        "vague": "Stack in the order of red, blue, green, yellow",
        "specific": "Build a tower by placing the red block first, then the blue block on top, then the green, and finally the yellow block on top as a single tower",
    }
    episode_length_s: int = 90
    attributes = ['stacking', 'color']
    subtasks = [
        Subtask(
            conditions=partial(stacked, objects=["red_block", "blue_block"], order="bottom_to_top"),
            score=0.33
        ),
        Subtask(
            conditions=partial(stacked, objects=["blue_block", "green_block"], order="bottom_to_top"),
            score=0.33
        ),
        Subtask(
            conditions=partial(stacked, objects=["green_block", "yellow_block"], order="bottom_to_top"),
            score=0.34
        ),
    ]
