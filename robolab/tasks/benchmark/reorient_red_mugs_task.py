# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, object_upright
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class ReorientRedMugTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_upright, params={"object": ["red_mug"], "up_axis": "z", "logical": "all", "require_gripper_detached": True})

@dataclass
class ReorientRedMugTask(Task):
    contact_object_list = ["red_mug", "bowl", "ceramic_mug", "upright_white_mug", "sideways_white_mug", "cordless_drill", "measuring_cup", "table"]
    scene = import_scene("mugs4_measuringcup_drill_bowl.usda", contact_object_list)
    terminations = ReorientRedMugTerminations
    instruction = {
        "default": "Put the red mug upright so that the opening is facing upwards.",
        "vague": "Fix the red cup",
        "specific": "Check the red ceramic mug and if it is tipped over or inverted, rotate it so the opening faces upward. Leave it if already upright.",
    }
    episode_length_s: int = 60
    attributes = ['reorientation', 'color']
    subtasks = [
        Subtask(
            name="reorient_red_mug",
            conditions={
                "reorienting_red_mug": [
                    partial(object_grabbed, object="red_mug"),
                    partial(object_upright, object="red_mug", require_gripper_detached=True),
                ],
            },
            logical="all",
            score=1.0
        )
    ]
