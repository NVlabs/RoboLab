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
class StackWhiteMugsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=stacked, params={"objects": ["sideways_white_mug", "upright_white_mug"], "order": "None"})

@dataclass
class StackWhiteMugsTask(Task):
    contact_object_list = ["red_mug", "bowl", "ceramic_mug", "upright_white_mug", "sideways_white_mug", "cordless_drill", "measuring_cup", "table"]
    scene = import_scene("mugs4_measuringcup_drill_bowl.usda", contact_object_list)
    terminations = StackWhiteMugsTerminations
    instruction = {
        "default": "Stack the white mugs on top of each other.",
        "vague": "Stack the white mugs ",
        "specific": "Take a white mug and place it on top of the other white mug",
    }
    episode_length_s: int = 60
    attributes = ['stacking', 'color']
    subtasks = [
        Subtask(
            name="stack_white_mugs",
            conditions=[
                    partial(object_grabbed, object="sideways_white_mug"),
                    partial(stacked, objects=["sideways_white_mug", "upright_white_mug"], order="None"),
            ],
            logical="all",
            score=1.0
            )
    ]
