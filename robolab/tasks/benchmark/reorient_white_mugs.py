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
class ReorientWhiteMugsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_upright, params={"object": ["upright_white_mug", "sideways_white_mug"], "up_axis": "z", "logical": "all", "require_gripper_detached": True})

@dataclass
class ReorientWhiteMugsTask(Task):
    contact_object_list = ["red_mug", "bowl", "ceramic_mug", "upright_white_mug", "sideways_white_mug", "cordless_drill", "measuring_cup", "table"]
    scene = import_scene("mugs4_measuringcup_drill_bowl_v2.usda", contact_object_list)
    terminations = ReorientWhiteMugsTerminations
    instruction = {
        "default": "Make sure all the white mugs are upright so that the opening is facing upwards.",
        "vague": "Fix the cups",
        "specific": "For each white mug that is tipped over or upside down, rotate it so the opening faces straight up",
    }
    episode_length_s: int = 60
    attributes = ['reorientation', 'color']
    subtasks = [
        Subtask(
            name="reorient_white_mugs",
            conditions={
                "reorienting_sideways_white_mug": [
                    partial(object_grabbed, object="sideways_white_mug"),
                    partial(object_upright, object="sideways_white_mug", require_gripper_detached=True),
                ],
            },
            logical="all",
            score=1.0
        )
    ]
