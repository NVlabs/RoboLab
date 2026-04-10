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
class ReorientAllMugsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_upright, params={"object": ["red_mug", "ceramic_mug", "sideways_white_mug", "upright_white_mug"], "up_axis": "z", "logical": "all", "require_gripper_detached": True})

@dataclass
class ReorientAllMugsTask(Task):
    contact_object_list = ["red_mug", "bowl", "ceramic_mug", "upright_white_mug", "sideways_white_mug", "cordless_drill", "measuring_cup", "table"]
    scene = import_scene("mugs4_measuringcup_drill_bowl.usda", contact_object_list)
    terminations = ReorientAllMugsTerminations
    instruction = {
        "default": "Reorient all the mugs upright so that the opening is facing upwards.",
        "vague": "Make sure the mugs are upright",
        "specific": "Check each mug on the table and if any are tipped over or inverted, rotate them so the opening faces upward. Leave it if it's already upright.",
    }
    episode_length_s: int = 90
    attributes = ['reorientation']
    subtasks = [
        Subtask(
            name="reorient_mugs",
            conditions={
                "reorienting_red_mug": [
                    partial(object_grabbed, object="red_mug"),
                    partial(object_upright, object="red_mug", require_gripper_detached=True),
                ],
                "reorienting_ceramic_mug": [
                    partial(object_grabbed, object="ceramic_mug"),
                    partial(object_upright, object="ceramic_mug", require_gripper_detached=True),
                ],
                "reorienting_sideways_white_mug": [
                    partial(object_grabbed, object="sideways_white_mug"),
                    partial(object_upright, object="sideways_white_mug", require_gripper_detached=True),
                ],
            },
            logical="all",
            score=1.0
        )
    ]
