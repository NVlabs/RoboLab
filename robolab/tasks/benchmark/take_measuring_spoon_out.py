# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, object_outside_of, stacked
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class TakeMeasuringSpoonOutTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_outside_of, params={"object": ["measuring_cup"], "container": "bowl", "logical": "all", "require_gripper_detached": True})

@dataclass
class TakeMeasuringSpoonOutTask(Task):
    contact_object_list = ["red_mug", "bowl", "ceramic_mug", "upright_white_mug", "sideways_white_mug", "cordless_drill", "measuring_cup", "table"]
    scene = import_scene("mugs4_measuringcup_drill_bowl_v2.usda", contact_object_list)
    terminations = TakeMeasuringSpoonOutTerminations
    instruction = {
        "default": "Take the white colored measuring spoon out of the red bowl and put it on the table.",
        "vague": "Take the measuring spoon out of the bowl",
        "specific": "Reach into the bowl, grasp the measuring spoon, lift it out, and place it on the table surface",
    }
    episode_length_s: int = 40
    attributes = ['semantics']
    subtasks = [
        Subtask(
            name="take_measuring_cup_out",
            conditions={
                "taking_measuring_cup_out": [
                    partial(object_grabbed, object="measuring_cup"),
                    partial(object_outside_of, object="measuring_cup", container="bowl"),
                ],
            },
            logical="all",
            score=1.0
        )
    ]
