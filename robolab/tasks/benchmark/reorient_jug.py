# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, object_in_container, object_upright
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_upright,
        params={
            "object": ["utilityjug_a02"],
            "up_axis": "z",
            "logical": "all",
            "require_gripper_detached": True,
        },
    )

@dataclass
class ReorientJugTask(Task):
    contact_object_list = [
        "large_storage_rack", "whitepackerbottle_a01", "whitepackerbottle_a02",
        "utilityjug_a01", "utilityjug_a02", "squarepail_a01", "plasticpail_a02",
        "whitepackerbottle_a03", "table"
    ]
    scene = import_scene("shelf_with_cleaning_products.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Stand the jug upright",
        "vague": "Fix the jug",
        "specific": "Grasp the jug that is tipped on its side place it upright",
    }
    episode_length_s: int = 60
    attributes = ['semantics', 'reorientation', 'affordance']
    subtasks = [
        Subtask(
            name="reorient_jug",
            conditions={
                "reorienting_jug": [
                    partial(object_grabbed, object="utilityjug_a02"),
                    partial(object_upright, object="utilityjug_a02", require_gripper_detached=True),
                ],
            },
        )
    ]
