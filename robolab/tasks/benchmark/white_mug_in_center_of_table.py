# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_dropped, object_grabbed, object_on_center
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class WhiteMugInCenterOfTableTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_on_center, params={"object": "mug", "reference_object": "table", "tolerance": 0.05, "require_contact_with": True, "require_gripper_detached": True})

@dataclass
class WhiteMugInCenterOfTableTask(Task):
    contact_object_list = ["mug", "bowl", "table", "alphabet_soup_can", "orange_juice_carton", "smartphone", "milk_carton"]
    scene = import_scene("objects_around_table.usda", contact_object_list)
    terminations = WhiteMugInCenterOfTableTerminations
    instruction = {
        "default": "Put the white mug in the center of the table.",
        "vague": "Move the mug to the center",
        "specific": "Pick up the white ceramic mug and place it in the center of the table surface away from the other objects",
    }
    episode_length_s: int = 30
    attributes = ['color', 'spatial']
    subtasks = [
        Subtask(
            name="white_mug_in_center_of_table",
            conditions=[
                partial(object_grabbed, object="mug"),
                partial(object_on_center, object="mug", reference_object="table", tolerance=0.05),
                partial(object_dropped, object="mug"),
            ],
            logical="all",
            score=1.0
        )
    ]
