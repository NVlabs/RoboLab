# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class BananasInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": ["banana", "banana_01", "banana_03"], "container": "grey_bin", "logical": "choose", "K": 1, "require_gripper_detached": True, 'require_contact_with': False})

@dataclass
class BananasInBinThreeTotalTask(Task):
    contact_object_list = ["banana", "banana_01", "banana_02", "banana_03", "banana_04", "grey_bin", "table"]
    scene = import_scene("bananas_5_grey_bin.usda", contact_object_list)
    terminations = BananasInBinTerminations
    instruction = {
        "default": "Make sure there are 3 (three) bananas in the grey bin.",
        "vague": "3 bananas in the bin",
        "specific": "In addition to bananas in the bin, add one more banana to the grey bin until there are exactly three bananas inside",
    }
    episode_length_s: int = 60
    attributes = ['semantics', 'counting']

    subtasks = [
        pick_and_place(
            object=["banana", "banana_01", "banana_03"],
            container="grey_bin",
            logical="choose",
            K=1,
            score=1.0
        )
    ]
