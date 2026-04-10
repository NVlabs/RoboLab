# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_dropped, object_grabbed, object_outside_of
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class BananasOutOfBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_outside_of,
        params={"object": ["banana_02", "banana_04"], "container": "grey_bin", "logical": "all", "tolerance": 0.00, "require_gripper_detached": True}
    )

@dataclass
class BananasOutOfBinTask(Task):
    contact_object_list = ["banana", "banana_01", "banana_02", "banana_03", "banana_04", "grey_bin", "table"]
    scene = import_scene("bananas_5_grey_bin.usda", contact_object_list)
    terminations = BananasOutOfBinTerminations
    instruction = {
        "specific": "Take all the bananas out of the grey bin and put it on the table.",
        "default": "Take the bananas out",
        "vague": "Empty the grey bin",
    }
    episode_length_s: int = 90
    attributes = ['semantics', 'spatial']

    subtasks = [
        Subtask(
            name="bananas_out_of_bin",
            conditions={
                "banana_02": [
                    partial(object_grabbed, object="banana_02"),
                    partial(object_outside_of, object="banana_02", container="grey_bin", tolerance=0.00),
                    partial(object_dropped, object="banana_02"),
                ],
                "banana_04": [
                    partial(object_grabbed, object="banana_04"),
                    partial(object_outside_of, object="banana_04", container="grey_bin", tolerance=0.00),
                    partial(object_dropped, object="banana_04"),
                ],
            },
            logical="all",
            score=1.0
        )
    ]
