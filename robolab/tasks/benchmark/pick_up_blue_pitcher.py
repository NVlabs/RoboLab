# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_above, object_grabbed, object_picked_up
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class PickUpBluePitcherTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_picked_up, params={"object": "pitcher", "surface": "table", "distance": 0.05})

@dataclass
class PickUpBluePitcherTask(Task):
    contact_object_list = ["plasticjerrican_a02", "plasticpail_a02", "pitcher", "table"]
    scene = import_scene("blue.usda", contact_object_list)
    terminations = PickUpBluePitcherTerminations
    instruction = {
        "default": "Pick up the large blue pitcher",
        "vague": "Pick up blue object",
        "specific": "Pick up the large blue pitcher on the table by the side handle and lift it off the table",
    }
    episode_length_s: int = 30
    attributes = ['color', 'affordance']
    subtasks = [
        Subtask(
            name="pick_blue_object",
            conditions=[
                partial(object_grabbed, object="pitcher"),
                partial(object_above, object="pitcher", reference_object="table", z_margin=0.05),
            ],
            logical="all",
            score=1.0
        )
    ]
