# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, object_picked_up
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class PickDrillTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_picked_up, params={"object": "cordless_drill", "surface": "table", "distance": 0.03})

@dataclass
class PickDrillTask(Task):
    contact_object_list = ["cordless_drill", "table"]
    scene = import_scene("mugs4_measuringcup_drill_bowl.usda", contact_object_list)
    terminations = PickDrillTerminations
    instruction = {
        "default": "Pick up the cordless drill.",
        "vague": "Get the drill",
        "specific": "Pick up the orange cordless electric drill and lift it off the table",
    }
    episode_length_s: int = 40
    attributes = ['semantics']
    subtasks = [
        Subtask(
            name="pick_drill",
            conditions=partial(object_grabbed, object="cordless_drill"),
            logical="all",
            score=1.0)
    ]
