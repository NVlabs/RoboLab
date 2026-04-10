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
class PickUpGreenObjectTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_picked_up, params={"object": "frozen_vegetable_block", "surface": "table", "distance": 0.05})

@dataclass
class PickUpGreenObjectTask(Task):
    contact_object_list = ["frozen_vegetable_block", "blackandbrassbowl_large", "screwtoppail_a01", "utilityjug_a02", "table"]
    scene = import_scene("green.usda", contact_object_list)
    terminations = PickUpGreenObjectTerminations
    instruction = {
        "default": "Pick up the green vegetable block",
        "vague": "Pick up the green object",
        "specific": "Identify the green vegetable block on the table and lift it off the table",
    }
    episode_length_s: int = 30
    attributes = ['color']
    subtasks = [
        Subtask(
            name="pick_green_object",
            conditions=[
                partial(object_grabbed, object="frozen_vegetable_block"),
                partial(object_above, object="frozen_vegetable_block", reference_object="table", z_margin=0.05),
            ],
            logical="all",
            score=1.0
        )
    ]
