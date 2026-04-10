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
class PickGlassesTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_picked_up, params={"object": "glasses", "surface": "table", "distance": 0.05})


@dataclass
class PickGlassesTask(Task):
    contact_object_list = [
        "ceramic_mug", "glasses", "keyboard", "lizard_figurine", "marker",
        "remote_control", "rubiks_cube", "smartphone", "wooden_bowl",
        "spoon_big", "computer_mouse", "yogurt_cup", "oatmeal_raisin_cookies",
        "granola_bars", "table"
    ]
    scene = import_scene("workdesk.usda", contact_object_list)
    terminations = PickGlassesTerminations
    instruction = {
        "default": "Pick up the eye glasses",
        "vague": "Grab the glasses",
        "specific": "Pick up the pair of black eyeglasses that's sitting on the table",
    }
    episode_length_s: int = 30
    attributes = ['semantics']
    subtasks = [
        Subtask(
            name="pick_glasses",
            conditions=[
                partial(object_grabbed, object="glasses"),
                partial(object_above, object="glasses", reference_object="table", z_margin=0.05),
            ],
            logical="all",
            score=1.0
        )
    ]
