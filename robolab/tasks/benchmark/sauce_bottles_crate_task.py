# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_above_bottom,
    object_dropped,
    object_grabbed,
    object_in_container,
    pick_and_place,
)
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": "bbq_sauce_bottle", "container": "purple_crate", "logical": "all", "tolerance": -0.01, "require_gripper_detached": True})

@dataclass
class SauceBottlesCrateTask(Task):
    contact_object_list = ["bbq_sauce_bottle", "purple_crate", "table"]
    scene = import_scene("bottles_crate.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the red bbq sauce bottle in the crate",
        "vague": "Put the red bottle away",
        "specific": "Pick up the red BBQ sauce bottle and place it inside the wooden crate",
    }
    episode_length_s: int = 40
    attributes = ['color', 'semantics']
    subtasks = [
        pick_and_place(
            object=["bbq_sauce_bottle"],
            container="purple_crate",
            logical="all",
            score=1.0
        )
    ]
