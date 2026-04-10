# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_outside_of_and_on_surface, pick_and_place_on_surface
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_outside_of_and_on_surface, params={"object": ["ceramic_mug", "mug"], "container": "rack_l04", "surface": "table", "logical": "all", "require_gripper_detached": True})

@dataclass
class TakeMugsOffOfShelfTask(Task):
    contact_object_list = ["ceramic_mug", "mug", "rack_l04", "serving_bowl", "utilityjug_a01", "table"]
    scene = import_scene("mugs_on_shelf.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Take the mugs off the shelf",
        "vague": "Clear the shelf",
        "specific": "Remove each mug from the second level of the shelf and place them on the table below",
    }
    episode_length_s: int = 180
    attributes = ['semantics', 'affordance']
    subtasks = [
        pick_and_place_on_surface(
            object=["ceramic_mug", "mug"],
            surface="table",
            logical="all",
            score=1.0
        )
    ]
