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
    success = DoneTerm(func=object_outside_of_and_on_surface, params={"object": ["spatula_01"], "container": "wireshelving_a01", "surface": "table", "logical": "all", "require_gripper_detached": True})

@dataclass
class TakeSpatulaOffShelfTask(Task):
    contact_object_list = ["table", "wireshelving_a01", "spatula_01", "plate_small", "fork_big", "fork_small", "ceramic_mug", "mug", "mug_01"]
    scene = import_scene("wire_shelf_mugs_plate_spatula.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Take the spatula off the shelf and put it on the table",
        "vague": "Take the spatula off the shelf",
        "specific": "Reach up and remove the spatula from the shelf, then place it flat on the table surface",
    }
    episode_length_s: int = 60
    attributes = ['affordance', 'spatial']
    subtasks = [
        pick_and_place_on_surface(
            object=["spatula_01"],
            surface="table",
            logical="all",
            K=2,
            score=1.0
        )
    ]
