# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place, pick_and_place_on_surface
from robolab.core.task.task import Task


@configclass
class RecycleCartonsOnBoxTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["milk_carton", "orange_juice_carton"], "container": "cubebox_a02", "logical": "all", "require_gripper_detached": True})

@dataclass
class RecycleCartonsOnBoxTask(Task):
    contact_object_list = ["cubebox_a02", "milk_carton", "orange_juice_carton", "alphabet_soup_can", "smartphone", "mayonnaise_bottle", "ketchup_bottle", "mug", "table"]
    scene = import_scene("cartons_on_box.usda", contact_object_list)
    terminations = RecycleCartonsOnBoxTerminations
    instruction = {
        "default": "Put the cartons that can be recycled on the box",
        "vague": "Put the carton recyclables on the box",
        "specific": "Identify the food cartons that can be recycled and put them on top of the brown square box.",
    }
    episode_length_s: int = 90
    attributes = ['semantics']
    subtasks = [
        pick_and_place_on_surface(
            object=["milk_carton", "orange_juice_carton"],
            surface="cubebox_a02",
            logical="all",
            score=1.0
        )
    ]
