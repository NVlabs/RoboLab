# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_on_top, pick_and_place
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_on_top, params={"object": ["ceramic_mug", "mug"], "reference_object": "rack_l04", "logical": "all", "require_gripper_detached": True})

@dataclass
class PutMugsOnShelfTask(Task):
    contact_object_list = ["ceramic_mug", "mug", "rack_l04", "serving_bowl", "utilityjug_a01", "table"]
    scene = import_scene("shelf_mugs_jug_bowl.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the two mugs on the shelf",
        "vague": "Put the mugs on shelf",
        "specific": "Pick up the two mugs from the table and place them on the shelf in front of you",
    }
    episode_length_s: int = 180
    attributes = ['affordance', 'spatial', 'counting']
    subtasks = [
        pick_and_place(
            object=["ceramic_mug", "mug"],
            container="rack_l04",
            logical="all",
            score=1.0
        )
    ]
