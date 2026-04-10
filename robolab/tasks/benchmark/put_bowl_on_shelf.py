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
class PutBowlOnShelfTopTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_on_top, params={"object": ["serving_bowl"], "reference_object": "rack_l04", "logical": "all", "require_gripper_detached": True})

@dataclass
class PutBowlOnShelfTopTask(Task):
    contact_object_list = ["ceramic_mug", "mug", "rack_l04", "serving_bowl", "utilityjug_a01", "table"]
    scene = import_scene("shelf_mugs_jug_bowl.usda", contact_object_list)
    terminations = PutBowlOnShelfTopTerminations
    instruction = {
        "default": "Put the serving bowl anywhere on the shelf in front of you",
        "vague": "Put the bowl on shelf",
        "specific": "Pick up the white serving bowl that's on the table to the right and place it on any open space on the shelf on the center of the table",
    }
    episode_length_s: int = 60
    attributes = ['spatial']
    subtasks = [
        pick_and_place(
            object=["serving_bowl"],
            container="rack_l04",
            logical="all",
            score=1.0
        )
    ]
