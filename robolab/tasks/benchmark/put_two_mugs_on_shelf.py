# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["ceramic_mug", "mug", "mug_01"], "container": "wireshelving_a01", "require_contact_with": True, "require_gripper_detached": True, "logical": "choose", "K": 2})

@dataclass
class PutTwoMugsOnShelfTask(Task):
    contact_object_list = ["table", "wireshelving_a01", "spatula_01", "plate_small", "fork_big", "fork_small", "ceramic_mug", "mug", "mug_01"]
    scene = import_scene("wire_shelf_mugs_plate_spatula.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put two (2) mugs on the wire shelf",
        "vague": "Put the mugs on shelf",
        "specific": "Select two mugs from the right side of the table and place them on the wire shelf in front of you",
    }
    episode_length_s: int = 180
    attributes = ['affordance', 'spatial', 'counting']
    subtasks = [
        pick_and_place(
            object=["ceramic_mug", "mug", "mug_01"],
            container="wireshelving_a01",
            logical="choose",
            K=2,
            score=1.0
        )
    ]
