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
class BananasInCrateTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["banana", "banana_01", "banana_02", "banana_03", "banana_04"], "container": "purple_crate", "logical": "choose", "K": 2, "require_gripper_detached": True, 'require_contact_with': False})

@dataclass
class BananasInCrateTask(Task):
    contact_object_list = ["banana", "banana_01", "banana_02", "banana_03", "banana_04", "purple_crate", "table"] # In this scene, banana_01 is already in the crate.
    scene = import_scene("bananas_5_in_crate.usda", contact_object_list)
    terminations = BananasInCrateTerminations
    instruction = {
        "default": "Put 2 bananas in the crate",
        "vague": "Put 2 bananas in the container",
        "specific": "Put 2 bananas in the purple crate. Make sure there are exactly 2 (two) bananas in the crate.",
    }
    episode_length_s: int = 60
    attributes = ['counting']
    subtasks = [
        pick_and_place(
            object=["banana", "banana_02", "banana_03", "banana_04"],
            container="purple_crate",
            logical="choose",
            K=1,
            score=1.0
        )
    ]
