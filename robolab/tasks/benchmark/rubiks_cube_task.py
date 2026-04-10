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
    success = DoneTerm(func=object_in_container, params={"object": "rubiks_cube", "container": "bowl", "require_contact_with": True, "require_gripper_detached": True})

@dataclass
class RubiksCubeTask(Task):
    contact_object_list = ["rubiks_cube", "bowl", "table"]
    scene = import_scene("rubiks_cube_bowl.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the cube in the bowl",
        "vague": "Put it in the bowl",
        "specific": "Pick up the rubiks cube that's on the table and place it inside the bowl",
    }
    episode_length_s: int = 40
    attributes = []
    subtasks = [
        pick_and_place(
            object=["rubiks_cube"],
            container="bowl",
            logical="all",
            score=1.0
        )
    ]
