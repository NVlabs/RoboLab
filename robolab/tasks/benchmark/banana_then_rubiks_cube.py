# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place, wrong_object_grabbed
from robolab.core.task.task import Task


@configclass
class Terminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(func=object_in_container, params={"object": ["rubiks_cube", "banana"], "container": "bowl", "logical": "all", "tolerance": 0.0, "require_contact_with": True, "require_gripper_detached": True})

@dataclass
class BananaThenRubiksCubeTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = import_scene("rubiks_cube_banana_bowl.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "put the banana then the cube in the bowl",
        "vague": "put banana then cube away",
        "specific": "First pick up the yellow banana and place it in the bowl, then pick up the rubiks cube and place it in the bowl after",
    }
    episode_length_s: int = 60
    attributes = ['conjunction']
    subtasks = [
        pick_and_place(
            object=["banana"],
            container="bowl",
            score=1.0
        ),
        pick_and_place(
            object=["rubiks_cube"],
            container="bowl",
            score=1.0
        )
    ]
