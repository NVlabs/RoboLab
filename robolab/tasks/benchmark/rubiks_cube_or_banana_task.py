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
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": ["rubiks_cube", "banana"], "container": "bowl", "logical": "any", "tolerance": 0.0, "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class RubiksCubeOrBananaTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = import_scene("rubiks_cube_banana_bowl.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the cube or the banana in the bowl",
        "vague": "Choose an object and put it in the bowl",
        "specific": "Choose either the rubiks cube or the yellow banana and place it inside the bowl",
    }
    episode_length_s: int = 30
    attributes = ['conjunction']

    subtasks = [
        pick_and_place(
            object=["rubiks_cube", "banana"],
            container="bowl",
            logical="any",  # Only one object needs to be placed
            score=1.0
        )
    ]
