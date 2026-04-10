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
class RubiksCubeAndBananaTerminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": ["rubiks_cube", "banana"], "container": "bowl", "logical": "all", "tolerance": 0.0, "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class RubiksCubeAndBananaTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = import_scene("rubiks_cube_banana_bowl.usda", contact_object_list)
    terminations = RubiksCubeAndBananaTerminations
    instruction = {
        "default": "Put the cube and the banana in the bowl",
        "vague": "Put everything in the bowl",
        "specific": "Pick up the rubiks cube and the yellow banana and place both inside the bowl",
    }
    episode_length_s: int = 60
    attributes = ['conjunction']

    # Updated to use new clean API
    subtasks = [
        pick_and_place(
            object=["rubiks_cube", "banana"],
            container="bowl",
            logical="all",  # Both objects must be placed
            score=1.0
        )
    ]
