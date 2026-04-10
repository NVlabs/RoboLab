# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_dropped, object_grabbed, object_right_of
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class RubiksCubeRightOfBowlTermination:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_right_of, params={"object": "rubiks_cube", "reference_object": "bowl", "frame_of_reference": "robot", "mirrored": False, "require_gripper_detached": True})

@dataclass
class RubiksCubeRightOfBowlTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "mug", "grey_bin", "table"]
    scene = import_scene("rubiks_cube_banana_bowl_mug_bin.usda", contact_object_list)
    terminations = RubiksCubeRightOfBowlTermination
    instruction = {
        "default": "Put the rubiks cube to the right of the bowl",
        "vague": "Put the cube right of the bowl",
        "specific": "Pick up the rubiks cube and place it on the table to the right side of the bowl",
    }
    attributes = ['spatial']
    episode_length_s: int = 30
    subtasks = [
        Subtask(
            name="rubiks_cube_right_of_bowl",
            conditions=[
                partial(object_grabbed, object="rubiks_cube"),
                partial(object_right_of, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot", mirrored=False),
                partial(object_dropped, object="rubiks_cube"),
            ],
            logical="all",
            score=1.0
        )
    ]
