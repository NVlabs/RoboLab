# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import stacked
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class Stack3RubiksCubeTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=stacked, params={"objects": ["rubiks_cube", "rubiks_cube_1", "rubiks_cube_2"], "order": "None"})

@dataclass
class Stack3RubiksCubeTask(Task):
    contact_object_list = ["rubiks_cube", "rubiks_cube_1", "rubiks_cube_2", "table"]
    scene = import_scene("rubiks_cube_3.usda", contact_object_list)
    terminations = Stack3RubiksCubeTerminations
    instruction = {
        "default": "Stack the rubiks cubes in a tower",
        "vague": "Stack everything in a tower",
        "specific": "Pick up the rubiks cubes and stack them vertically on top of each other on the table",
    }
    episode_length_s: int = 60
    attributes = ['stacking']

    # Progressive stacking subtasks - track intermediate stack formations
    # Since order doesn't matter, we track any 2 cubes stacked, then all 3
    subtasks = [
        Subtask(
            name="stack_any_2_cubes",
            conditions={
                "cube_0_and_1": partial(stacked, objects=["rubiks_cube", "rubiks_cube_1"], order=None),
                "cube_0_and_2": partial(stacked, objects=["rubiks_cube", "rubiks_cube_2"], order=None),
                "cube_1_and_2": partial(stacked, objects=["rubiks_cube_1", "rubiks_cube_2"], order=None),
            },
            logical="any",  # Any pair stacked counts as progress
            score=0.5
        ),
        Subtask(
            name="stack_all_3_cubes",
            conditions=partial(stacked, objects=["rubiks_cube", "rubiks_cube_1", "rubiks_cube_2"], order=None),
            score=0.5
        ),
    ]
