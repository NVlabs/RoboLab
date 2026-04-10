# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class BowlStackingRightOnLeftTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": ["bowl_1"], "container": "bowl_2", "gripper_name": "gripper", "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class BowlStackingRightOnLeftTask(Task):
    contact_object_list = ["bowl_1", "bowl_2", "table"]
    scene = import_scene("bowls_2_table.usda", contact_object_list)
    terminations = BowlStackingRightOnLeftTerminations
    instruction = {
        "default": "Stack the right bowl on the left bowl",
        "vague": "Stack the right bowl on the left",
        "specific": "Pick up the bowl to the right of the robot and place it on top of the bowl to the left of the robot",
    }
    episode_length_s: int = 20
    attributes = ['spatial']
    subtasks = [
        Subtask(
            name="stack_bowl_right_on_left",
            conditions=partial(object_in_container, object=["bowl_1"], container="bowl_2", gripper_name="gripper", require_contact_with=True, require_gripper_detached=True),
            logical="all",
            score=1.0
        )
    ]
