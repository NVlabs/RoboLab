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
class BowlStackingLeftOnRightTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": ["bowl_2"], "container": "bowl_1", "gripper_name": "gripper", "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class BowlStackingLeftOnRightTask(Task):
    contact_object_list = ["bowl_1", "bowl_2", "table"]
    scene = import_scene("bowls_2_table.usda", contact_object_list)
    terminations = BowlStackingLeftOnRightTerminations
    instruction = {
        "default": "Stack the left bowl on the right bowl",
        "vague": "Stack the left bowl on the right",
        "specific": "Pick up the bowl to the left of the robot and place it on top of the bowl to the right of the robot",
    }
    episode_length_s: int = 20
    attributes = ['spatial']
    subtasks = [
        Subtask(
            name="stack_bowl_left_on_right",
            conditions=partial(object_in_container, object=["bowl_2"], container="bowl_1", gripper_name="gripper", require_contact_with=True, require_gripper_detached=True),
            logical="all",
            score=1.0
        )
    ]
