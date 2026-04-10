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
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["apple_01", "yogurt_cup"],
            "container": "plasticpail_a02",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class ThrowAwaySnacksTask(Task):
    """Task: Throw away the snacks."""
    contact_object_list = [
        "ceramic_mug", "glasses", "keyboard", "marker", "remote_control",
        "smartphone", "wooden_bowl", "spoon_big", "computer_mouse",
        "yogurt_cup", "pitcher", "plasticpail_a02", "apple_01", "table"
    ]
    scene = import_scene("workdesk_snacks.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put away the snacks in the bin",
        "vague": "Throw away the snacks",
        "specific": "Pick up each snack item from the table and drop it into the bin to discard",
    }
    episode_length_s: int = 120
    attributes = ['semantics']
    subtasks = [
        pick_and_place(
            object=["apple_01", "yogurt_cup"],
            container="plasticpail_a02",
            logical="all",
            score=1.0
        )
    ]
