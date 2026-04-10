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
            "object": "apple_01",
            "container": "plasticpail_a02",
            "require_gripper_detached": True
        },
    )


@dataclass
class ThrowAwayAppleTask(Task):
    """Task: Throw away the apple."""
    contact_object_list = [
        "ceramic_mug", "glasses", "keyboard", "marker", "remote_control",
        "smartphone", "wooden_bowl", "spoon_big", "computer_mouse",
        "yogurt_cup", "pitcher", "plasticpail_a02", "apple_01", "table"
    ]
    scene = import_scene("workdesk_snacks.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Throw away the apple",
        "vague": "Toss the apple",
        "specific": "Pick up the red apple from the table and drop it into the grey bin to discard it",
    }
    episode_length_s: int = 60
    attributes = ['semantics']
    subtasks = [
        pick_and_place(
            object=["apple_01"],
            container="plasticpail_a02",
            logical="all",
            score=1.0
        )
    ]
