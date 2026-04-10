# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, object_on_top, pick_and_place
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_on_top,
        params={
            "object": ["computer_mouse"],
            "reference_object": "keyboard",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class MouseOnKeyboardTask(Task):
    contact_object_list = [
        "ceramic_mug", "glasses", "keyboard", "lizard_figurine", "marker",
        "remote_control", "rubiks_cube", "smartphone", "wooden_bowl",
        "spoon_big", "computer_mouse", "yogurt_cup", "oatmeal_raisin_cookies",
        "granola_bars", "table"
    ]
    scene = import_scene("workdesk.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the computer mouse on the keyboard",
        "vague": "Put the mouse on the keyboard",
        "specific": "Pick up the computer mouse and place it on top of the keyboard",
    }
    episode_length_s: int = 60
    attributes = ['semantics']
    subtasks = [
        pick_and_place(
            object=["computer_mouse"],
            container="keyboard",
            logical="all",
            score=1.0
        )
    ]
