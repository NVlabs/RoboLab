# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_outside_of_and_on_surface, pick_and_place_on_surface
from robolab.core.task.task import Task


@configclass
class KeyboardOutOfBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_outside_of_and_on_surface,
        params={
            "object": "keyboard",
            "container": "grey_bin",
            "surface": "table",
            "require_gripper_detached": True
        },
    )


@dataclass
class KeyboardOutOfBinTask(Task):
    """Task: Take the keyboard out of the bin and put it on the table."""
    contact_object_list = [
        "ceramic_mug", "glasses", "keyboard", "lizard_figurine", "marker",
        "remote_control", "smartphone", "wooden_bowl", "spoon_big",
        "computer_mouse", "yogurt_cup", "granola_bars", "grey_bin", "table"
    ]
    scene = import_scene("workdesk_bin.usda", contact_object_list)
    terminations = KeyboardOutOfBinTerminations
    instruction = {
        "default": "Take the keyboard out of the bin and put it on the table",
        "vague": "Take the keyboard out",
        "specific": "Reach into the bin, grasp the keyboard, lift it out, and place it on the table surface",
    }
    episode_length_s: int = 60
    attributes = ['spatial']
    subtasks = [
        pick_and_place_on_surface(
            object=["keyboard"],
            surface="table",
            logical="all",
            score=1.0
        )
    ]
