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
            "object": ["keyboard", "remote_control", "computer_mouse", "smartphone", "glasses"],
            "container": "grey_bin",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class BlackItemsInBinTask(Task):
    contact_object_list = [
        "ceramic_mug", "glasses", "keyboard", "lizard_figurine", "marker",
        "remote_control", "smartphone", "wooden_bowl", "spoon_big",
        "computer_mouse", "yogurt_cup", "granola_bars", "grey_bin", "table"
    ]
    scene = import_scene("workdesk_bin.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the black items in the grey bin",
        "vague": "Clean up the dark items",
        "specific": "Identify all black-colored objects on the table and place them into the grey bin",
    }
    episode_length_s: int = 120
    attributes = ['color', 'sorting']
    subtasks = [
        pick_and_place(
            object=["keyboard", "remote_control", "computer_mouse", "smartphone", "glasses"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
