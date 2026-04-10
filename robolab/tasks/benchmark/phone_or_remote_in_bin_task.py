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
class PhoneOrRemoteInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["smartphone", "remote_control"],
            "container": "grey_bin",
            "logical": "any",
            "require_gripper_detached": True
        },
    )


@dataclass
class PhoneOrRemoteInBinTask(Task):
    """Task: Put either the phone or the remote in the bin."""
    contact_object_list = [
        "ceramic_mug", "glasses", "keyboard", "lizard_figurine", "marker",
        "remote_control", "smartphone", "wooden_bowl", "spoon_big",
        "computer_mouse", "yogurt_cup", "granola_bars", "grey_bin", "table"
    ]
    scene = import_scene("workdesk_bin.usda", contact_object_list)
    terminations = PhoneOrRemoteInBinTerminations
    instruction = {
        "default": "Put the phone or the remote in the grey bin",
        "vague": "Put the phone or the remote away",
        "specific": "Choose either the black phone or the remote control and it into the grey bin next to the keyboard",
    }
    episode_length_s: int = 60
    attributes = ['conjunction']
    subtasks = [
        pick_and_place(
            object=["smartphone", "remote_control"],
            container="grey_bin",
            logical="any",
            score=1.0
        )
    ]
