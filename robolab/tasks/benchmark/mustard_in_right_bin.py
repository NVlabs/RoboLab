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
            "object": "mustard",
            "container": "grey_bin_right",
            "require_gripper_detached": True
        },
    )


@dataclass
class MustardInRightBinTask(Task):
    """Task: Put the mustard in the right bin."""
    contact_object_list = ["table", "mustard", "grey_bin_right", "grey_bin_left"]
    scene = import_scene("two_bin.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the mustard in the right bin",
        "vague": "Use the right bin for mustard",
        "specific": "Pick up the mustard bottle and place it in the bin on the right side",
    }
    episode_length_s: int = 30
    attributes = ['spatial']
    subtasks = [
        pick_and_place(
            object=["mustard"],
            container="grey_bin_right",
            logical="all",
            score=1.0
        )
    ]
