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
    """Termination configuration for butter above raisin box task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "raisin_box", "container": "grey_bin", "tolerance": 0.02, "require_gripper_detached": True},
    )

@dataclass
class LargerObjectRaisinBoxInBinTask(Task):
    """Task: place the butter on top of the raisin box."""
    contact_object_list = ["butter", "grey_bin", "raisin_box", "table"]
    scene = import_scene("butter_raisin_box_grey_bin.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Place the larger object in the grey bin.",
        "vague": "Put the larger object away",
        "specific": "Compare the objects and put the larger raisin box in the grey bin",
    }
    episode_length_s: int = 30
    attributes = ['size']
    subtasks = [
        pick_and_place(
            object=["raisin_box"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
