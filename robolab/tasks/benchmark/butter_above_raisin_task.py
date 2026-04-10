# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_above, object_on_top, pick_and_place
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class Terminations:
    """Termination configuration for butter above raisin box task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_on_top,
        params={"object": "butter", "reference_object": "raisin_box", "require_gripper_detached": True},
    )

@dataclass
class ButterAboveRaisinTask(Task):
    """Task: place the butter on top of the raisin box."""
    contact_object_list = ["butter", "raisin_box", "table"]
    scene = import_scene("butter_raisin_box.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "specific": "Place the butter on top of the raisin box. The butter should end up on the raisin box.",
        "vague": "Put butter on raisin box",
        "default": "Pick up the butter box and place it on top of the raisin box",
    }
    episode_length_s: int = 40
    attributes = ['spatial']
    subtasks = [
        pick_and_place(
            object=["butter"],
            container="raisin_box",
            logical="all",
            score=1.0
        )
    ]
