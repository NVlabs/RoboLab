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
    """Termination configuration for mustard above raisin box task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_on_top,
        params={"object": "mustard_bottle", "reference_object": "raisin_box", "require_gripper_detached": True},
    )

@dataclass
class MustardAboveRaisinTask(Task):
    """Task: place the mustard on top of the raisin box."""
    contact_object_list = ["mustard_bottle", "raisin_box", "table"]
    scene = import_scene("mustard_raisin_box.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Place the mustard on the raisin box. ",
        "vague": "Put bottle on box",
        "specific": "Pick up the mustard bottle and put it on top of the raisin box. The mustard should end up directly above the raisin box.",
    }
    episode_length_s: int = 40
    attributes = ['spatial']
    subtasks = [
        pick_and_place(
            object=["mustard_bottle"],
            container="raisin_box",
            logical="all",
            score=1.0
        )
    ]
