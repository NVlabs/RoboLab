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
class MarkerInMugTerminations:
    """Termination configuration for marker in mug task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "marker",
            "container": "ceramic_mug",
            "require_gripper_detached": True
        },
    )


@dataclass
class MarkerInMugTask(Task):
    """Task: Put the marker in the mug."""
    contact_object_list = [
        "ceramic_mug", "glasses", "keyboard", "lizard_figurine", "marker",
        "remote_control", "rubiks_cube", "smartphone", "wooden_bowl",
        "spoon_big", "computer_mouse", "yogurt_cup", "oatmeal_raisin_cookies",
        "granola_bars", "table"
    ]
    scene = import_scene("workdesk.usda", contact_object_list)
    terminations = MarkerInMugTerminations
    instruction = {
        "default": "Put the whiteboard marker in the mug",
        "vague": "Put the marker in the mug",
        "specific": "Pick up the marker and drop it vertically into the mug, make sure the marker stands upright",
    }
    episode_length_s: int = 40
    attributes = ['affordance']
    subtasks = [
        pick_and_place(
            object=["marker"],
            container="ceramic_mug",
            logical="all",
            score=1.0
        )
    ]
