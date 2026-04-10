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
class SpoonInMugTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "spoon_big",
            "container": "ceramic_mug",
            "require_gripper_detached": True
        },
    )


@dataclass
class SpoonInMugTask(Task):
    """Task: Put the spoon in the mug."""
    contact_object_list = [
        "ceramic_mug", "glasses", "keyboard", "lizard_figurine", "marker",
        "remote_control", "smartphone", "wooden_bowl", "spoon_big",
        "computer_mouse", "yogurt_cup", "granola_bars", "grey_bin", "table"
    ]
    scene = import_scene("workdesk_bin.usda", contact_object_list)
    terminations = SpoonInMugTerminations
    instruction = {
        "default": "Put the metal spoon that's in the wooden bowl in the mug",
        "vague": "Put the spoon in the mug",
        "specific": "Take the metal spoon in the wooden bowl and place it inside the ceramic mug with the handle sticking out",
    }
    episode_length_s: int = 60
    attributes = ['affordance', 'spatial']
    subtasks = [
        pick_and_place(
            object=["spoon_big"],
            container="ceramic_mug",
            logical="all",
            score=1.0
        )
    ]
