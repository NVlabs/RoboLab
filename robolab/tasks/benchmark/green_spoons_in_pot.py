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
        params={"object": ["ladle", "ladle_01", "green_serving_spoon"], "container": "anza_medium", "logical": "all", "tolerance": 0.0, "require_gripper_detached": True}
    )

@dataclass
class GreenSpoonsInPotTask(Task):
    contact_object_list = ["table", "anza_medium", "ladle", "plate_large", "plate_small", "fork_big", "fork_small", "spatula_13", "spatula_14", "spatula_15", "pink_spaghetti_spoon", "ladle_01", "red_serving_spoon", "green_serving_spoon"]
    scene = import_scene("ladle_pot.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the green spoons in the pot",
        "vague": "Put the green utensils in the pot",
        "specific": "Pick up all green-colored spoons and place them inside the cooking pot",
    }
    episode_length_s: int = 180
    attributes = ['reorientation', 'color']

    subtasks = [
        pick_and_place(
            object=["ladle", "ladle_01", "green_serving_spoon"],
            container="anza_medium",
            logical="all",
            score=1.0
        )
    ]
