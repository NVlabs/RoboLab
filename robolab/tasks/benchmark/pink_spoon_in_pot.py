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
        params={"object": ["pink_spaghetti_spoon"], "container": "anza_medium", "logical": "all", "tolerance": 0.0, "require_gripper_detached": True}
    )

@dataclass
class PinkSpoonInPotTask(Task):
    contact_object_list = ["table", "anza_medium", "ladle", "plate_large", "plate_small", "fork_big", "fork_small", "spatula_13", "spatula_14", "spatula_15", "pink_spaghetti_spoon", "ladle_01", "red_serving_spoon", "green_serving_spoon"]
    scene = import_scene("ladle_pot.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the pink spaghetti spoon in the pot",
        "vague": "Put the pink spoon in the pot",
        "specific": "Pick up the pink spaghetti serving spoon that's on the left plate and place it inside the pot in the center of the table",
    }
    episode_length_s: int = 60
    attributes = ['color', 'affordance']

    subtasks = [
        pick_and_place(
            object=["pink_spaghetti_spoon"],
            container="anza_medium",
            logical="all",
            score=1.0
        )
    ]
