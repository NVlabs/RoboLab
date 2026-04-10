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
class BigPumpkinInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": "pumpkinlarge", "container": "right_bin", "require_gripper_detached": True})

@dataclass
class BigPumpkinInBinTask(Task):
    contact_object_list = ["lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "red_onion", "whitepackerbottle_a01", "avocado01", "crabbypenholder", "milkjug_a01", "serving_bowl", "utilityjug_a03", "right_bin", "table"]
    scene = import_scene("clutter_fruit_bottle_bluebin.usda", contact_object_list)
    terminations = BigPumpkinInBinTerminations
    instruction = {
        "default": "Put the bigger pumpkin in the bin",
        "vague": "Put the big pumpkin away",
        "specific": "Identify the larger pumpkin of the two pumpkins on the table and place it in the bin",
    }
    episode_length_s: int = 60
    attributes = ['size']
    subtasks = [
        pick_and_place(
            object=["pumpkinlarge"],
            container="right_bin",
            logical="all",
            score=1.0
        )
    ]
