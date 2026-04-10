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
            "object": "coffee_pot",
            "container": "grey_bin",
            "require_gripper_detached": True
        },
    )


@dataclass
class CoffeePotInBinTask(Task):
    contact_object_list = [
        "grey_bin", "mug", "mustard", "bowl", "ranch_dressing",
        "bbq_sauce_bottle", "oatmeal_raisin_cookies", "canned_tuna",
        "soft_scrub", "wood_block", "coffee_pot", "bbq_sauce_bottle_01", "table"
    ]
    scene = import_scene("bin_condiments.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Put the coffee pot in the grey bin",
        "vague": "Put away the coffee pot",
        "specific": "Pick up the black speckled coffee pot and place it inside the grey bin",
    }
    episode_length_s: int = 60
    attributes = ['semantics']
    subtasks = [
        pick_and_place(
            object=["coffee_pot"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
