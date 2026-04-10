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
class CondimentsInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["mustard", "ranch_dressing", "bbq_sauce_bottle", "bbq_sauce_bottle_01"],
            "container": "grey_bin",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class CondimentsInBinTask(Task):
    contact_object_list = [
        "grey_bin", "mug", "mustard", "bowl", "ranch_dressing",
        "bbq_sauce_bottle", "oatmeal_raisin_cookies", "canned_tuna",
        "soft_scrub", "wood_block", "coffee_pot", "bbq_sauce_bottle_01", "table"
    ]
    scene = import_scene("bin_condiments.usda", contact_object_list)
    terminations = CondimentsInBinTerminations
    instruction = {
        "default": "Sort the sauce condiments into the grey bin",
        "vague": "Put the sauces away",
        "specific": "Pick up each sauce condiment bottle from the table and place it into the grey bin",
    }
    episode_length_s: int = 180
    attributes = ['semantics', 'sorting']
    subtasks = [
        pick_and_place(
            object=["mustard", "ranch_dressing", "bbq_sauce_bottle", "bbq_sauce_bottle_01"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
