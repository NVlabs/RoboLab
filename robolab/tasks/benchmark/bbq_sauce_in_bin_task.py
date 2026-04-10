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
class BBQSauceInBinTerminations:
    """Termination configuration for BBQ sauce in bin task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["bbq_sauce_bottle", "bbq_sauce_bottle_01"],
            "container": "grey_bin",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class BBQSauceInBinTask(Task):
    """Task: Put the BBQ sauce bottles in the grey bin."""
    contact_object_list = [
        "grey_bin", "mug", "mustard", "bowl", "ranch_dressing",
        "bbq_sauce_bottle", "oatmeal_raisin_cookies", "canned_tuna",
        "soft_scrub", "wood_block", "coffee_pot", "bbq_sauce_bottle_01", "table"
    ]
    scene = import_scene("bin_condiments.usda", contact_object_list)
    terminations = BBQSauceInBinTerminations
    instruction = {
        "default": "Put the red BBQ sauce bottles in the grey bin",
        "vague": "Put the bbq sauce bottles away",
        "specific": "Pick up the two red BBQ sauce bottles from the table and place them into the grey bin",
    }
    episode_length_s: int = 90
    attributes = ['color', 'semantics']
    subtasks = [
        pick_and_place(
            object=["bbq_sauce_bottle", "bbq_sauce_bottle_01"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
