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
class JugsOnShelfTerminations:
    """Termination configuration for jugs on shelf task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["utilityjug_a01", "utilityjug_a02"],
            "container": "large_storage_rack",
            "logical": "all",
            "require_gripper_detached": True
        },
    )


@dataclass
class JugsOnShelfTask(Task):
    """Task: Put all the jugs on the shelf."""
    contact_object_list = [
        "large_storage_rack", "whitepackerbottle_a01", "whitepackerbottle_a02",
        "utilityjug_a01", "utilityjug_a02", "squarepail_a01", "plasticpail_a02",
        "whitepackerbottle_a03", "table"
    ]
    scene = import_scene("shelf_with_cleaning_products.usda", contact_object_list)
    terminations = JugsOnShelfTerminations
    instruction = {
        "default": "Put all the jugs on the shelf",
        "vague": "Put the jugs on the shelf",
        "specific": "Pick up both jugs from the table and place them on the shelf",
    }
    episode_length_s: int = 120
    attributes = ['semantics']
    subtasks = [
        pick_and_place(
            object=["utilityjug_a01", "utilityjug_a02"],
            container="large_storage_rack",
            logical="all",
            score=1.0
        )
    ]
