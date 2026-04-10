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
class WhiteMugsInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["mug", "mug_01"], "container": "grey_bin", "logical": "all", "require_gripper_detached": True})

@dataclass
class WhiteMugsInBinTask(Task):
    contact_object_list = ["mug", "mug_01", "grey_bin", "banana_near", "banana_far", "rubiks_cube_middle", "rubiks_cube_top", "rubiks_cube_bottom", "ketchup_bottle", "table"]
    scene = import_scene("mugs2_bananas2_ketchup_rubiks3_bin.usda", contact_object_list)
    terminations = WhiteMugsInBinTerminations
    instruction = {
        "specific": "Put the two white mugs in the grey bin",
        "default": "Clean up the white mugs",
        "vague": "Put away all mugs",
    }
    episode_length_s: int = 60
    attributes = ['color']
    subtasks = [
        pick_and_place(
            object=["mug", "mug_01"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
