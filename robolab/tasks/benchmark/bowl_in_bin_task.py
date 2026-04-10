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
class BowlInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": "bowl", "container": "grey_bin", "require_gripper_detached": True})

@dataclass
class BowlInBinTask(Task):
    contact_object_list = ["mustard", "bowl", "dry_erase_marker", "mug", "grey_bin", "table"]
    scene = import_scene("bin_mug_mustard_marker_bowl.usda", contact_object_list)
    terminations = BowlInBinTerminations
    instruction = {
        "default": "put the bowl in the grey bin",
        "vague": "put away bowl",
        "specific": "Pick up the bowl from the table and place it inside the grey bin",
    }
    episode_length_s: int = 60
    attributes = []
    subtasks = [
        pick_and_place(
            object=["bowl"],
            container="grey_bin",
            logical="all",
            score=1.0
        )
    ]
