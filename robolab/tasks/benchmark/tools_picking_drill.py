# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_outside_of_and_on_surface, pick_and_place_on_surface
from robolab.core.task.task import Task


@configclass
class ToolsPickingDrillTerminations:
    """Termination configuration for drill picking task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_outside_of_and_on_surface,
        params={"object": ["cordless_drill"], "container": "left_bin", "surface": "table", "logical": "all", "tolerance": 0.0, "require_gripper_detached": True}
    )

@dataclass
class ToolsPickingDrillTask(Task):
    contact_object_list = ["table", "clamp", "cordless_drill", "spring_clamp", "husky_hammer", "blue_hammer", "red_hammer", "wood_hammer", "left_bin", "center_bin", "right_bin"]
    scene = import_scene("tools_picking.usda", contact_object_list)
    terminations = ToolsPickingDrillTerminations
    instruction = {
        "default": "Select the cordless drill and put it on the table",
        "vague": "Get the drill",
        "specific": "Pick up the cordless electric drill and place it on the table surface",
    }
    episode_length_s: int = 60
    attributes = ['spatial', 'semantics']

    subtasks = [
        pick_and_place_on_surface(
            object=["cordless_drill"],
            surface="table",
            logical="all",
            score=1.0
        )
    ]
