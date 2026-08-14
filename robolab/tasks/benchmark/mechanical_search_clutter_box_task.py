# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_dropped, object_grabbed, object_outside_of
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


TARGET_OBJECT = os.environ.get("ROBOLAB_MECHANICAL_SEARCH_TARGET", "banana")
TARGET_PLACEMENT = os.environ.get("ROBOLAB_MECHANICAL_SEARCH_TARGET_PLACEMENT", "middle")
SCENE_FILE = os.environ.get("ROBOLAB_MECHANICAL_SEARCH_SCENE", "mechanical_search_clutter_box.usda")
CONTAINER_OBJECT = "search_box"


@configclass
class MechanicalSearchClutterBoxTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_outside_of,
        params={
            "object": TARGET_OBJECT,
            "container": CONTAINER_OBJECT,
            "tolerance": 0.0,
            "require_gripper_detached": True,
        },
    )


@dataclass
class MechanicalSearchClutterBoxTask(Task):
    """Task: Find and retrieve a target object from dense clutter in a box."""

    contact_object_list = [
        "table",
        CONTAINER_OBJECT,
        TARGET_OBJECT,
        "rubiks_cube",
        "red_block",
        "green_block",
        "blue_block",
        "yellow_block",
        "tuna_can",
        "jello",
        "brick",
        "dry_erase_marker",
        "spam_can",
        "tomato_soup_can",
        "clamp",
        "scissors",
        "sugar_box",
        "mustard",
    ]
    scene = import_scene(SCENE_FILE, contact_object_list)
    terminations = MechanicalSearchClutterBoxTerminations
    instruction = {
        "default": f"Find the {TARGET_OBJECT} in the cluttered box",
        "vague": "Find the target object in the cluttered box",
        "specific": (
            f"Search through the densely packed objects in the box, grasp the {TARGET_PLACEMENT}-layer "
            f"{TARGET_OBJECT}, and pick it from the cluttered box"
        ),
    }
    episode_length_s: int = 180
    attributes = ["semantics", "spatial"]
    subtasks = [
        Subtask(
            name="target_out_of_clutter_box",
            conditions={
                TARGET_OBJECT: [
                    partial(object_grabbed, object=TARGET_OBJECT),
                    partial(
                        object_outside_of,
                        object=TARGET_OBJECT,
                        container=CONTAINER_OBJECT,
                        tolerance=0.0,
                        require_gripper_detached=True,
                    ),
                    partial(object_dropped, object=TARGET_OBJECT),
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
