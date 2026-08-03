# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from functools import partial

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, object_moved_to_container, object_outside_of
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


PIPER_FINGER_CONTACTS = [
    "left_left_finger",
    "left_right_finger",
    "right_left_finger",
    "right_right_finger",
]


@configclass
class PiperPickAndPlaceTerminations:
    """Termination configuration for the piper single-object pick-and-place task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_moved_to_container,
        params={
            "object": "rubiks_cube",
            "target_container": "place_box",
            "source_container": "pick_box",
            # Double-piper has two arms and two finger links per arm. Contact is
            # detected per finger sensor and OR'ed across this list.
            "gripper_name": PIPER_FINGER_CONTACTS,
            "tolerance": 0.05,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class PiperSingleObjectPickPlaceTask(Task):
    contact_object_list = ["rubiks_cube", "pick_box", "place_box", "table"]
    # The pick/place boxes are payload containers whose outer prims are valid
    # predicate references but not valid ContactSensor bodies. Keep them as
    # filters while only monitoring forces on the cube.
    contact_sensor_body_object_list = ["rubiks_cube"]
    scene = import_scene("piper_single_object_pick_place.usda", contact_object_list)
    terminations = PiperPickAndPlaceTerminations
    instruction: str = "Pick up the rubiks cube and place it in the box"
    episode_length_s: int = 20

    # pick_and_place() hardcodes gripper_name="gripper" internally, but the
    # double-piper robot's contact_gripper dict has no "gripper" key. Build the
    # same grab->place condition sequence directly so the four finger contact
    # sensor names are threaded through (any finger may grasp/release).
    subtasks = [
        Subtask(
            name="pick_and_place",
            conditions={
                "rubiks_cube": [
                    (
                        partial(
                            object_grabbed,
                            object="rubiks_cube",
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        0.0,
                    ),
                    (
                        partial(
                            object_outside_of,
                            object="rubiks_cube",
                            container="pick_box",
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        0.0,
                    ),
                    (
                        partial(
                            object_moved_to_container,
                            object="rubiks_cube",
                            target_container="place_box",
                            source_container="pick_box",
                            require_contact_with=False,
                            require_gripper_detached=True,
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        1.0,
                    ),
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
