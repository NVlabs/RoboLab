# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, object_moved_to_container, object_outside_of
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task
from robolab.tasks.piper.piper_single_object_pick_place_task import PIPER_FINGER_CONTACTS


@configclass
class PiperBananaPickAndPlaceTerminations:
    """Termination configuration for the Piper single-banana pick-and-place task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_moved_to_container,
        params={
            "object": "banana",
            "target_container": "place_box",
            "source_container": "pick_box",
            "gripper_name": PIPER_FINGER_CONTACTS,
            "tolerance": 0.05,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class PiperSingleBananaPickPlaceTask(Task):
    contact_object_list = ["banana", "pick_box", "place_box", "table"]
    contact_sensor_body_object_list = ["banana"]
    scene = import_scene("piper_single_banana_pick_place.usda", contact_object_list)
    terminations = PiperBananaPickAndPlaceTerminations
    instruction: str = "Pick up the banana and place it in the box"
    episode_length_s: int = 20

    subtasks = [
        Subtask(
            name="pick_and_place",
            conditions={
                "banana": [
                    (
                        partial(
                            object_grabbed,
                            object="banana",
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        0.0,
                    ),
                    (
                        partial(
                            object_outside_of,
                            object="banana",
                            container="pick_box",
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        0.0,
                    ),
                    (
                        partial(
                            object_moved_to_container,
                            object="banana",
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
