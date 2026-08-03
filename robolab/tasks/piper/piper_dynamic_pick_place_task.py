# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial
import json
import os

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, object_moved_to_container, object_outside_of
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task
from robolab.tasks.piper.dynamic_scene_utils import (
    GENERATED_SCENE_ENV,
    INSTRUCTION_ENV,
    OBJECT_NAME_ENV,
    OBJECT_NAMES_ENV,
    build_instruction,
)
from robolab.tasks.piper.piper_single_object_pick_place_task import PIPER_FINGER_CONTACTS


OBJECT_NAME = os.environ.get(OBJECT_NAME_ENV, "banana")
OBJECT_NAMES = json.loads(os.environ.get(OBJECT_NAMES_ENV, f'["{OBJECT_NAME}"]'))
SCENE_PATH = os.environ.get(GENERATED_SCENE_ENV)
INSTRUCTION = os.environ.get(INSTRUCTION_ENV, build_instruction(OBJECT_NAME))

if not SCENE_PATH:
    raise RuntimeError(
        "PiperDynamicPickPlaceTask requires a generated scene. "
        "Use policies/pi0_family/run_piper.py with --dynamic-object or --dynamic-object-usd."
    )


@configclass
class PiperDynamicPickAndPlaceTerminations:
    """Termination configuration for runtime-generated Piper pick-place tasks."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_moved_to_container,
        params={
            "object": OBJECT_NAME,
            "target_container": "place_box",
            "source_container": "pick_box",
            "gripper_name": PIPER_FINGER_CONTACTS,
            "tolerance": 0.05,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class PiperDynamicPickPlaceTask(Task):
    task_name = "PiperDynamicPickPlaceTask"
    contact_object_list = [*OBJECT_NAMES, "pick_box", "place_box", "table"]
    contact_sensor_body_object_list = [OBJECT_NAME]
    scene = import_scene(SCENE_PATH, contact_object_list)
    terminations = PiperDynamicPickAndPlaceTerminations
    instruction: str = INSTRUCTION
    episode_length_s: int = 20

    subtasks = [
        Subtask(
            name="pick_and_place",
            conditions={
                OBJECT_NAME: [
                    (
                        partial(
                            object_grabbed,
                            object=OBJECT_NAME,
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        0.0,
                    ),
                    (
                        partial(
                            object_outside_of,
                            object=OBJECT_NAME,
                            container="pick_box",
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        0.0,
                    ),
                    (
                        partial(
                            object_moved_to_container,
                            object=OBJECT_NAME,
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
