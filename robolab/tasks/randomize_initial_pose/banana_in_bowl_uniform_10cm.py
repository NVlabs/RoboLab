# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.events.reset_pose import reset_pose_uniform
from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.task import Task


@configclass
class RandomizeInitPoseUniform():
    """Configuration for randomizing initial pose uniformly."""
    randomize_init_pose = EventTerm(
                func=reset_pose_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
                    "velocity_range": {},
                    "asset_cfg": ['banana', 'bowl'],
                    "reset_to_default_otherwise": True,
                    "use_collision_check": True,
                }
            )

@configclass
class BananaInBowlTerminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "banana", "container": "bowl", "gripper_name": "gripper", "tolerance": 0.0, "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class BananaInBowlUniformInitPose10cmTask(Task):
    contact_object_list = ["banana", "bowl", "table"]
    scene = import_scene("banana_bowl.usda", contact_object_list)
    terminations = BananaInBowlTerminations
    events = RandomizeInitPoseUniform
    instruction: str = "Pick up the banana and place it in the bowl"
    episode_length_s: int = 50
    attributes = ['simple', 'specific', 'recognition']
    task_name = "BananaInBowlTask"

    # Updated to use new clean API
    subtasks = [
        pick_and_place(
            object=["banana"],
            container="bowl",
            logical="all",
            score=1.0
        )
    ]
