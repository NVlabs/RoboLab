from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass

from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class FruitsMovingAny2Terminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": ["lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01"], "container": "serving_bowl", "logical": "choose", "K": 2, "tolerance": 0.0, "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class FruitsMovingAny2Task(Task):
    contact_object_list = ["table", "lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "redonion", "serving_bowl", "clay_plates", "wooden_bowl", "wooden_spoons", "spatula", "storage_box"]
    scene = import_scene("fruits_in_basket.usda", contact_object_list)
    terminations = FruitsMovingAny2Terminations
    instruction: str = "Move two different fruits from the plate to the serving bowl"
    episode_length_s: int = 180
    attributes = ['simple', 'semantics', 'counting']

    subtasks = [
        pick_and_place(
            object=["lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01"],
            container="serving_bowl",
            logical="choose",
            K=2,
            score=1.0
        )
    ]
