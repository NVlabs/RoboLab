from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass

from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_in_container, pick_and_place, object_outside_of
from robolab.core.scenes.utils import import_scene

@configclass
class CookingTakeOutPotatoMasherTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_outside_of,
        params={"object": ["potato_masher"], "container": "serving_bowl", "logical": "all", "tolerance": 0.0, "require_gripper_detached": True}
    )

@dataclass
class CookingTakeOutPotatoMasherTask(Task):
    contact_object_list = ["table", "redonion", "serving_bowl", "clay_plates", "wooden_spoons", "spatula", "storage_box", "tomato_sauce_can", "measuring_cups_1", "pink_spaghetti_spoon", "spoon_1", "green_serving_spoon", "storage_box_01", "ladle", "wooden_bowl", "potato_masher"]
    scene = import_scene("cooking_table.usda", contact_object_list)
    terminations = CookingTakeOutPotatoMasherTerminations
    instruction: str = "Take out the potato masher from the bowl"
    episode_length_s: int = 60
    attributes = ['simple', 'semantics']

    # Updated to use new clean API
    # subtasks = [
    #     pick_and_place(
    #         object=["rubiks_cube", "banana"],
    #         container="bowl",
    #         logical="all",  # Both objects must be placed
    #         score=1.0
    #     )
    # ]
