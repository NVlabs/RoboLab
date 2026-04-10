from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass

from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class OnionOnFruitTerminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # success = DoneTerm(
    #     func=object_placed_in_container,
    #     params={"object": ["rubiks_cube", "banana"], "container": "bowl", "logical": "all", "tolerance": 0.0}
    # )

@dataclass
class OnionOnFruitTask(Task):
    contact_object_list = ["table", "lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "redonion", "serving_bowl", "clay_plates", "wooden_bowl", "wooden_spoons", "spatula", "storage_box"]
    scene = import_scene("fruits_in_basket.usda", contact_object_list)
    terminations = OnionOnFruitTerminations
    instruction: str = "Stack the onion on top of the rest of the fruits"
    episode_length_s: int = 60
    attributes = ['moderate', 'semantics', 'spatial']

    # Updated to use new clean API
    # subtasks = [
    #     pick_and_place(
    #         object=["rubiks_cube", "banana"],
    #         container="bowl",
    #         logical="all",  # Both objects must be placed
    #         score=1.0
    #     )
    # ]
