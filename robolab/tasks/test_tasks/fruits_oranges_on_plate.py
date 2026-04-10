from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass

from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_on_top
from robolab.core.scenes.utils import import_scene

@configclass
class FruitsOrangesOnPlateTerminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # success = DoneTerm(
    #     func=object_on_top,
    #     params={"object": ["orange_01", "orange_02"], "reference_object": "clay_plates", "logical": "all", "require_gripper_detached": True}
    # )
# TODO: Fix this -- this is somehow not working.
@dataclass
class FruitsOrangesOnPlateTask(Task):
    contact_object_list = ["table", "lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "orange_03", "orange_04", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "redonion", "serving_bowl", "clay_plates", "wooden_spoons", "spatula", "storage_box"]
    scene = import_scene("fruits_out_of_basket_2oranges.usda", contact_object_list)
    terminations = FruitsOrangesOnPlateTerminations
    instruction: str = "Put all the oranges on the plate"
    episode_length_s: int = 800
    attributes = ['complex', 'counting', 'vague']
