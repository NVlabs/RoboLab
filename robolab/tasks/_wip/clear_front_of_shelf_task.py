from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class ClearFrontOfShelfTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["milkjug_a01", "blackandbrassbowl_large", "gardenplanter_large"], "container": "sm_rack_m01", "logical": "all", "require_contact_with": True, "require_gripper_detached": True})

@dataclass
class ClearFrontOfShelfTask(Task):
    contact_object_list = ["sm_rack_m01", "spatula_05", "spoon_big", "spoon_small", "fork_big", "fork_small", "milkjug_a01", "blackandbrassbowl_large", "gardenplanter_large", "table"]
    scene = import_scene("front_of_shelf.usda", contact_object_list)
    terminations = ClearFrontOfShelfTerminations
    instruction: str = "Clear the table area in front of the shelf by moving those objects onto the shelf"
    episode_length_s: int = 240
    attributes = ['complex', 'spatial', 'semantics', 'specific']
    subtasks = [
        pick_and_place(
            object=["milkjug_a01", "blackandbrassbowl_large", "gardenplanter_large"],
            container="sm_rack_m01",
            logical="all",
            score=1.0
        )
    ]
