from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_on_top, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class MugOnShelfMultipleTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_on_top, params={"object": ["ceramic_mug", "mug"], "reference_object": "rack_l04", "logical": "all", "require_gripper_detached": True})

@dataclass
class MugOnShelfMultipleTask(Task):
    contact_object_list = ["ceramic_mug", "mug", "rack_l04", "serving_bowl", "utilityjug_a01", "table"]
    scene = import_scene("shelf_mugs_jug_bowl.usda", contact_object_list)
    terminations = MugOnShelfMultipleTerminations
    instruction: str = "Put all the mugs on the shelf"
    episode_length_s: int = 120
    attributes = ['moderate', 'semantics', 'spatial', 'vague']
    subtasks = [
        pick_and_place(
            object=["ceramic_mug", "mug"],
            container="rack_l04",
            logical="all",
            score=1.0
        )
    ]
