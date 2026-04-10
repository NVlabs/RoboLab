from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_on_top, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class PutBowlOnShelfTopTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_on_top, params={"object": ["serving_bowl"], "reference_object": "rack_l04", "logical": "all", "require_gripper_detached": True})

@dataclass
class PutBowlOnShelfTopTask(Task):
    contact_object_list = ["ceramic_mug", "mug", "rack_l04", "serving_bowl", "utilityjug_a01"]
    scene = import_scene("shelf_mugs_jug_bowl.usda", contact_object_list)
    terminations = PutBowlOnShelfTopTerminations
    instruction: str = "Put the serving bowl anywhere on the shelf"
    episode_length_s: int = 90
    attributes = ['simple', 'semantics', 'spatial']
    subtasks = [
        pick_and_place(
            object=["serving_bowl"],
            container="rack_l04",
            logical="all",
            score=1.0
        )
    ]
