from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_on_top, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class MugOnShelfVagueTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_on_top, params={"object": ["ceramic_mug", "mug"], "reference_object": "wireshelving_a01", "logical": "any", "require_gripper_detached": True})

@dataclass
class MugOnShelfVagueTask(Task):
    contact_object_list = ["ceramic_mug", "mug", "mug_01", "wireshelving_a01", "spatula_01", "plate_small", "fork_big", "fork_small", "table"]
    scene = import_scene("wire_shelf_mugs_plate_spatula.usda", contact_object_list)
    terminations = MugOnShelfVagueTerminations
    instruction: str = "Put a mug on shelf"
    episode_length_s: int = 60
    attributes = ['simple', 'semantics', 'spatial', 'vague']
    subtasks = [
        pick_and_place(
            object=["ceramic_mug", "mug"],
            container="wireshelving_a01",
            logical="any",
            score=1.0
        )
    ]
