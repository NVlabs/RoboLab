from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class LadleToPotTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": "ladle", "container": "anza_medium", "require_gripper_detached": True})

@dataclass
class LadleToPotTask(Task):
    contact_object_list = ["anza_medium", "ladle", "plate_large", "plate_small", "fork_big", "fork_small", "spatula_13", "spatula_14", "spatula_15", "table"]
    scene = import_scene("ladle_pot.usda", contact_object_list)
    terminations = LadleToPotTerminations
    instruction: str = "Grab ladle from holder and put in pot"
    episode_length_s: int = 40
    attributes = ['simple', 'semantics']
    subtasks = [
        pick_and_place(
            object=["ladle"],
            container="anza_medium",
            logical="all",
            score=1.0
        )
    ]
