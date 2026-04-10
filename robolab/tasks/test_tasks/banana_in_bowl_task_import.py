from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class BananaInBowlTerminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "banana", "container": "bowl", "gripper_name": "gripper", "tolerance": 0.05, "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class BananaInBowlTableTask(Task):
    contact_object_list = ["banana", "bowl", "table"]
    scene = import_scene("banana_bowl.usda", contact_object_list)
    terminations = BananaInBowlTerminations
    instruction: str = "Pick up the banana and place it in the bowl"
    episode_length_s: int = 20

    # Updated to use new clean API
    subtasks = [
        pick_and_place(
            object=["banana"],
            container="bowl",
            logical="all",
            score=1.0
        )
    ]
