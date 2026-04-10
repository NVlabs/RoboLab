from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class NestGardenPlantersBySizeTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": ["gardenplanter_small", "gardenplanter_medium"], "container": "gardenplanter_large", "logical": "all", "require_contact_with": True, "require_gripper_detached": True})

@dataclass
class NestGardenPlantersBySizeTask(Task):
    contact_object_list = ["gardenplanter_large", "gardenplanter_small", "gardenplanter_medium", "table"]
    scene = import_scene("garden_planter.usda", contact_object_list)
    terminations = NestGardenPlantersBySizeTerminations
    instruction: str = "Nest the small planter inside the medium planter, then nest the medium inside the large planter"
    episode_length_s: int = 60
    attributes = ['moderate', 'spatial', 'size']
    subtasks = [
        pick_and_place(
            object=["gardenplanter_small"],
            container="gardenplanter_medium",
            logical="all",
            score=0.5
        ),
        pick_and_place(
            object=["gardenplanter_medium"],
            container="gardenplanter_large",
            logical="all",
            score=0.5
        )
    ]
