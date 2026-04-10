from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass

from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class ToolsSorting2BinsTerminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # success = DoneTerm(
    #     func=object_placed_in_container,
    #     params={"object": ["rubiks_cube", "banana"], "container": "bowl", "logical": "all", "tolerance": 0.0}
    # )

@dataclass
class ToolsSorting2BinsTask(Task):
    contact_object_list = ["container_f24", "container_b16", "hammer_7", "hammer_8", "cordless_drill", "spring_clamp", "table"]
    scene = import_scene("tools_container.usda", contact_object_list)
    terminations = ToolsSorting2BinsTerminations
    instruction: str = "Put hammers in one bin and everything else in the other bin"
    episode_length_s: int = 240
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
