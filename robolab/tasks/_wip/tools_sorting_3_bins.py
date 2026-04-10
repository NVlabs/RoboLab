from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass

from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.scenes.utils import import_scene

@configclass
class ToolsSorting3BinsTerminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": ["husky_hammer", "blue_hammer", "red_hammer", "wood_hammer", "spring_clamp", "cordless_drill"], "container": "left_bin", "logical": "all", "tolerance": 0.0, "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class ToolsSorting3BinsTask(Task):
    contact_object_list = ["table", "clamp", "cordless_drill", "spring_clamp", "husky_hammer", "blue_hammer", "red_hammer", "wood_hammer", "left_bin", "center_bin", "right_bin"]
    scene = import_scene("tools_sorting.usda", contact_object_list)
    terminations = ToolsSorting3BinsTerminations
    instruction: str = "Put hammers in the left bin, clamps or tools in the middle bin, drills in the right bin"
    episode_length_s: int = 600
    attributes = ['complex', 'semantics', 'spatial']

    # Updated to use new clean API
    subtasks = [
        pick_and_place(
            object=["rubiks_cube", "banana"],
            container="bowl",
            logical="all",  # Both objects must be placed
            score=1.0
        )
    ]
