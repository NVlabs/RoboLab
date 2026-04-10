from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from functools import partial
from robolab.core.task.task import Task
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import object_grabbed, object_dropped, object_in_container
from robolab.core.scenes.utils import import_scene

@configclass
class MoveRedFromLeftShelfToRightShelfTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_in_container, params={"object": "ketchup_bottle", "container": "rack_l04"})

@dataclass
class MoveRedFromLeftShelfToRightShelfTask(Task):
    contact_object_list = ["sm_rack_m01", "rack_l04", "ketchup_bottle", "mustard", "table"]
    scene = import_scene("shelf2_without_condiments.usda", contact_object_list)
    terminations = MoveRedFromLeftShelfToRightShelfTerminations
    instruction: str = "Move all red objects to the right shelf; do not move non-red items or items already on the right shelf"
    episode_length_s: int = 50
    attributes = ['moderate', 'color', 'spatial', 'specific']
    subtasks = [
        Subtask(
            name="move_red_to_right_shelf",
            conditions={
                "ketchup_to_right_shelf": [
                    partial(object_grabbed, object="ketchup_bottle"),
                    partial(object_in_container, object="ketchup_bottle", container="rack_l04"),
                    partial(object_dropped, object="ketchup_bottle"),
                ],
            },
            logical="all",
            score=1.0
        )
    ]
