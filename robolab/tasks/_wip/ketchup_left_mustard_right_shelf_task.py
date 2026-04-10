from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from functools import partial
from robolab.core.task.task import Task
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import object_left_of, object_right_of, object_grabbed, object_dropped
from robolab.core.scenes.utils import import_scene

@configclass
class KetchupLeftMustardRightShelfTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_left_of, params={"object": "ketchup_bottle", "reference_object": "mustard", "frame_of_reference": "robot", "mirrored": False, "require_gripper_detached": True})

@dataclass
class KetchupLeftMustardRightShelfTask(Task):
    contact_object_list = ["sm_rack_m01", "rack_l04", "ketchup_bottle", "mustard", "table"]
    scene = import_scene("shelf2_without_condiments.usda", contact_object_list)
    terminations = KetchupLeftMustardRightShelfTerminations
    instruction: str = "Place the ketchup on the left end of the shelf and the mustard on the right end"
    episode_length_s: int = 60
    attributes = ['moderate', 'spatial']
    subtasks = [
        Subtask(
            name="arrange_condiments_on_shelf",
            conditions={
                "ketchup_on_left": [
                    partial(object_grabbed, object="ketchup_bottle"),
                    partial(object_left_of, object="ketchup_bottle", reference_object="mustard", frame_of_reference="robot", mirrored=False),
                    partial(object_dropped, object="ketchup_bottle"),
                ],
                "mustard_on_right": [
                    partial(object_grabbed, object="mustard"),
                    partial(object_right_of, object="mustard", reference_object="ketchup_bottle", frame_of_reference="robot", mirrored=False),
                    partial(object_dropped, object="mustard"),
                ],
            },
            logical="all",
            score=1.0
        )
    ]
