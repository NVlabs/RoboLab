from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from functools import partial
from robolab.core.task.task import Task
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import object_outside_of, object_grabbed, object_dropped
from robolab.core.scenes.utils import import_scene

@configclass
class CutleryOutOfContainer1Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_outside_of,
        params={"object": ["fork_big", "fork_small", "spoon_big", "spoon_small"], "container": "sm_rack_m01", "logical": "any", "tolerance": 0.00, "require_gripper_detached": True}
    )

@dataclass
class CutleryOutOfContainer1Task(Task):
    contact_object_list = ["spatula_05", "spatula_14", "spoon_big", "spoon_small", "fork_big", "fork_small", "sm_rack_m01", "table"]
    scene = import_scene("cutlery_shelf.usda", contact_object_list)
    terminations = CutleryOutOfContainer1Terminations
    instruction: str = "Take any cutlery out of container"
    episode_length_s: int = 40
    attributes = ['simple', 'semantics', 'spatial', 'vague']

    subtasks = [
        Subtask(
            name="cutlery_out_of_container",
            conditions={
                "fork_big": [
                    partial(object_grabbed, object="fork_big"),
                    partial(object_outside_of, object="fork_big", container="sm_rack_m01", tolerance=0.00),
                    partial(object_dropped, object="fork_big"),
                ],
                "fork_small": [
                    partial(object_grabbed, object="fork_small"),
                    partial(object_outside_of, object="fork_small", container="sm_rack_m01", tolerance=0.00),
                    partial(object_dropped, object="fork_small"),
                ],
                "spoon_big": [
                    partial(object_grabbed, object="spoon_big"),
                    partial(object_outside_of, object="spoon_big", container="sm_rack_m01", tolerance=0.00),
                    partial(object_dropped, object="spoon_big"),
                ],
                "spoon_small": [
                    partial(object_grabbed, object="spoon_small"),
                    partial(object_outside_of, object="spoon_small", container="sm_rack_m01", tolerance=0.00),
                    partial(object_dropped, object="spoon_small"),
                ],
            },
            logical="any",
            score=1.0
        )
    ]
