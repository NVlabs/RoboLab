from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from functools import partial
from robolab.core.task.task import Task
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import object_right_of, object_grabbed, object_dropped, object_in_container
from robolab.core.scenes.utils import import_scene

@configclass
class MugOnShelfSpecificTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_right_of, params={"object": ["ceramic_mug", "mug", "mug_01"], "reference_object": "fork_small", "frame_of_reference": "robot", "mirrored": False, "logical": "any", "require_gripper_detached": True})

@dataclass
class MugOnShelfSpecificTask(Task):
    contact_object_list = ["ceramic_mug", "mug", "mug_01", "wireshelving_a01", "spatula_01", "plate_small", "fork_big", "fork_small", "table"]
    scene = import_scene("wire_shelf_mugs_plate_spatula.usda", contact_object_list)
    terminations = MugOnShelfSpecificTerminations
    instruction: str = "Put a mug on shelf to the right of the small fork"
    episode_length_s: int = 180
    attributes = ['spatial', 'complex', 'vague']
    subtasks = [
        Subtask(
            name="mug_right_of_plate_on_shelf",
            conditions={
                "ceramic_mug": [
                    partial(object_grabbed, object="ceramic_mug"),
                    partial(object_in_container, object="ceramic_mug", container="wireshelving_a01"),
                    partial(object_right_of, object="ceramic_mug", reference_object="fork_small", frame_of_reference="robot", mirrored=False),
                    partial(object_dropped, object="ceramic_mug"),
                ],
                "mug": [
                    partial(object_grabbed, object="mug"),
                    partial(object_in_container, object="mug", container="wireshelving_a01"),
                    partial(object_right_of, object="mug", reference_object="fork_small", frame_of_reference="robot", mirrored=False),
                    partial(object_dropped, object="mug"),
                ],
                "mug_01": [
                    partial(object_grabbed, object="mug_01"),
                    partial(object_in_container, object="mug_01", container="wireshelving_a01"),
                    partial(object_right_of, object="mug_01", reference_object="fork_small", frame_of_reference="robot", mirrored=False),
                    partial(object_dropped, object="mug_01"),
                ],
            },
            logical="any",
            score=1.0
        )
    ]
