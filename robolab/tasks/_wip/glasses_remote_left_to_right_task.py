from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from functools import partial
from robolab.core.task.task import Task
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import object_left_of, object_grabbed, object_dropped
from robolab.core.scenes.utils import import_scene

@configclass
class GlassesPhoneRemoteLeftToRightTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_left_of, params={"object": ["glasses", "computer_mouse"], "reference_object": "remote_control", "frame_of_reference": "robot", "mirrored": False, "logical": "all", "require_gripper_detached": True})

@dataclass
class GlassesPhoneRemoteLeftToRightTask(Task):
    contact_object_list = ["ceramic_mug",  "glasses", "keyboard", "lizard_figurine", "marker", "remote_control", "rubiks_cube", "smartphone", "wooden_bowl", "spoon_big", "computer_mouse", "yogurt_cup", "oatmeal_raisin_cookies", "granola_bars", "table"]
    scene = import_scene("workdesk.usda", contact_object_list)
    terminations = GlassesPhoneRemoteLeftToRightTerminations
    instruction: str = "Arrange from left to right: glasses, mouse, remote control; keep other items where they are"
    episode_length_s: int = 90
    attributes = ['extremely_complex', 'spatial']
    subtasks = [
        Subtask(
            name="arrange_glasses_phone_remote",
            conditions={
                "glasses_left_of_smartphone": [
                    partial(object_grabbed, object="glasses"),
                    partial(object_left_of, object="glasses", reference_object="smartphone", frame_of_reference="robot", mirrored=False),
                    partial(object_dropped, object="glasses"),
                ],
                "smartphone_left_of_remote": [
                    partial(object_grabbed, object="computer_mouse"),
                    partial(object_left_of, object="computer_mouse", reference_object="remote_control", frame_of_reference="robot", mirrored=False),
                    partial(object_dropped, object="computer_mouse"),
                ],
            },
            logical="all",
            score=1.0
        )
    ]
