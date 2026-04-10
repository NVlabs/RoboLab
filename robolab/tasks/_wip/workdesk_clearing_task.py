from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from functools import partial
from robolab.core.task.task import Task
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import object_grabbed, object_dropped, object_behind
from robolab.core.scenes.utils import import_scene

@configclass
class WorkDeskClearingTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_behind, params={"object": ["remote_control", "glasses", "ceramic_mug", "marker", "rubiks_cube", "computer_mouse", "lizard_figurine"], "reference_object": "keyboard", "frame_of_reference": "robot", "mirrored": False, "logical": "all", "require_gripper_detached": True})

@dataclass
class WorkDeskClearingTask(Task):
    contact_object_list = ["ceramic_mug", "glasses", "keyboard", "lizard_figurine", "marker", "remote_control", "rubiks_cube", "smartphone", "wooden_bowl", "spoon_big", "computer_mouse", "yogurt_cup", "oatmeal_raisin_cookies", "granola_bars", "table"]
    scene = import_scene("workdesk.usda", contact_object_list)
    terminations = WorkDeskClearingTerminations
    instruction: str = "Clear the space in front of the keyboard"
    episode_length_s: int = 360
    attributes = ['extremely_complex', 'spatial', 'semantics', 'vague']
    subtasks = [
        Subtask(
            name="clear_space_in_front_of_keyboard",
            conditions={
                "remote_control": [
                    partial(object_grabbed, object="remote_control"),
                    partial(object_dropped, object="remote_control"),
                ],
                "glasses": [
                    partial(object_grabbed, object="glasses"),
                    partial(object_dropped, object="glasses"),
                ],
                "ceramic_mug": [
                    partial(object_grabbed, object="ceramic_mug"),
                    partial(object_dropped, object="ceramic_mug"),
                ],
                "marker": [
                    partial(object_grabbed, object="marker"),
                    partial(object_dropped, object="marker"),
                ],
                "foam_roller": [
                    partial(object_grabbed, object="foam_roller"),
                    partial(object_dropped, object="foam_roller"),
                ],
                "rubiks_cube": [
                    partial(object_grabbed, object="rubiks_cube"),
                    partial(object_dropped, object="rubiks_cube"),
                ],
                "computer_mouse": [
                    partial(object_grabbed, object="computer_mouse"),
                    partial(object_dropped, object="computer_mouse"),
                ],
                "lizard_figurine": [
                    partial(object_grabbed, object="lizard_figurine"),
                    partial(object_dropped, object="lizard_figurine"),
                ],
            },
            logical="all",
            score=1.0
        )
    ]
