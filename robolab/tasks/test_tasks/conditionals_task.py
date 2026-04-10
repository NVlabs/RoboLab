import os
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass

from robolab.constants import SCENE_DIR
from robolab.core.task.task import Task
from robolab.core.scenes.utils import import_scene_and_contact_object_list

@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@dataclass
class ConditionalsTask(Task):
    scene, contact_object_list = import_scene_and_contact_object_list("conditionals_test_scene.usda")
    terminations = Terminations
    instruction: str = "Test conditionals"
    episode_length_s: int = 1
