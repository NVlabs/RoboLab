import os
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass

from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.constants import SCENE_DIR

@configclass
class BananaBowlTableOakScene:
    """Scene configuration for banana task."""

    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(SCENE_DIR, "banana_bowl.usda"),
            activate_contact_sensors=True,
        ),
    )
    banana = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/banana",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.19, 0.08),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/bowl",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.33, -0.1, 0.11),
            rot=(0.67, -0.74, 0.0, 0.0),
        )
    )

@configclass
class BananaInBowlTerminations:
    """Termination configuration for banana task."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "banana", "container": "bowl", "gripper_name": "gripper", "tolerance": 0.05, "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class BananaInBowlTableTask(Task):
    scene = BananaBowlTableOakScene
    terminations = BananaInBowlTerminations
    contact_object_list = ["banana", "bowl", "table"]
    instruction: str = "Pick up the banana and place it in the bowl"
    episode_length_s: int = 20

    # Updated to use new clean API
    subtasks = [
        pick_and_place(
            object=["banana"],
            container="bowl",
            logical="all",
            score=1.0
        )
    ]
