# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, noise

from robolab.robots.droid import ProprioceptionObservationCfg


@configclass
class ImageObsCfg(ObsGroup):
    """Observations for policy."""

    # arm_joint_pos = ObsTerm(func=arm_joint_pos)
    # gripper_pos = ObsTerm(
    #     func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
    # )
    external_cam = ObsTerm(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("external_cam"),
                "data_type": "rgb",
                "normalize": False,
                }
            )
    wrist_cam = ObsTerm(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
                }
            )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False

@configclass
class ObservationCfg:
    image_obs = ImageObsCfg()
    proprio_obs = ProprioceptionObservationCfg()
