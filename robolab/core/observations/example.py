# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class CameraObservationCfg:
    """Static camera observation configuration - example implementation."""
    @configclass
    class ImageObsCfg(ObsGroup):
        """Observations for policy."""
        egocentric_wide_angle_camera = ObsTerm(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("egocentric_wide_angle_camera"),
                "data_type": "rgb",
                "normalize": False,
                }
            )

        egocentric_mirrored_wide_angle_camera = ObsTerm(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("egocentric_mirrored_camera"),
                "data_type": "rgb",
                "normalize": False,
                }
            )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    image_obs: ImageObsCfg = ImageObsCfg()
