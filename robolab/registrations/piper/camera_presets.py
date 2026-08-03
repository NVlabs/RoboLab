# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Piper-specific camera preset bundles for the policy's image observations.

Each preset is a list of camera config classes that feed both the scene
(``camera_cfg=``) and the image observation group (via
``generate_image_obs_from_cameras``).

Callers pass one of these lists directly to the registration function:

    from robolab.registrations.piper.camera_presets import ALL
    auto_register_piper_envs(cameras=ALL)
"""

from robolab.robots.piper import FirstPersonCameraCfg, LeftHandCameraCfg, RightHandCameraCfg

ALL = [
    LeftHandCameraCfg,
    RightHandCameraCfg,
    FirstPersonCameraCfg,
]

HANDS_ONLY = [
    LeftHandCameraCfg,
    RightHandCameraCfg,
]
