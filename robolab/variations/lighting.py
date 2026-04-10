# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


@configclass
class SphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    sphere_light = AssetBaseCfg(
        prim_path="/World/sphere",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )


@configclass
class RedSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    red_sphere_light = AssetBaseCfg(
        prim_path="/World/sphere_red",
        spawn=sim_utils.SphereLightCfg(intensity=100000, color=(1.0, 0.0, 0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
@configclass
class BlueSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    blue_sphere_light = AssetBaseCfg(
        prim_path="/World/sphere_blue",
        spawn=sim_utils.SphereLightCfg(intensity=100000, color=(0.0, 0.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
@configclass
class GreenSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    green_sphere_light = AssetBaseCfg(
        prim_path="/World/sphere_green",
        spawn=sim_utils.SphereLightCfg(intensity=100000, color=(0.0, 1.0, 0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


@configclass
class ExtremelyDimSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    sphere_light = AssetBaseCfg(
        prim_path="/World/sphere",
        spawn=sim_utils.SphereLightCfg(intensity=50),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

@configclass
class FrontDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    front_directional_light = AssetBaseCfg(
        prim_path="/World/front_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, 0.0, 0.7071, 0.0)),
    )

@configclass
class BehindDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    behind_directional_light = AssetBaseCfg(
        prim_path="/World/behind_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, 0.0, -0.7071, 0.0)),
    )

@configclass
class TopDownDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    top_down_directional_light = AssetBaseCfg(
        prim_path="/World/top_down_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=0.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

@configclass
class LeftDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    left_directional_light = AssetBaseCfg(
        prim_path="/World/left_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, -0.7071, 0.0, 0.0)),
    )

@configclass
class RightDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    right_directional_light = AssetBaseCfg(
        prim_path="/World/right_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, 0.7071, 0.0, 0.0)),
    )
