# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete lighting rigs, including one global HDR dome.

Pass one class as lighting_cfg and omit background_cfg. Area lights are scoped
per environment. Their positions assume the standard RoboLab tabletop near z=0.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from pxr import Gf

from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.lighting import ExtremelyDimSphereLightCfg, SphereLightCfg
from robolab.variations.lighting_parameters import PRESETS


@configclass
class RectLightSpawnCfg(sim_utils.LightCfg):
    """USD rectangle using IsaacLab's generic light spawner (also on IsaacLab 2.2)."""

    prim_type = "RectLight"
    width: float = 1.0
    height: float = 1.0


def _dome(selection):
    dome = HomeOfficeBackgroundCfg().dome_light
    dome.spawn.intensity = PRESETS[selection]["dome"]
    return dome


def _area(selection, name):
    spec = next(light for light in PRESETS[selection]["areas"] if light["name"] == name)
    direction = (Gf.Vec3d(*spec["target"]) - Gf.Vec3d(*spec["pos"])).GetNormalized()
    quat = Gf.Rotation(Gf.Vec3d(0, 0, -1), direction).GetQuat()
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/lighting_{name}",
        spawn=RectLightSpawnCfg(
            width=spec["size"][0], height=spec["size"][1],
            intensity=spec["intensity"], normalize=False, exposure=0.0,
            enable_color_temperature=True, color_temperature=spec["kelvin"],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(spec["pos"]), rot=(quat.GetReal(), *quat.GetImaginary()),
        ),
    )


def _directional(selection):
    spec = PRESETS[selection]["directional"]
    quat = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*spec["direction"]).GetNormalized()).GetQuat()
    return AssetBaseCfg(
        # Distant lights illuminate every environment: spawn once, like the dome.
        prim_path="/World/lighting_directional",
        spawn=sim_utils.DistantLightCfg(
            intensity=spec["intensity"], angle=spec["angle"], exposure=spec["exposure"],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(quat.GetReal(), *quat.GetImaginary())),
    )


@configclass
class SphereDefaultLightingCfg(SphereLightCfg):
    """Original sphere light plus home-office HDR at intensity 500."""

    dome_light = _dome("sphere_default")


@configclass
class CeilingSoftLightingCfg:
    """Broad overhead light at 5000 K with moderate HDR fill."""

    dome_light = _dome("ceiling_soft")
    ceiling_light = _area("ceiling_soft", "ceiling")


@configclass
class CeilingCoolLightingCfg:
    """Cool 7000 K overhead light with subdued HDR fill."""

    dome_light = _dome("ceiling_cool")
    ceiling_light = _area("ceiling_cool", "ceiling")


@configclass
class WindowNeutralLightingCfg:
    """5600 K side window plus a soft ceiling fill and HDR environment."""

    dome_light = _dome("window_neutral")
    window_light = _area("window_neutral", "window")
    ceiling_light = _area("window_neutral", "ceiling")


@configclass
class WindowWarmLightingCfg:
    """4500 K side window plus a soft ceiling fill and HDR environment."""

    dome_light = _dome("window_warm")
    window_light = _area("window_warm", "window")
    ceiling_light = _area("window_warm", "ceiling")


@configclass
class DarkLightingCfg(ExtremelyDimSphereLightCfg):
    """Legacy dim sphere with low HDR fill for low-light evaluation."""

    dome_light = _dome("dark")


@configclass
class HarshShadowsLightingCfg:
    """Angled distant light with low fill for harsh cast shadows."""

    dome_light = _dome("harsh_shadows")
    directional_light = _directional("harsh_shadows")


@configclass
class SpotlightLightingCfg:
    """Overhead spotlight selection, implemented with one global distant light."""

    dome_light = _dome("spotlight")
    directional_light = _directional("spotlight")


@configclass
class SoftShadowsLightingCfg:
    """Finite area source for softer shadow edges, with moderate HDR fill."""

    dome_light = _dome("soft_shadows")
    soft_light = _area("soft_shadows", "soft")


LIGHTING_SELECTIONS = {
    "sphere_default": SphereDefaultLightingCfg,
    "ceiling_soft": CeilingSoftLightingCfg,
    "ceiling_cool": CeilingCoolLightingCfg,
    "window_neutral": WindowNeutralLightingCfg,
    "window_warm": WindowWarmLightingCfg,
    "dark": DarkLightingCfg,
    "harsh_shadows": HarshShadowsLightingCfg,
    "spotlight": SpotlightLightingCfg,
    "soft_shadows": SoftShadowsLightingCfg,
}


def get_lighting_selection(name):
    """Return a complete scene-mixin class; reject misspellings explicitly."""
    try:
        return LIGHTING_SELECTIONS[name]
    except KeyError:
        raise ValueError(f"Unknown lighting selection {name!r}; choose from {', '.join(LIGHTING_SELECTIONS)}") from None
