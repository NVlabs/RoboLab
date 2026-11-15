# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native light spawning and the DROID bundled-light registration contract."""
import isaaclab.sim as sim_utils
import pytest
from isaaclab.assets import AssetBaseCfg
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from pxr import Gf, UsdGeom, UsdLux

from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
from robolab.registrations.droid.auto_env_registrations_rel_ik import auto_register_droid_rel_ik_envs
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.lighting import SphereLightCfg
from robolab.variations.lighting_parameters import PRESETS
from robolab.variations.lighting_selections import LIGHTING_SELECTIONS, get_lighting_selection


@pytest.mark.parametrize("selection", LIGHTING_SELECTIONS)
def test_complete_rig_spawns_native_lights(selection):
    create_new_stage()
    stage = get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    cfg = get_lighting_selection(selection)()
    for asset in vars(cfg).values():
        if not isinstance(asset, AssetBaseCfg):
            continue
        if isinstance(asset.spawn, sim_utils.DomeLightCfg):
            # Test native schemas without requiring the asset library in a sparse worktree.
            asset.spawn.texture_file = ""
        path = asset.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        asset.spawn.func(path, asset.spawn, translation=asset.init_state.pos, orientation=asset.init_state.rot)
    lights = [prim for prim in stage.Traverse() if prim.HasAPI(UsdLux.LightAPI)]
    assert sum(prim.IsA(UsdLux.DomeLight) for prim in lights) == 1
    spec = PRESETS[selection]
    assert sum(prim.IsA(UsdLux.RectLight) for prim in lights) == len(spec["areas"])
    assert sum(prim.IsA(UsdLux.SphereLight) for prim in lights) == ("sphere" in spec)
    assert sum(prim.IsA(UsdLux.DistantLight) for prim in lights) == ("directional" in spec)
    dome = next(prim for prim in lights if prim.IsA(UsdLux.DomeLight))
    assert UsdLux.DomeLight(dome).GetIntensityAttr().Get() == spec["dome"]
    if "sphere" in spec:
        sphere = next(prim for prim in lights if prim.IsA(UsdLux.SphereLight))
        assert UsdLux.SphereLight(sphere).GetIntensityAttr().Get() == spec["sphere"]
    for area in spec["areas"]:
        key = UsdLux.RectLight(stage.GetPrimAtPath(f"/World/envs/env_0/lighting_{area['name']}"))
        assert key.GetColorTemperatureAttr().Get() == area["kelvin"]
        assert key.GetEnableColorTemperatureAttr().Get()
        assert not key.GetNormalizeAttr().Get()
        assert key.GetWidthAttr().Get() == pytest.approx(area["size"][0])
        assert key.GetHeightAttr().Get() == pytest.approx(area["size"][1])
        assert key.GetIntensityAttr().Get() == area["intensity"]
        assert key.GetExposureAttr().Get() == 0
        matrix = UsdGeom.Xformable(key).ComputeLocalToWorldTransform(0)
        direction = matrix.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
        target_direction = (Gf.Vec3d(*area["target"]) - matrix.ExtractTranslation()).GetNormalized()
        assert Gf.Dot(direction, target_direction) > 0.999999
    if "directional" in spec:
        distant = next(prim for prim in lights if prim.IsA(UsdLux.DistantLight))
        assert str(distant.GetPath()).startswith("/World/lighting_")  # One global source, not one per clone.
        light = UsdLux.DistantLight(distant)
        expected = spec["directional"]
        assert light.GetIntensityAttr().Get() == expected["intensity"]
        assert light.GetExposureAttr().Get() == expected["exposure"]
        assert light.GetAngleAttr().Get() == pytest.approx(expected["angle"])
        matrix = UsdGeom.Xformable(distant).ComputeLocalToWorldTransform(0)
        direction = matrix.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
        assert Gf.Dot(direction, Gf.Vec3d(*expected["direction"]).GetNormalized()) > 0.999999


def test_selection_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown lighting selection"):
        get_lighting_selection("windwo_neutral")


def test_selection_instances_do_not_change_default_or_each_other():
    first = get_lighting_selection("window_neutral")()
    first.dome_light.spawn.intensity = 10
    first.window_light.spawn.intensity = 10
    second = get_lighting_selection("window_neutral")()
    assert second.dome_light.spawn.intensity == 350
    assert second.window_light.spawn.intensity == 5500
    assert HomeOfficeBackgroundCfg().dome_light.spawn.intensity == 500
    assert get_lighting_selection("sphere_default")().dome_light.spawn.intensity == 500


@pytest.mark.parametrize("register", [
    auto_register_droid_envs, auto_register_droid_abs_ik_envs, auto_register_droid_rel_ik_envs,
])
def test_droid_lighting_selection_and_default_backgrounds(monkeypatch, register):
    captured = []
    monkeypatch.setattr("robolab.core.environments.factory.auto_discover_and_create_cfgs",
                        lambda **kwargs: captured.append(kwargs) or {})
    for name, cls in LIGHTING_SELECTIONS.items():
        register(task="GrabAFruitTask", cameras=[], lighting_selection=name)
        assert captured[-1]["lighting_cfg"] is cls
        assert captured[-1]["background_cfg"] is None

    register(task="GrabAFruitTask", cameras=[])
    assert captured[-1]["lighting_cfg"] is SphereLightCfg
    assert captured[-1]["background_cfg"] is HomeOfficeBackgroundCfg

    # Existing background randomization still reaches the per-task factory.
    monkeypatch.setattr("robolab.variations.backgrounds.find_background_files", lambda: ["test.exr"])
    background = object()
    monkeypatch.setattr("robolab.variations.backgrounds.generate_background_config", lambda path: background)
    register(task="GrabAFruitTask", cameras=[], randomize_background=True, background_seed=1)
    assert captured[-1]["lighting_cfg"] is SphereLightCfg
    assert captured[-1]["background_cfg"]() is background

    calls_before_errors = len(captured)
    with pytest.raises(ValueError, match="disable randomize_background"):
        register(cameras=[], lighting_selection="window_neutral", randomize_background=True)
    with pytest.raises(ValueError, match="Unknown lighting selection"):
        register(cameras=[], lighting_selection="windwo_neutral")
    assert len(captured) == calls_before_errors


def test_droid_selection_rejects_intensity_override():
    with pytest.raises(ValueError, match="omit lighting_intensity"):
        auto_register_droid_envs(cameras=[], lighting_selection="window_neutral", lighting_intensity=50)


def test_volo_forwards_lighting_selection(monkeypatch):
    from policies.volo.registration import register_volo_envs

    captured = []
    monkeypatch.setattr("robolab.core.environments.factory.auto_discover_and_create_cfgs",
                        lambda **kwargs: captured.append(kwargs) or {})
    register_volo_envs(task="GrabAFruitTask", cameras=[], lighting_selection="ceiling_cool")
    assert captured[-1]["lighting_cfg"] is get_lighting_selection("ceiling_cool")
    assert captured[-1]["background_cfg"] is None
