# Lighting

RoboLab uses IsaacLab's light spawners to configure scene lighting. Lighting configs are `@configclass` objects that add light sources to the simulation and are passed as `lighting_cfg` during [environment registration](environment_registration.md). These configs can live in your own repository.

## Built-in Lighting Configurations

RoboLab ships several lighting presets in `robolab/variations/lighting.py`:

### Sphere Lights

| Config | Intensity | Color | Position | Notes |
|--------|-----------|-------|----------|-------|
| `SphereLightCfg` | 5,000 | White | (0.0, -0.6, 0.7) | Default for most registrations |
| `ExtremelyDimSphereLightCfg` | 50 | White | (0.0, -0.6, 0.7) | For testing low-light conditions |
| `RedSphereLightCfg` | 100,000 | Red | (0.0, 0.0, 1.0) | Colored light variation |
| `GreenSphereLightCfg` | 100,000 | Green | (0.0, 0.0, 1.0) | Colored light variation |
| `BlueSphereLightCfg` | 100,000 | Blue | (0.0, 0.0, 1.0) | Colored light variation |

### Directional Lights

| Config | Direction | Intensity | Notes |
|--------|-----------|-----------|-------|
| `FrontDirectionalLightCfg` | Front | 3,000 | Facing the robot from the front |
| `BehindDirectionalLightCfg` | Behind | 3,000 | Facing the robot from behind |
| `TopDownDirectionalLightCfg` | Top-down | 3,000 | Overhead lighting |
| `LeftDirectionalLightCfg` | Left | 3,000 | Side lighting from the left |
| `RightDirectionalLightCfg` | Right | 3,000 | Side lighting from the right |

## Using a Lighting Config

Import the config and pass it as `lighting_cfg` in your registration function (see [Environment Registration](environment_registration.md#step-2-write-a-registration-function) for the full example):

```python
from robolab.variations.lighting import SphereLightCfg

# Inside your register_envs() function:
auto_discover_and_create_cfgs(
    lighting_cfg=SphereLightCfg,
    # ... other registration kwargs
)
```

## Defining Custom Lighting

A lighting config is a `@configclass` with one or more `AssetBaseCfg` fields, each spawning a light. It can live in your own repository.

### Sphere Light

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


@configclass
class MySphereLightCfg:
    my_light = AssetBaseCfg(
        prim_path="/World/my_sphere_light",
        spawn=sim_utils.SphereLightCfg(
            intensity=8000,
            color=(1.0, 0.95, 0.9),   # Warm white
            radius=0.1,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.5, 1.0)),
    )
```

### Directional (Distant) Light

```python
@configclass
class MyDirectionalLightCfg:
    directional_light = AssetBaseCfg(
        prim_path="/World/directional_light",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000,
            angle=0.53,
            exposure=3.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 5.0),
            rot=(0.7071, 0.0, 0.7071, 0.0),  # Front-facing
        ),
    )
```

### Multiple Lights

You can define multiple light fields in one config:

```python
@configclass
class MyStudioLightingCfg:
    key_light = AssetBaseCfg(
        prim_path="/World/key_light",
        spawn=sim_utils.SphereLightCfg(intensity=10000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, -0.8, 1.2)),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/fill_light",
        spawn=sim_utils.SphereLightCfg(intensity=3000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, -0.3, 0.8)),
    )
```

## Available Light Types

IsaacLab provides the following light spawners through `isaaclab.sim`:

| Spawner | Description |
|---------|-------------|
| `sim_utils.SphereLightCfg` | Point light from a sphere; parameters: `intensity`, `color`, `radius` |
| `sim_utils.DistantLightCfg` | Directional light (sun-like); parameters: `intensity`, `angle`, `exposure` |
| `sim_utils.DomeLightCfg` | Environment map light (HDR/EXR); used for [Backgrounds](background.md) |
| `sim_utils.DiskLightCfg` | Area light from a disk |
| `sim_utils.CylinderLightCfg` | Area light from a cylinder |
| `sim_utils.RectLightCfg` | Area light from a rectangle |

## Lighting Variation for Robustness Testing

To evaluate policy robustness under different lighting conditions, register multiple environment variants with different `lighting_cfg` values. See `robolab/registrations/droid_jointpos/auto_env_registrations_lighting_variations.py` for a complete example.

The built-in evaluation script `examples/policy/run_eval_lighting.py` runs benchmarks across lighting conditions automatically.

## See Also

- [Backgrounds](background.md) — HDR dome lights for environment backgrounds
- [Environment Registration](environment_registration.md) — Passing lighting into registered environments
- [Running Environments](environment_run.md) — Robustness evaluation scripts
