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
| `RectLightSpawnCfg` (RoboLab) | Area light from a rectangle; see complete selections below |

## Lighting Variation for Robustness Testing

To evaluate policy robustness under different lighting conditions, register multiple environment variants with different `lighting_cfg` values. See `robolab/registrations/droid/auto_env_registrations_lighting_variations.py` for a complete example.

The built-in evaluation script `policies/pi0_family/run_lighting.py` runs benchmarks across lighting conditions automatically.

## See Also

- [Backgrounds](background.md) — HDR dome lights for environment backgrounds
- [Environment Registration](environment_registration.md) — Passing lighting into registered environments
- [Running Environments](environment_run.md) — Robustness evaluation scripts

## Complete lighting selections

The built-in selections bundle the main lights and the home-office HDR dome.
They assume the standard RoboLab tabletop near `z=0`. Selecting one is opt-in:
DROID keeps its original sphere and HDR default.

| Selection | Config class | Main light | HDR intensity |
| --- | --- | --- | --- |
| `sphere_default` | `SphereDefaultLightingCfg` | Original sphere, intensity 5000 | 500 |
| `ceiling_soft` | `CeilingSoftLightingCfg` | 1.5 × 1.5 m ceiling, 4000 at 5000 K | 350 |
| `ceiling_cool` | `CeilingCoolLightingCfg` | 1.5 × 1.5 m ceiling, 1800 at 7000 K | 150 |
| `window_neutral` | `WindowNeutralLightingCfg` | 1.2 × 1.5 m window, 5500 at 5600 K; ceiling fill 900 at 4500 K | 350 |
| `window_warm` | `WindowWarmLightingCfg` | Same window rig, window at 4500 K | 350 |

### Robustness lighting suites

These four additional complete selections are available through the same Python
API. They do not change any registration defaults.

| Selection | Config class | Main light | HDR intensity |
| --- | --- | --- | --- |
| `dark` | `DarkLightingCfg` | Existing dim sphere, intensity 50 | 5 |
| `harsh_shadows` | `HarshShadowsLightingCfg` | Angled distant light, intensity 3000, exposure 0; rays along (-0.6, -0.8, -1) | 20 |
| `spotlight` | `SpotlightLightingCfg` | Overhead distant source: rays along -Z, intensity 3000, exposure 0 | 200 |
| `soft_shadows` | `SoftShadowsLightingCfg` | 0.3 × 0.3 m area light, intensity 250000 at 5600 K | 100 |

Directions use world axes. Both distant lights have a 0.53° angular
size and spawn once globally, avoiding extra illumination from environment
clones. `harsh_shadows` adds an oblique downward direction and reduces HDR fill
for stronger cast shadows. `dark` reduces both sources so a bright dome cannot
wash out the low-light effect.

`soft_shadows` places a small bright source at `(0.55, 1.2, 1.1)`, aimed at the
tabletop `(0.55, 0, 0)`. Its finite area gives softer shadow edges than
`harsh_shadows`; bright highlights depend on camera angle and material roughness.
`spotlight` uses an overhead distant source. These selections do not change
materials, camera exposure, or renderer bloom. The legacy RGB light configs
remain available separately in `lighting.py`.

Register a programmable suite with distinct environment names, for example:

```python
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

for selection in ("dark", "harsh_shadows", "spotlight", "soft_shadows"):
    auto_register_droid_envs(
        task="GrabAFruitTask",
        lighting_selection=selection,
        env_postfix=f"Lighting_{selection}",
    )
```

Create one lighting variant at a time: the HDR dome and distant light are global
to the stage. Standard policy runners continue to use their existing defaults,
with no lighting CLI argument.

### Lighting choice cards

The breakfast comparison shows the original four presets under fixed scene,
camera, and renderer settings. `window_neutral` gives directional side lighting;
`ceiling_soft` gives overhead illumination; `window_warm` adds a warmer cast.
The historical `baseline` label means `sphere_default`.

![Breakfast lighting choices](images/lighting/breakfast_comparison.png)

The cooler `ceiling_cool` preset places its ceiling at `(0.55, 0, 1.9)`, aimed
at `(0.55, 0, 0)`, with normalization disabled and exposure zero.

### Selecting a preset

Policy evaluation entrypoints use the embodiment's registration defaults: the
original sphere/HDR setup for DROID. Lighting is
configured through the Python registration API when defining a custom setup.
The dedicated legacy lighting-variation experiment script defines its own sweep.

Import the classes or select by name from
`robolab.variations.lighting_selections`. The rectangle spawner uses IsaacLab's
generic `LightCfg` with the USD `RectLight` schema, so it also works on the pinned
IsaacLab 2.2 stack, which has no built-in `sim_utils.RectLightCfg`.

```python
from robolab.variations.lighting_selections import get_lighting_selection

# Pass to auto_discover_and_create_cfgs or generate_scene_env_cfg:
lighting_cfg = get_lighting_selection("window_neutral")
background_cfg = None  # The selection already includes the HDR dome.
```

Set `lighting_selection="sphere_default"` to explicitly select the original sphere
light and HDR dome. Use a different `env_postfix` for each lighting selection if
registering several in one process.

For DROID, the joint-position, absolute-IK and relative-IK registration functions
also accept `lighting_selection`:

```python
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

auto_register_droid_envs(
    task="GrabAFruitTask",
    lighting_selection="window_neutral",
)
```

The same argument works with `auto_register_droid_abs_ik_envs` and
`auto_register_droid_rel_ik_envs` in their respective registration modules.
Omitting it preserves the original lights and background behavior. A selection
cannot be combined with `randomize_background=True`, or with the joint-position
wrapper's `lighting_intensity` argument, because each preset fixes its HDR dome
and light intensities. Select lighting before creating the environment.

For other embodiments, pass the complete lighting config through the factory;
registrations that hardcode their background need the same opt-in wiring.

The breakfast comparison image is the retained documentation gallery. Source
scene assets and materials are unchanged.
