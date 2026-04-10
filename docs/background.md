# Backgrounds

RoboLab uses HDR/EXR environment maps rendered as dome lights to provide realistic scene backgrounds. A background config is a `@configclass` with a `dome_light` field and is passed as `background_cfg` during [environment registration](environment_registration.md). These configs can live in your own repository.

## Built-in Backgrounds

RoboLab ships HDR/EXR background assets in `assets/backgrounds/` organized by category:

```
assets/backgrounds/
├── default/          # Default backgrounds
├── indoors/          # Indoor environments
├── outdoors/         # Outdoor environments
└── _utils/           # Background management scripts
```

### Pre-defined Background Configs

Available in `robolab/variations/backgrounds.py`:

| Config | File | Description |
|--------|------|-------------|
| `HomeOfficeBackgroundCfg` | `home_office.exr` | Default for most registrations |
| `EmptyWarehouseBackgroundCfg` | `empty_warehouse.hdr` | Industrial warehouse |
| `BilliardHallBackgroundCfg` | `billiard_hall.hdr` | Billiard hall |
| `BrownPhotoStudioBackgroundCfg` | `brown_photostudio.hdr` | Photo studio |

## Using a Background Config

Import the config and pass it as `background_cfg` in your registration function (see [Environment Registration](environment_registration.md#step-2-write-a-registration-function) for the full example):

```python
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg

# Inside your register_envs() function:
auto_discover_and_create_cfgs(
    background_cfg=HomeOfficeBackgroundCfg,
    # ... other registration kwargs
)
```

## Defining a Custom Background

A background config is a `@configclass` with a `dome_light` field that spawns a `DomeLightCfg`. It can live in your own repository.

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


@configclass
class MyBackgroundCfg:
    dome_light = AssetBaseCfg(
        prim_path="/World/background",
        spawn=sim_utils.DomeLightCfg(
            texture_file="/path/to/my_background.hdr",
            intensity=500.0,
            visible_in_primary_ray=True,
            texture_format="latlong",
        ),
    )
```

Key parameters of `DomeLightCfg`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `texture_file` | Absolute path to `.hdr` or `.exr` file | *(required)* |
| `intensity` | Light intensity | `500.0` |
| `visible_in_primary_ray` | Whether the dome is visible in camera renders | `True` |
| `texture_format` | Texture projection format | `"latlong"` |

## Generating Backgrounds Dynamically

For programmatic generation (e.g., iterating over many HDR files), use the helper functions:

```python
from robolab.variations.backgrounds import find_and_generate_background_config

# Generate a config from a specific file
bg_config = find_and_generate_background_config(
    filename="my_scene.hdr",
    folder_path="/path/to/my/backgrounds",
    intensity=600.0,
)

# Pass to your registration function
auto_discover_and_create_cfgs(background_cfg=bg_config, ...)
```

`find_and_generate_background_config` searches the given folder recursively for the named file and returns a `@configclass` ready to use as `background_cfg`.

To generate from an absolute path directly:

```python
from robolab.variations.backgrounds import generate_background_config

bg_config = generate_background_config(
    background_path="/absolute/path/to/scene.hdr",
    intensity=500.0,
)
```

## Background Variation for Robustness Testing

To evaluate policy robustness under different visual environments, register multiple environment variants with different `background_cfg` values. See `robolab/registrations/droid_jointpos/auto_env_registrations_bg_variations.py` for a complete example.

The built-in evaluation script `examples/policy/run_eval_background_variation.py` runs benchmarks across background conditions automatically.

## Using Your Own HDR/EXR Files

You can use any HDR or EXR environment map. Free sources include:

- [Poly Haven](https://polyhaven.com/hdris) — CC0-licensed HDR environment maps
- [HDRI Haven](https://hdrihaven.com/) — High-quality indoor and outdoor HDRIs

Place your `.hdr` or `.exr` files anywhere on disk and reference them by absolute path in your background config.

## See Also

- [Lighting](lighting.md) — Scene lighting (sphere, directional, and other light types)
- [Environment Registration](environment_registration.md) — Passing backgrounds into registered environments
- [Running Environments](environment_run.md) — Background variation evaluation scripts
