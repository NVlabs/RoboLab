# Cameras

RoboLab supports two types of cameras: **scene cameras** (fixed in the world) and **robot-attached cameras** (mounted on the robot, e.g., wrist cameras). Both use IsaacLab's `TiledCameraCfg` and are passed into [environment registration](environment_registration.md) as config classes.

## Scene Cameras

Scene cameras are fixed-position cameras defined as standalone `@configclass` objects. They are independent of the robot and observe the scene from a static viewpoint. These configs can live in your own repository.

```python
# my_repo/cameras.py

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass


@configclass
class MyExternalCameraCfg:
    external_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66),
            rot=(-0.393, -0.195, 0.399, 0.805),
            convention="opengl",
        ),
    )
```

Key fields:

| Field | Description |
|-------|-------------|
| `prim_path` | Scene path; use `{ENV_REGEX_NS}/<camera_name>` for multi-env support |
| `height` / `width` | Image resolution in pixels |
| `data_types` | List of data types to capture (e.g., `["rgb"]`, `["rgb", "depth"]`) |
| `spawn` | Camera model — `PinholeCameraCfg` with focal length, aperture, etc. |
| `offset` | Camera pose: `pos` (x, y, z), `rot` (quaternion w, x, y, z), and `convention` (`"opengl"` or `"ros"`) |

### Built-in Scene Cameras

RoboLab ships several scene camera presets in `robolab/variations/camera.py`:

| Config | Attribute Name | Description |
|--------|---------------|-------------|
| `OverShoulderLeftCameraCfg` | `external_cam` | Over-the-shoulder view from the left |
| `EgocentricWideAngleCameraCfg` | `egocentric_wide_angle_camera` | Wide-angle front view |
| `EgocentricMirroredCameraCfg` | `egocentric_mirrored_camera` | Front-facing mirrored view (480×864) |
| `EgocentricMirroredWideAngleCameraCfg` | `egocentric_mirrored_wide_angle_camera` | Wide-angle front mirrored view |
| `EgocentricMirroredWideAngleHighCameraCfg` | `egocentric_mirrored_wide_angle_high_camera` | High-angle front mirrored view |

### Passing Scene Cameras to Registration

Scene cameras are passed as `camera_cfg` (a single config or a list) in your registration function (see [Environment Registration](environment_registration.md#step-2-write-a-registration-function) for the full example):

```python
from robolab.variations.camera import OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg

# Inside your register_envs() function:
auto_discover_and_create_cfgs(
    camera_cfg=[OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg],
    # ... other registration kwargs
)
```

## Robot-Attached Cameras

Robot-attached cameras (e.g., wrist cameras, head cameras) move with the robot during execution. These are defined as fields on the **robot config** rather than as standalone config classes.

> **Important:** A robot-attached camera's `prim_path` must be under the robot's prim hierarchy in the USD scene. This ensures the camera moves rigidly with the robot link it is attached to.

For example, the built-in `DroidCfg` defines a wrist camera:

```python
@configclass
class DroidCfg:
    robot = ArticulationCfg(...)

    wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074),
            rot=(-0.420, 0.570, 0.576, -0.409),
            convention="opengl",
        ),
    )
```

When defining your own robot with a wrist camera, ensure the camera's `prim_path` points to a link in your robot's USD. The `offset` is relative to that link. See [Robots](robots.md#adding-a-wrist-camera-to-your-robot) for a full example.

## Wiring Cameras to Observations

Camera names in the observation config must match the attribute names on the camera config classes. For example, if your scene camera config has an attribute `external_cam` and your robot config has `wrist_cam`, then your observation config references those same names:

```python
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg
import isaaclab.envs.mdp as mdp

@configclass
class ImageObsCfg(ObsGroup):
    external_cam = ObsTerm(
        func=mdp.observations.image,
        params={"sensor_cfg": SceneEntityCfg("external_cam"), "data_type": "rgb", "normalize": False},
    )
    wrist_cam = ObsTerm(
        func=mdp.observations.image,
        params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False
```

The `SceneEntityCfg("external_cam")` string must match the **attribute name** on the camera config class (e.g., `OverShoulderLeftCameraCfg.external_cam`).

## Camera Pose Randomization

For robustness testing, you can randomize camera poses at episode reset. See [Running Environments — Initial Condition Randomization](environment_run.md#initial-condition-randomization):

```python
from robolab.core.events.reset_camera import RandomizeCameraPoseUniform

events = RandomizeCameraPoseUniform.from_params(
    cameras=["external_cam"],
    pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
)
env, env_cfg = create_env("BananaInBowlTask", events=events)
```

## See Also

- [Robots](robots.md) — Robot definitions and wrist camera attachment
- [Environment Registration](environment_registration.md) — Wiring cameras into registered environments
- [Running Environments](environment_run.md) — Camera pose variation at runtime
