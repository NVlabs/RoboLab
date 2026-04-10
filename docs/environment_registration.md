# Environment Registration

This guide covers how to register RoboLab tasks as runnable Gymnasium environments with your specific robot, observations, actions, and simulation settings. You do **not** need to modify RoboLab — registration can be done from your own repository.

## When You Need Custom Registration

You need a custom registration if:
- You're using a different [robot](robots.md) (not DROID)
- You need a different action space (e.g., end-effector control instead of joint position) — see [Robots](robots.md)
- You need custom observations (e.g., depth images, different [camera](camera.md) placements)
- You want different [lighting](lighting.md), [backgrounds](background.md), or simulation parameters (dt, decimation, render interval)

> **If you're using DROID with joint-position actions**, RoboLab ships a ready-to-use registration. You can skip to [Evaluating a New Policy](policy.md), which shows how to use the built-in registration directly.

## Step 1: Define Your Observation Config

Define which sensor data the simulator should provide to your policy. This uses IsaacLab's observation manager.

**For DROID users:** RoboLab already ships image and proprioception observation configs. If these work for your policy, you can skip this step and import them directly in Step 2:

```python
# Image observations (external camera + wrist camera)
from robolab.registrations.droid_jointpos.observations import ImageObsCfg

# Proprioception (joint positions, gripper state, EEF pose, etc.)
from robolab.robots.droid import ProprioceptionObservationCfg
```

If you need different observations (e.g., different cameras, depth data, or custom proprioception), define your own:

```python
# my_policy/observations.py

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class ImageObsCfg(ObsGroup):
    """Camera observations for your policy."""
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

The camera names (`external_cam`, `wrist_cam`) are mapped from the camera configurations you provide in Step 2. You can mix and match — use the built-in `ProprioceptionObservationCfg` from `robolab.robots.droid` with your own custom `ImageObsCfg`, or vice versa.

## Step 2: Write a Registration Function

Create a function that combines the benchmark tasks with your robot, observations, actions, and simulation settings:

```python
# my_policy/register_envs.py

from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


def register_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None):
    from robolab.core.environments.factory import auto_discover_and_create_cfgs, create_env_cfg
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.robots.droid import (
        DroidCfg,
        DroidJointPositionActionCfg,
        ProprioceptionObservationCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg, OverShoulderLeftCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    # Use robolab's built-in DROID observations, or import your own from Step 1
    from robolab.registrations.droid_jointpos.observations import ImageObsCfg
    # from my_policy.observations import ImageObsCfg  # ← use this if you defined custom observations

    # Build the full observation config
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg(),
    })

    shared_kwargs = dict(
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidJointPositionActionCfg(),
        robot_cfg=DroidCfg,
        camera_cfg=[OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),       # Physics timestep
        render_interval=8,      # Render every N physics steps
        decimation=8,           # Action repeat
        seed=1,
    )

    if task is not None:
        tasks = task if isinstance(task, list) else [task]
        for t in tasks:
            create_env_cfg(t, task_dir=TASK_DIR, env_prefix="", env_postfix="", **shared_kwargs)
    else:
        for subdir in task_dirs:
            auto_discover_and_create_cfgs(
                task_dir=TASK_DIR, task_subdirs=[subdir], pattern="*.py",
                env_prefix="", env_postfix="", **shared_kwargs,
            )
```

## Step 3: Verify

You can verify your registration works by calling `print_env_table()` after registration:

```python
from robolab.core.environments.factory import print_env_table

register_envs()
print_env_table()
```

This prints a table of all registered environments with their configurations.

> **Note:** It is recommended that you check your environments are created correctly before running any policy. You can also run `scripts/check_registered_envs.py` with your registration function.

## Registering Your Own Custom Tasks

The examples above register RoboLab's built-in benchmark tasks (from `TASK_DIR`). If you've created your own tasks (see [Tasks](task.md)), you can register them in the same function.

### Register individual tasks by file path

Use `create_env_cfg()` with the **full file path** to your task file:

```python
def register_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None):
    ...

    # Register a specific task by absolute path
    create_env_cfg(
        "/path/to/my_tasks/tasks/my_task.py",
        env_prefix="", env_postfix="",
        **shared_kwargs,
    )
```

### Auto-discover tasks from your own directory

Use `auto_discover_and_create_cfgs()` with `task_dir` pointing to your tasks folder:

```python
def register_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None):
    ...

    # Register all tasks from your own directory
    auto_discover_and_create_cfgs(
        task_dir="/path/to/my_tasks/tasks",
        task_subdirs=[""],
        pattern="*.py",
        env_prefix="", env_postfix="Custom",
        **shared_kwargs,
    )
```

You can register both benchmark tasks and your own tasks in the same function — just call `auto_discover_and_create_cfgs()` or `create_env_cfg()` multiple times. Use `env_postfix` or `add_tags` to distinguish them.

## Example Registration Files

For complete working examples inside the RoboLab repo:

- [`robolab/registrations/example/auto_env_registration.py`](../robolab/registrations/example/auto_env_registration.py) — Basic registration example
- [`robolab/registrations/droid_jointpos/auto_env_registrations.py`](../robolab/registrations/droid_jointpos/auto_env_registrations.py) — Full DROID robot registration

---

## API Reference

### Naming Conventions

| Name | Description | Example |
|------|-------------|---------|
| **Task Name** | The base task class name (from the `Task` class). Groups task variants together. | `BananaInBowlTask` |
| **Environment Name** | The environment name including the task, robot, and any configurations. Each env name uniquely identifies a task and a runnable env. This is the name you query in any eval script. | `BananaInBowlTaskHomeOffice` |
| **Config Class Name** | The generated configuration class. Always `<env_name>EnvCfg`. | `BananaInBowlTaskHomeOfficeEnvCfg` |
| **Tag** | A group of related environments for easy querying. | `pick_place`, `all` |

### Environment Name Formula

```
env_name = <env_prefix> + <TaskClassName> + <env_postfix>
```

For example, with `env_prefix=""` and `env_postfix="HomeOffice"`:
- Task class: `BananaInBowlTask`
- Environment name: `BananaInBowlTaskHomeOffice`
- Config class: `BananaInBowlTaskHomeOfficeEnvCfg`

A single **Task Name** can have multiple **Environment Names** (variants):

```
Task Name: BananaInBowlTask
├── Environment: BananaInBowlTaskHomeOffice      (HomeOffice background)
├── Environment: BananaInBowlTaskWarehouse       (Warehouse background)
└── Environment: BananaInBowlTaskBilliardHall    (BilliardHall background)
```

### Querying Environments

Use `get_envs()` to query environments. Only one argument is allowed at a time:

```python
from robolab.core.environments.factory import get_envs

# Query by task name — returns all variants for this task
envs = get_envs(task="BananaInBowlTask")
# Returns: ["BananaInBowlTaskHomeOffice", "BananaInBowlTaskWarehouse", ...]

# Query by exact environment name — returns exactly one environment
envs = get_envs(env="BananaInBowlTaskHomeOffice")
# Returns: ["BananaInBowlTaskHomeOffice"]

# Query by tag — returns all environments with the tag
envs = get_envs(tag="pick_place")
# Returns: ["BananaInBowlTaskHomeOffice", "AppleInBowlTaskHomeOffice", ...]

# Get all environments
envs = get_envs()
# Returns: all environments
```

Convenience functions for single queries:

```python
from robolab.core.environments.factory import get_envs_by_task, get_envs_by_tag

envs = get_envs_by_task("BananaInBowlTask")
envs = get_envs_by_tag("pick_place")
```

### Tags

Every environment is automatically added to the `"all"` tag. Task attributes (if defined on the Task class) are also automatically added as tags. You can add custom tags at registration time:

```python
auto_discover_and_create_cfgs(
    add_tags=["pick_place", "easy_tasks"],
    ...
)
```

### Configuration Options

Policy-specific parameters:

- `observations_cfg` — Custom observations configuration (see [Cameras — Wiring Cameras to Observations](camera.md#wiring-cameras-to-observations))
- `actions_cfg` — Custom actions configuration (see [Robots](robots.md) for built-in action configs)
- `robot_cfg` — Robot articulation, actuators, and attached sensors (see [Robots](robots.md))
- `decimation` — Decimation factor (action repeat)
- `dt` — Physics timestep
- `render_interval` — Render every N physics steps

Scene parameters (each can live in your own repository — no need to modify RoboLab):

- `camera_cfg` — Scene camera configuration, single config or list (see [Cameras](camera.md))
- `lighting_cfg` — Lighting configuration, single config or list (see [Lighting](lighting.md))
- `background_cfg` — HDR/EXR dome light background (see [Backgrounds](background.md))

## Next Steps

Once your environments are registered:

- **Run evaluation** — Follow the [Evaluating a New Policy](policy.md) guide to implement your inference client and run evaluation.
- **Manage a task library** — If you are building a collection of tasks, see [Task Libraries](task_libraries.md) for generating metadata, README tables, and statistics for your tasks.
