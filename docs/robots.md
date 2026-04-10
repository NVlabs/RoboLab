# Robots

RoboLab uses IsaacLab's `ArticulationCfg` to define robots. For details, refer to IsaacLab's documentation on robots. The robot config is passed as `robot_cfg` during RoboLab's [environment registration](environment_registration.md).

## Built-in Robot Configurations

| Config | File | USD Asset | Gripper | Notes |
|--------|------|-----------|---------|-------|
| `DroidCfg` | `robolab/robots/droid.py` | Franka + Robotiq 2F-85 | Robotiq binary | High PD gains (400/80), wrist camera attached, gravity disabled |
| `FrankaCfg` | `robolab/robots/franka.py` | Franka Panda | Panda fingers | Standard PD gains (80/4), frame transformers for EE |
| `FrankaCfg` (high PD) | `robolab/robots/franka_high_pd.py` | Franka Panda  | Panda fingers | High PD gains (400/80), gravity disabled |

Each robot file also defines:
- **Action configs** — Joint position, IK, or relative IK action spaces
- **Proprioception observations** — Joint positions, gripper state, EE pose
- **Contact gripper** — Prim paths for contact detection on gripper fingers

## Using a Built-in Robot

Import the robot config and pass it as `robot_cfg` in your registration function (see [Environment Registration](environment_registration.md#step-2-write-a-registration-function) for the full example):

```python
from robolab.robots.droid import DroidCfg, DroidJointPositionActionCfg, contact_gripper

# Inside your register_envs() function:
auto_discover_and_create_cfgs(
    robot_cfg=DroidCfg,
    actions_cfg=DroidJointPositionActionCfg(),
    contact_gripper=contact_gripper,
    # ... other registration kwargs
)
```

## Defining a Custom Robot

> [!NOTE]
> **Creating a new robot in RoboLab is exactly the same as creating one in IsaacLab.**
> You can bring over any robot configuration from IsaacLab (including all built-in configs and custom assets you've defined for IsaacLab), or create a new `ArticulationCfg`/`@configclass` robot from scratch by following the IsaacLab [asset configuration](https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html) and [robot configuration](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-second-robot-with-isaac-lab/02-robot-configuration-in-isaac-lab.html) tutorials.
>
> There are no RoboLab-specific requirements for robot definition beyond having a `robot` field of type `ArticulationCfg` inside a configclass.
>
> **If it works in IsaacLab, it will work with RoboLab (plus [one small addition](#Contact-Gripper))!**


A robot config for RoboLab is a `@configclass` with a `robot` field (an `ArticulationCfg`) and optionally sensor fields (e.g., cameras). It can live in your own repository — there is no requirement to add it to the RoboLab package.


IsaacLab ships USD assets and pre-built configurations for many robots. You can use any of these.
For how to write an `ArticulationCfg` (spawn settings, initial state, actuators, etc.), refer to IsaacLab's documentation:
- [Writing an Asset Configuration](https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html) — How to define `ArticulationCfg` with USD assets, rigid body properties, and actuators
- [Interacting with an Articulation](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/run_articulation.html) — Spawning and controlling articulated robots in simulation
- [Robot Configuration in IsaacLab](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-second-robot-with-isaac-lab/02-robot-configuration-in-isaac-lab.html) — End-to-end tutorial for configuring a new robot

The RoboLab-specific wrapper is a `@configclass` that exposes the `ArticulationCfg` as a `robot` field:

```python
# my_repo/my_robot.py

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg


@configclass
class MyRobotCfg:
    robot = ArticulationCfg(
        # See IsaacLab docs for full ArticulationCfg reference:
        # spawn, init_state, actuators, rigid_props, articulation_props, etc.
        ...
    )
```

The field **must** be named `robot` and use `prim_path="{ENV_REGEX_NS}/robot"` for multi-env compatibility.

### Adding a Wrist Camera to Your Robot

Robot-attached cameras (e.g., wrist cameras) are defined as fields on the robot config. The camera's `prim_path` must be **under the robot's USD hierarchy** — see [Cameras](camera.md) for details.

```python
from isaaclab.sensors import TiledCameraCfg

@configclass
class MyRobotCfg:
    robot = ArticulationCfg(...)

    wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/ee_link/wrist_cam",
        height=720, width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.01, -0.03, -0.07),
            rot=(-0.42, 0.57, 0.58, -0.41),
            convention="opengl",
        ),
    )
```

### Defining Actions and Proprioception

You also need to define an action config and proprioception observations that match your robot's joints. See the built-in examples:

- **Joint position actions:** `DroidJointPositionActionCfg` in `robolab/robots/droid.py`
- **IK actions:** `FrankaIKActionCfg` / `FrankaRelIKActionCfg` in `robolab/robots/franka_definitions.py`
- **Proprioception:** `ProprioceptionObservationCfg` in `robolab/robots/droid.py`

### Contact Gripper

For RoboLab, you must define the gripper contact prim paths. This highlights which grippers are "in contact" with an object.

```python
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/my_gripper/.*finger"}
```

Pass this as `contact_gripper=contact_gripper` in your registration kwargs.

## See Also

- [Cameras](camera.md) — Camera placement (scene cameras and robot-attached)
- [Environment Registration](environment_registration.md) — Wiring robot, cameras, observations, and actions into environments
