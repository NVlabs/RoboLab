# Robots

RoboLab uses IsaacLab's `ArticulationCfg` to define robots. For details, refer to IsaacLab's documentation on robots. The robot config is passed as `robot_cfg` during RoboLab's [environment registration](environment_registration.md).

## Built-in Robots

The list of built-in robots — with images, embodiment tags, available action spaces, and per-robot
details — lives in [`robolab/robots/README.md`](../robolab/robots/README.md). That README is the
canonical robot list; this page covers how to *use* a built-in robot and how to define your own.

Each robot file defines:
- **Action configs** — Joint position, IK, or relative IK action spaces
- **Proprioception observations** — Joint positions, gripper state, EE pose
- **Contact gripper** — Prim paths for contact detection on gripper fingers

Coordinate frames for observations, actions, and recorded data are a contract,
not a convention — see [frames.md](frames.md). In short: EE observations and
recordings are relative to the robot's root link, Cartesian actions are always
interpreted in the robot-root frame, and scene/object data is env-local.

### Table Fixture

Tabletop manipulators are mounted on a table fixture. The fixture belongs to
the **robot**, not the scene: at registration time the env factory deactivates
the legacy `franka_table` prim still authored inside task scene USDs and
spawns the fixture the robot declares, at the declared pose.

A robot declares its fixture via an optional class-attribute label, read by
the env factory at registration time:

```python
from robolab.core.environments.scene_fixture import FRANKA_TABLE_FIXTURE, TableFixtureCfg

@configclass
class MyRobotCfg:
    robot = ArticulationCfg(...)

# Standard pedestal (also the default for robots without the label):
MyRobotCfg.table_fixture = FRANKA_TABLE_FIXTURE

# Robots with their own base (e.g. Galbot One Golf) — no fixture:
MyRobotCfg.table_fixture = None
```

Floor-standing robots declare the `root_z_above_ground` label in the robot
cfg: the distance from the robot root to its lowest colliders (measure it
with `debug/scripts/check_standing_contact.py`). The env factory rebases the
root onto each scene's authored `/GroundPlane` plus that offset, so the robot
stands exactly on the floor at any scene ground height. Grounds are per-scene
(`tests/test_scene_ground.py` locks them): canonical scenes author −0.697
(tabletop at the env origin); a set of legacy scenes keeps its original −0.65
ground to preserve replay compatibility with existing recordings — the task
table is a dynamic body resting on the floor, so the ground height sets the
tabletop height. Because robot, table, and objects all stand on the same
floor, their relative geometry is identical in every scene.

You can also declare your own fixture directly in the robot file — any USD,
posed relative to the env origin or the robot root:

```python
MyRobotCfg.table_fixture = TableFixtureCfg(
    usd_path="/path/to/my_pedestal.usd",
    pos=(-0.1, 0.0, 0.0),        # meters
    rot=(1.0, 0.0, 0.0, 0.0),    # quaternion (w, x, y, z)
    frame="robot",               # "robot" = relative to robot root, "origin" = env origin
)
```

### End-Effector Pose Recording

Every robot cfg **must** declare the `ee_recorder_bodies` label: a dict mapping
an HDF5 channel name to the articulation body recorded for that channel (pose
in the robot-root frame, see [frames.md](frames.md)). There is no default —
env generation fails with a `ValueError` naming the robot cfg if the label is
missing. Use `{}` to explicitly disable EE-pose recording.

```python
# Single-arm (DROID: Robotiq gripper base):
DroidCfg.ee_recorder_bodies = {"ee_pose": "base_link"}

# Bimanual (Galbot One Golf): one channel per arm
GalbotGolfFixedBaseCfg.ee_recorder_bodies = {
    "left_ee_pose": "left_arm_link7",
    "right_ee_pose": "right_arm_link7",
}
```

### Label Assignment Rules

Assign the labels **after the class definition — and after any subclasses**,
not in the class body: `@configclass` converts every class member it sees at
decoration time — even `ClassVar`-annotated or inherited ones — into a config
field. A post-definition assignment stays a plain class attribute and is
inherited by subclasses defined earlier in the file.

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

Every robot declares its gripper contact bodies in a `contact_gripper` dict and passes it
as `contact_gripper=contact_gripper` in its registration kwargs. Task conditionals check
grasp/release against these entries by name.

Each entry is one of two kinds:

- **Concrete label** — maps a name of your choosing to a prim path. RoboLab builds contact
  sensors against every task object for each concrete label.
- **Alias group** — maps a name to a *list of concrete labels*, meaning "any of these".
  Groups get no sensors of their own; they resolve to their members' sensors at query time.
  Groups must be flat (members must be concrete labels in the same dict).

Every robot **must** declare a `"gripper"` entry, because benchmark task conditionals
default to `gripper_name="gripper"`. Registration fails with a clear error if it is missing.

Single-arm robot — `"gripper"` is the one concrete body:

```python
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/my_gripper/.*finger"}
```

Bimanual robot — label each hand, and declare that a generic "gripper" check means either:

```python
contact_gripper = {
    "left": "{ENV_REGEX_NS}/robot/left_gripper_l_finger_link",
    "right": "{ENV_REGEX_NS}/robot/right_gripper_l_finger_link",
    "gripper": ["left", "right"],
}
```

Dexterous hand — label per fingertip; the same pattern scales:

```python
contact_gripper = {
    "thumb": "{ENV_REGEX_NS}/robot/thumb_distal",
    "index": "{ENV_REGEX_NS}/robot/index_distal",
    "middle": "{ENV_REGEX_NS}/robot/middle_distal",
    "ring": "{ENV_REGEX_NS}/robot/ring_distal",
    "gripper": ["thumb", "index", "middle", "ring"],
}
```

On the task side, conditionals take `gripper_name` as a label, a group, or a list of
labels — see [Task Conditionals](task_conditionals.md#gripper-names) for the any/all rules
(a group means "any member"; a list means "all of them at once"; detachment checks always
mean "touching none").

## See Also

- [Cameras](camera.md) — Camera placement (scene cameras and robot-attached)
- [Environment Registration](environment_registration.md) — Wiring robot, cameras, observations, and actions into environments
