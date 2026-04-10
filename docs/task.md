# Tasks

A **task** binds a **scene** to a **language instruction** and **termination criteria**. Tasks are agnostic to the robot and other environment settings, and can live in your own repository.

## Task Structure

A task is a Python file containing a single `Task` dataclass. It assumes a USD scene already exists (see [Scenes](scene.md) for creating scenes).

```
my_tasks/
  scenes/
    my_scene.usda
  tasks/
    my_task.py
```

### Complete Example

```python
# my_tasks/tasks/my_task.py

import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.task import Task

SCENE_DIR = os.path.join(os.path.dirname(__file__), "..", "scenes")


@configclass
class MyTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "apple",
            "container": "bowl",
            "gripper_name": "gripper",
            "require_gripper_detached": True,
        },
    )


@dataclass
class MyTask(Task):
    contact_object_list = ["apple", "bowl", "table"]
    scene = import_scene(os.path.join(SCENE_DIR, "my_scene.usda"), contact_object_list)
    terminations = MyTerminations
    instruction = {
        "default": "Pick up the apple and place it in the bowl",
        "vague": "Put the fruit in the bowl",
        "specific": "Grasp the red apple and place it inside the ceramic bowl on the table",
    }
    episode_length_s: int = 30
    attributes = ["pick_place", "semantics"]
    subtasks = [
        pick_and_place(object=["apple"], container="bowl", logical="all", score=1.0)
    ]
```

## Task Base Class Reference

```python
@dataclass
class Task:
    scene: InteractiveSceneCfg | Any = None
    instruction: str | dict[str, str] = ""
    terminations: TerminationCfg | Any = None
    subtasks: Any = None
    contact_object_list: list[str] | None = None

    episode_length_s: int = 60*10  # 10 minutes
    attributes: list[str] = None
    task_name: str = None
```

### Required fields

- **`contact_object_list`** — Objects tracked for contact sensing. Names must match prim names in the USD scene. Used downstream by contact sensors, terminations, and subtask checking.
- **`scene`** — Scene configuration from `import_scene()`. See [Scenes](scene.md).
- **`terminations`** — Success/failure conditions (see [Termination Conditions](#termination-conditions)).
- **`instruction`** — Language instruction string or dict of variants (see [Instruction Variants](#instruction-variants)).
- **`episode_length_s`** — Maximum episode duration in seconds.

### Optional fields

- **`subtasks`** — List of subtasks for granular progress tracking (see [Subtask Definition](#subtask-definition)).
- **`attributes`** — Tags for categorizing tasks (e.g., `['pick_place', 'semantics']`). Automatically added as tags during environment registration.
- **`task_name`** — Explicit name for grouping task variants. Defaults to the class name.

## Importing Scenes

Use `import_scene()` to load a USD scene. For scenes inside the RoboLab repo, pass the filename — it will be found automatically. For scenes in your own repository, use an absolute path:

```python
from robolab.core.scenes.utils import import_scene

# RoboLab built-in scene (resolved automatically)
scene = import_scene("banana_bowl.usda", contact_object_list)

# External scene (absolute path)
SCENE_DIR = os.path.join(os.path.dirname(__file__), "..", "scenes")
scene = import_scene(os.path.join(SCENE_DIR, "my_scene.usda"), contact_object_list)
```

`import_scene` automatically:
- Discovers rigid bodies (movable) and static bodies (fixed) in the USD
- Preserves exact object positions and orientations
- Generates IsaacLab `RigidObjectCfg` and `AssetBaseCfg` entries
- Enables contact sensors on all detected objects

### Auto-generating `contact_object_list`

If you don't want to manually enumerate contact objects, `import_scene_and_contact_object_list` extracts all dynamic rigid bodies from the scene automatically:

```python
from robolab.core.scenes.utils import import_scene_and_contact_object_list

MyScene, contact_object_list = import_scene_and_contact_object_list("/path/to/my_scene.usda")
# contact_object_list = ["apple", "bowl", "spoon", ...]
```

The returned list can be assigned directly to the task's `contact_object_list` field.

### Manual scene configuration

For full control, define the scene configuration directly instead of using `import_scene`:

```python
@configclass
class MyScene:
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(SCENE_DIR, "my_scene.usd"),
            activate_contact_sensors=True,
        ),
    )
    object1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene/object1",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.19, 0.08),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
```

See [Scenes](scene.md) for creating new USD scene files.

## Termination Conditions

Terminations define when the task succeeds or fails. Every task needs at least a `time_out` and a `success` condition:

```python
@configclass
class MyTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "apple", "container": "bowl", "gripper_name": "gripper", "require_gripper_detached": True},
    )
```

The `params` dict names must match the object names in `contact_object_list`. See [Available Conditional Functions](#available-conditional-functions) below for the full list.

## Instruction Variants

A task can define a single instruction or multiple variants for evaluation under different levels of ambiguity.

**Single instruction:**

```python
@dataclass
class MyTask(Task):
    instruction: str = "Pick up the banana and place it on the plate"
```

**Multiple variants:**

```python
@dataclass
class MyTask(Task):
    # Omit the type annotation when using a dict to avoid dataclass mutable-default errors.
    instruction = {
        "default": "Pick up the banana and place it on the plate",
        "vague": "Put stuff on the plate",
        "specific": "Grasp the yellow banana by its middle section and place it in the center of the white ceramic plate",
    }
```

At runtime, a single variant is selected via the `--instruction-type` flag. Resolution order:
1. If the requested `instruction_type` key exists, use it.
2. Otherwise fall back to `"default"`.
3. If neither exists, raise a `ValueError`.

Tasks with a single string instruction are fully backward compatible — `instruction_type` is ignored.

See [environment_run.md](environment_run.md) for how to select an instruction type at runtime.

## Task Variants with `task_name`

When creating variations of a task (e.g., with different randomization settings), use `task_name` to group them under a common name:

```python
@dataclass
class BananaInBowlUniformInitPose10cmTask(Task):
    contact_object_list = ["banana", "bowl", "table"]
    scene = import_scene("banana_bowl.usda", contact_object_list)
    terminations = BananaInBowlTerminations
    events = RandomizeInitPoseUniform
    instruction: str = "Pick up the banana and place it in the bowl"
    episode_length_s: int = 50
    attributes = ['specific', 'recognition']
    task_name = "BananaInBowlTask" # This will set the task_name to the original task for easier analysis.
```

`task_name` defaults to the class name automatically. Setting it explicitly is only needed to group variants under a common name — this matters in [results logging and analysis](analysis.md), where `task_name` is a field in every episode result and can be used to aggregate metrics across variants (e.g., all `*10cm`, `*20cm`, `*30cm` variants share the same `task_name`).


## Subtask Definition

Subtasks provide granular progress tracking within an episode. They are optional — omitting `subtasks` turns off subtask checking.

```python
from robolab.core.task.conditionals import pick_and_place

@dataclass
class MyTask(Task):
    ...
    subtasks = [
        pick_and_place(object=["apple"], container="bowl", logical="all", score=1.0)
    ]
```

See [Subtask Checking](subtask.md) for the full API including scoring.

## Available Conditional Functions

Imported from `robolab.core.task.conditionals`, used in termination and subtask definitions:

| Function | Description |
|----------|-------------|
| `object_in_container` | Object is inside a container |
| `object_on_top` | Object is on top of another object |
| `object_on_bottom` | Object is on the bottom of another object |
| `object_left_of` / `object_right_of` | Spatial relation relative to robot frame |
| `object_in_front_of` / `object_behind` | Spatial relation relative to robot frame |
| `object_above` / `object_below` | Vertical spatial relation |
| `object_upright` | Object is in an upright orientation |
| `object_next_to` | Object is adjacent to another object |
| `object_inside` / `object_outside_of` | Containment relations |
| `object_grabbed` / `object_dropped` | Gripper interaction state |
| `object_picked_up` | Object has been lifted |
| `stacked` | Objects are stacked |
| `pick_and_place` | Compound: grab, move, and place (for subtasks) |
| `pick_and_place_on_surface` | Compound: grab, move, and place on a surface (for subtasks) |

See [Task Conditionals](task_conditionals.md) for the full list with parameter documentation.

## Accessing Task Information at Runtime

After creating an environment with `create_env()`, you can access task information via `env_cfg`:

```python
env, env_cfg = create_env(env_name, device=device, num_envs=1)

task_name = env_cfg._task_name        # e.g., "BananaInBowlTask"
attributes = env_cfg._task_attributes  # e.g., ['specific', 'recognition']
instruction = env_cfg.instruction      # resolved language instruction string
```

## Managing a Task Library

You must validate your tasks after generation. See [Task Libraries](task_libraries.md) for organizing tasks, generating metadata, computing statistics, and [validating your tasks](task_libraries.md#validate-tasks).

## AI Workflows: Task Generation

We provide a Claude Code agent skill that you can use to help you generate task files given scenes using the `/robolab-taskgen` skill. Describe the goal, objects, and scene, and the agent will produce a complete, valid task file. See [`skills/robolab-taskgen/`](../skills/robolab-taskgen/) for details.

## Register and Run

For registration workflow, see [Environment Registration](environment_registration.md).