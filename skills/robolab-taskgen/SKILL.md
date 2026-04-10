---
name: robolab-taskgen
description: >
  Generate RoboLab task files from natural language descriptions of robot manipulation goals.
  Use this skill when a user wants to create, write, or generate a new task definition,
  or asks how to define success conditions, terminations, subtasks, or instructions for
  a robot manipulation task.
license: CC-BY-NC-4.0
compatibility: >
  Requires a RoboLab project with access to robolab.core.task and a USD scene file.
metadata:
  author: nvidia
  version: "1.0.0"
---

# Task Generation

Generate complete RoboLab task files from natural language descriptions.

A **task** is a Python file containing a `Task` dataclass that binds a USD scene to a language instruction and termination criteria. Tasks are agnostic to the robot, observation space, and action space.

## Reference Files

The `references/` directory contains detailed documentation loaded on-demand:

- `references/conditionals.md` — Complete conditional function reference with signatures and parameters
- `references/examples.md` — Three annotated examples of increasing complexity

## Prerequisites

- A USD scene file (`.usda`) already exists with the objects needed for the task. See `docs/scene.md` for creating scenes.
- Object names used in the task must match the prim names in the USD scene.

## When Invoked

When the user invokes this skill, display the following message **verbatim**:

---

I'll help you generate a RoboLab task file. I need a few things:

1. **Scene file** — Either:
   - A filename (e.g., `banana_bowl.usda`) if the scene is in `robolab/assets/scenes/`
   - A full path to the `.usda` file if it's elsewhere
2. **Task instruction** — What should the robot do? (e.g., "Pick up the banana and place it in the bowl"). This becomes the `default` instruction; I'll generate `vague` and `specific` variants automatically unless you provide them.
3. **Episode length** — How long should the robot have to complete the task, in seconds? (e.g., 50 for a simple pick-and-place, 90-120 for multi-step tasks)
4. **Output directory** — Where should I save the task file?
   - `robolab/tasks/benchmark/`
   - `robolab/tasks/<name>/` — give me a folder name and I'll create it
   - Or specify another path

Here's an example of what I'll generate — a file called `banana_in_bowl_task.py`:

```python
# banana_in_bowl_task.py

@configclass
class BananaInBowlTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "banana", "container": "bowl", ...},
    )

@dataclass
class BananaInBowlTask(Task):
    contact_object_list = ["banana", "bowl", "table"]
    scene = import_scene("banana_bowl.usda", contact_object_list)
    terminations = BananaInBowlTerminations
    instruction = {
        "default": "Pick up the banana and place it in the bowl",
        "vague": "Put the fruit in the bowl",
        "specific": "Grasp the yellow banana and place it inside the red bowl on the table",
    }
    episode_length_s: int = 50
    subtasks = [pick_and_place(object=["banana"], container="bowl", logical="all", score=1.0)]
```

---

## After the User Provides Information

After receiving the user's input:

1. **Check for duplicate task names.** Search for existing task classes with the same name across `robolab/tasks/` and the user's output directory. If a task with the same class name already exists, warn the user and ask them to choose a different name. Do not overwrite existing task files.
2. **Always ask the user where to save the task on the first invocation.** Do not assume a default — you must wait for the user to confirm a directory before writing any file. Offer these options:
   - `robolab/tasks/benchmark/` (existing benchmark tasks)
   - `robolab/tasks/<name>/` (a new folder — ask the user for `<name>`)
   - Another path of their choosing

   Once the user has chosen a directory, reuse it for all subsequent tasks in the same session without asking again.
3. **State the termination criterion explicitly** before generating the file. For example: "Based on your description, I'm using `object_in_container` as the success condition — this checks that the object is inside the container and the gripper has released it."
4. Use the user's instruction text as the `"default"` instruction variant. Generate `"vague"` and `"specific"` variants automatically unless the user explicitly provides them.
5. Extract the object list from the scene file and the user's description.
6. Proceed with the [Step-by-Step Generation Workflow](#step-by-step-generation-workflow).

## Information to Gather

Before generating a task, collect the following:

| Field | Required | Description |
|-------|----------|-------------|
| **Scene file** | Yes | Filename (if in `robolab/assets/scenes/`) or full path to the USD scene (`.usda`) |
| **Task instruction** | Yes | What should the robot accomplish? Becomes the `default` instruction variant. |
| **Episode length** | Yes | Max seconds before timeout (simple: 30-50s, multi-object: 60-90s, complex: 90-120s) |
| **Success condition** | Auto | Inferred from the instruction. Maps to a conditional function. State explicitly to the user. |
| **Objects** | Auto | Extracted from the scene file and instruction. |
| **Subtasks** | No | Intermediate checkpoints for progress tracking |
| **Attributes** | No | Tags for categorization (see [Attribute Tags](#attribute-tags)) |

## Intent Routing

- User wants to create a new task from scratch --> Display the [When Invoked](#when-invoked) message, then follow the [Step-by-Step Workflow](#step-by-step-generation-workflow)
- User needs help choosing a conditional function --> Read `references/conditionals.md`
- User wants to see examples --> Read `references/examples.md`
- User wants to validate an existing task --> [Validation](#validation)
- User wants to organize tasks into a library --> See `docs/task_libraries.md`
- User wants to register and run tasks --> See `docs/environment_registration.md`

## Task File Template

```python
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import <CONDITIONAL_IMPORTS>
from robolab.core.task.task import Task

SCENE_DIR = os.path.join(os.path.dirname(__file__), "..", "scenes")


@configclass
class <TaskName>Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=<conditional_function>,
        params={<params_dict>},
    )


@dataclass
class <TaskName>Task(Task):
    contact_object_list = [<all_object_names>]
    scene = import_scene(os.path.join(SCENE_DIR, "<scene_file>.usda"), contact_object_list)
    terminations = <TaskName>Terminations
    instruction = {
        "default": "<clear, natural instruction>",
        "vague": "<ambiguous version>",
        "specific": "<detailed version with colors, sizes, exact locations>",
    }
    episode_length_s: int = <seconds>
    attributes = [<attribute_tags>]
    subtasks = [<optional_subtasks>]
```

### Importing Scenes

For scenes inside the RoboLab repo, pass just the filename (auto-resolved):

```python
scene = import_scene("banana_bowl.usda", contact_object_list)
```

For scenes in your own repository, use an absolute path:

```python
SCENE_DIR = os.path.join(os.path.dirname(__file__), "..", "scenes")
scene = import_scene(os.path.join(SCENE_DIR, "my_scene.usda"), contact_object_list)
```

To auto-extract the contact object list from a scene:

```python
from robolab.core.scenes.utils import import_scene_and_contact_object_list
MyScene, contact_object_list = import_scene_and_contact_object_list("/path/to/scene.usda")
```

## Step-by-Step Generation Workflow

1. **Identify the goal and objects.** List all objects the robot may touch, including surfaces like `"table"`.

2. **Choose the termination conditional.** Match the success condition to a function (see `references/conditionals.md` for the full list):
   - "Put X in Y" --> `object_in_container`
   - "Put X on Y" --> `object_on_top`
   - "Stack X on Y on Z" --> `stacked`
   - "Move X left of Y" --> `object_left_of`
   - "Sort X into Y and Z" --> `object_groups_in_containers`
   - "Take X out of Y" --> `object_outside_of`

3. **Write the terminations class.** Always include `time_out` and `success`. Set `require_gripper_detached=True` for any placement condition.

4. **Write instruction variants.** Default (clear), vague (ambiguous), specific (detailed). See [Instruction Variants](#instruction-variants).

5. **Decompose into subtasks** if multi-step. Use `pick_and_place` for standard pick-and-place; use `Subtask` with `partial` for custom conditions. See [Subtasks](#subtask-decomposition).

6. **Set episode length.** Simple tasks: 30-50s. Multi-object: 60-90s. Complex sorting/stacking: 90-120s.

7. **Choose attributes.** Tag based on the skills required. See [Attribute Tags](#attribute-tags).

8. **Assemble the task file.** Follow the template above.

## Instruction Variants

Every task should define three instruction variants:

- **`default`**: Clear, natural language instruction. This is the primary instruction used during evaluation.
  - Example: `"Pick up the banana and place it in the bowl"`
- **`vague`**: Ambiguous version that tests semantic understanding. Omit specific object names or use general terms.
  - Example: `"Put the fruit in the bowl"`
- **`specific`**: Highly detailed with colors, sizes, materials, and exact locations.
  - Example: `"Grasp the yellow banana and place it inside the red bowl on the wooden table"`

When using a dict for instructions, **omit the type annotation** to avoid dataclass mutable-default errors:

```python
# Correct:
instruction = {"default": "...", "vague": "...", "specific": "..."}

# Wrong (will cause dataclass error):
instruction: dict = {"default": "...", "vague": "...", "specific": "..."}
```

## Subtask Decomposition

Subtasks provide granular progress tracking. They are **optional** -- omit `subtasks` to disable checking.

### When to add subtasks

- The task involves multiple sequential steps (pick then place)
- The task involves multiple objects that can be tracked independently
- You want partial-credit scoring

### Using composite functions

For pick-and-place tasks, use the `pick_and_place` or `pick_and_place_on_surface` composite:

```python
from robolab.core.task.conditionals import pick_and_place

subtasks = [
    pick_and_place(object=["apple"], container="bowl", logical="all", score=1.0)
]
```

For multiple sequential pick-and-place operations:

```python
subtasks = [
    pick_and_place(object=["mustard"], container="bin_a", logical="all", score=0.5),
    pick_and_place(object=["coffee_can"], container="bin_b", logical="all", score=0.5),
]
```

### Using raw Subtask for custom conditions

For non-pick-and-place tasks (e.g., stacking), use `Subtask` directly with `functools.partial`:

```python
from functools import partial
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import stacked

subtasks = [
    Subtask(
        conditions=partial(stacked, objects=["red_block", "blue_block"], order="bottom_to_top"),
        score=0.5
    ),
    Subtask(
        conditions=partial(stacked, objects=["blue_block", "green_block"], order="bottom_to_top"),
        score=0.5
    ),
]
```

### Scoring

- Subtask scores should sum to 1.0 for the entire task.
- Each subtask's `score` represents its weight in the overall completion metric.

## Attribute Tags

Attributes categorize tasks for analysis. Choose from:

**Visual** (skill weight 0):
- `color` -- Task requires distinguishing objects by color
- `semantics` -- Task requires recognizing object types/categories
- `size` -- Task requires distinguishing objects by size

**Relational** (skill weight 1-2):
- `spatial` -- Task involves spatial reasoning (left/right/front/behind) -- weight 1
- `conjunction` -- Task involves logical AND/OR conditions -- weight 0
- `counting` -- Task requires counting objects -- weight 2
- `sorting` -- Task requires sorting objects into groups -- weight 2
- `stacking` -- Task requires stacking objects -- weight 2
- `affordance` -- Task requires understanding object affordances -- weight 2

**Procedural** (skill weight 3):
- `reorientation` -- Task requires changing object orientation -- weight 3

## After Generating the Task File

After writing the task file, display the following **next steps** message to the user verbatim, replacing `<tasks_folder>` and `<metadata_folder>` with the actual paths used:

---

Task file created! Here's what to do next:

**1. Validate the task** — checks contact objects, terminations, and scene references:
```bash
python scripts/check_tasks_valid.py --tasks-folder <tasks_folder>
```

**2. Update the task metadata registry** — regenerates JSON, CSV, and README table:
```bash
python robolab/tasks/_utils/generate_task_metadata.py \
    --tasks-folder <tasks_folder> \
    --output-folder <metadata_folder>
```

**3. Test the task in simulation** — run the empty demo (no policy) to verify the scene loads and terminations work:
```bash
python examples/demo/run_empty.py --task <TaskClassName>
```

See [Task Libraries](docs/task_libraries.md) for metadata details and [Environment Registration](docs/environment_registration.md) for how to register and run your task with a policy.

---

**Important:** These commands must be run inside the project's Docker image. See the top of `CLAUDE.md` for how to run commands in Docker.

## Further Reading

- `docs/task.md` -- Full task authoring guide
- `docs/subtask.md` -- Subtask system reference
- `docs/task_conditionals.md` -- Conditional function details
- `docs/scene.md` -- Creating USD scenes
- `docs/task_libraries.md` -- Organizing tasks, generating metadata, validation
- `docs/environment_registration.md` -- Registering tasks as runnable environments
