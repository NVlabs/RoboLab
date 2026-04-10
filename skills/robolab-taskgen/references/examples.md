# Task Generation Examples

Three annotated examples of increasing complexity.

## Example 1: Simple Pick-and-Place

**Goal:** Pick up a banana and place it in a bowl.

- **Conditional:** `object_in_container` -- single object into an open-top container
- **Subtask:** `pick_and_place` composite -- tracks grab and placement
- **Attributes:** `semantics` -- requires recognizing "banana" and "bowl"

```python
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
class BananaInBowlTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "banana",
            "container": "bowl",
            "gripper_name": "gripper",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class BananaInBowlTask(Task):
    contact_object_list = ["banana", "bowl", "table"]
    scene = import_scene(os.path.join(SCENE_DIR, "banana_bowl.usda"), contact_object_list)
    terminations = BananaInBowlTerminations
    instruction = {
        "default": "Pick up the banana and place it in the bowl",
        "vague": "Put the fruit in the bowl",
        "specific": "Grasp the yellow banana and place it inside the red bowl on the table",
    }
    episode_length_s: int = 50
    attributes = ['semantics']
    subtasks = [
        pick_and_place(object=["banana"], container="bowl", logical="all", score=1.0)
    ]
```

## Example 2: Multi-Object Sorting

**Goal:** Sort objects by color into different bins.

- **Conditional:** `object_groups_in_containers` -- multiple groups, each with a target container
- **Subtask:** Multiple `pick_and_place` calls, one per sorted group
- **Attributes:** `spatial` (left/right bins), `sorting`, `color`

```python
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_groups_in_containers, pick_and_place
from robolab.core.task.task import Task

SCENE_DIR = os.path.join(os.path.dirname(__file__), "..", "scenes")


@configclass
class FoodPackingByColorTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {
                    "object": ["coffee_can"],
                    "container": "bin_a",
                    "logical": "all",
                    "require_contact_with": False,
                    "require_gripper_detached": True,
                },
                {
                    "object": ["mustard", "sugar_box"],
                    "container": "bin_b",
                    "logical": "all",
                    "require_contact_with": False,
                    "require_gripper_detached": True,
                },
            ]
        },
    )


@dataclass
class FoodPackingByColorTask(Task):
    contact_object_list = [
        "bin_a", "bin_b", "cheez_it", "chocolate_pudding",
        "coffee_can", "mustard", "sugar_box", "table",
    ]
    scene = import_scene(os.path.join(SCENE_DIR, "food_packing.usda"), contact_object_list)
    terminations = FoodPackingByColorTerminations
    instruction = {
        "default": "Pack yellow objects in the right container and blue object in the left container",
        "vague": "Sort things by color",
        "specific": "Pick up the yellow can and place it in the right container, then pick up the blue object and place it in the left container",
    }
    episode_length_s: int = 120
    attributes = ['spatial', 'sorting', 'color']
    subtasks = [
        pick_and_place(object=["mustard"], container="bin_a", logical="all", score=0.5),
        pick_and_place(object=["coffee_can"], container="bin_b", logical="all", score=0.5),
    ]
```

## Example 3: Ordered Stacking

**Goal:** Stack colored blocks in a specific order.

- **Conditional:** `stacked` with `order="bottom_to_top"` -- checks full stack order
- **Subtask:** Raw `Subtask` with `partial(stacked, ...)` -- tracks incremental pairs
- **Attributes:** `stacking`, `color`
- **Note:** Subtask scores are split across pairs (0.33 + 0.33 + 0.34 = 1.0)

```python
import os
from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import stacked
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

SCENE_DIR = os.path.join(os.path.dirname(__file__), "..", "scenes")


@configclass
class BlockStackingTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=stacked,
        params={
            "objects": ["red_block", "blue_block", "green_block", "yellow_block"],
            "order": "bottom_to_top",
        },
    )


@dataclass
class BlockStackingSpecifiedOrderTask(Task):
    contact_object_list = ["red_block", "blue_block", "green_block", "yellow_block", "table"]
    scene = import_scene(os.path.join(SCENE_DIR, "colored_blocks.usda"), contact_object_list)
    terminations = BlockStackingTerminations
    instruction = {
        "default": "Stack the blocks in the order from bottom to top: red, blue, green, yellow",
        "vague": "Stack in the order of red, blue, green, yellow",
        "specific": "Build a tower by placing the red block first, then the blue block on top, then the green, and finally the yellow block on top as a single tower",
    }
    episode_length_s: int = 90
    attributes = ['stacking', 'color']
    subtasks = [
        Subtask(
            conditions=partial(stacked, objects=["red_block", "blue_block"], order="bottom_to_top"),
            score=0.33,
        ),
        Subtask(
            conditions=partial(stacked, objects=["blue_block", "green_block"], order="bottom_to_top"),
            score=0.33,
        ),
        Subtask(
            conditions=partial(stacked, objects=["green_block", "yellow_block"], order="bottom_to_top"),
            score=0.34,
        ),
    ]
```
