# RoboLab Subtask System: Comprehensive Guide

This guide provides a complete reference for defining subtasks in RoboLab, covering all supported formats, logical modes, and use cases.

## Table of Contents

1. [Overview](#overview)
2. [Subtask Definitions](#subtask-definitions)
3. [The Subtask Dataclass](#the-subtask-dataclass)
4. [Condition Formats](#condition-formats)
   - [Format 1: Single Callable (Atomic Subtask)](#format-1-single-callable-atomic-subtask)
   - [Format 2: List of Callables](#format-2-list-of-callables)
   - [Format 3: List of Tuples (Callable, Score)](#format-3-list-of-tuples-callable-score)
   - [Format 4: Set of Callables](#format-4-set-of-callables)
   - [Format 5: Set of Tuples (Callable, Score)](#format-5-set-of-tuples-callable-score)
   - [Format 6: Dict with Single Callables](#format-6-dict-with-single-callables)
   - [Format 7: Dict with List of Callables](#format-7-dict-with-list-of-callables)
   - [Format 8: Dict with List of Tuples (Callable, Score)](#format-8-dict-with-list-of-tuples-callable-score)
   - [Format 9: Dict with Set of Callables](#format-9-dict-with-set-of-callables)
5. [Logical Modes](#logical-modes)
   - [Mode 1: "all" (Default)](#mode-1-all-default)
   - [Mode 2: "any"](#mode-2-any)
   - [Mode 3: "choose"](#mode-3-choose)
6. [Composite Functions](#composite-functions)
7. [Sequential Subtasks](#sequential-subtasks)
8. [Scoring System](#scoring-system)
9. [Effective Operations & Difficulty Scoring](#effective-operations--difficulty-scoring)

---

## Overview

RoboLab uses a hierarchical state machine architecture to manage complex manipulation tasks involving multiple objects and sequential execution phases.

## Subtask Definitions

Subtasks are defined as a list of subtask groups, where each subtask group can handle parallel subtask checking and composite tasks such as "and"/"any"/"choose" type subtasks. Each subtask group must be complete until the next subtask state machine step.

<p align="center">
  <img src="images/subtask_state_machine.png" alt="Subtask State Machine Diagram" width="300px">
</p>

### Key Components

```
Subtask(conditions, logical, score, K)  ← Subtask dataclass
    ↓
SubtaskStateMachine  ← Manages sequential subtasks
    ↓
ConditionalsStateMachine  ← Manages parallel conditions
    ↓
Parallel execution with regression checking
```

## The Subtask Dataclass

The `Subtask` dataclass is the core building block for defining task conditions. Each subtask must be completed before moving to the next subtask.

```python
from robolab.core.task.subtask import Subtask

@dataclass
class Subtask:
    """Self-documenting container for a group of parallel conditions."""
    conditions: Union[Callable, list, set, dict]
    score: float = 1.0
    logical: Literal["all", "any", "choose"] = "all"
    K: Optional[int] = None  # Required when logical="choose"
    name: str = "unnamed_subtask"
```

## Condition Formats

The `conditions` parameter supports 8 different formats for convenience, but will all be unified at loading time:

### Format 1: Single Callable (Atomic Subtask)

**Use Case**: Single condition to check

```python
from functools import partial
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import object_grabbed
Subtask(
    conditions=partial(object_grabbed, object='banana'),
    name="grab_banana"
)
```

**Internal Structure**: `{"conditions": [(func, 1.0)]}`

---

### Format 2: List of Callables

**Use Case**: Multiple conditions as separate groups, each with equal score

```python
Subtask(
    conditions=[
        partial(object_grabbed, object='banana'),
        partial(object_grabbed, object='rubiks_cube')
    ],
    logical="any",  # Grab ANY one object
    name="grab_any_object"
)
```

**Internal Structure**: Each callable becomes its own group with equal scores
```python
{
    "group1": [(func1, 0.5)],
    "group2": [(func2, 0.5)]
}
```

---

### Format 3: List of Tuples (Callable, Score)

**Use Case**: Multiple conditions with custom score weighting

```python
Subtask(
    conditions=[
        (partial(object_grabbed, object='banana'), 0.3),
        (partial(object_in_container, object='banana', container='bowl'), 0.7)
    ],
    logical="all",
    name="weighted_conditions"
)
```

**Internal Structure**: Each tuple becomes its own group
```python
{
    "group1": [(func1, 1.0)],  # Scores normalized within group
    "group2": [(func2, 1.0)]
}
```

---

### Format 4: Set of Callables

**Use Case**: Same as list, but order doesn't matter semantically

```python
Subtask(
    conditions={
        partial(object_grabbed, object='banana'),
        partial(object_grabbed, object='rubiks_cube')
    },
    logical="any",
    name="grab_any_set"
)
```

**Note**: Converted to list internally (order may vary)

---

### Format 5: Set of Tuples (Callable, Score)

**Use Case**: Multiple conditions with scores, order-agnostic

```python
Subtask(
    conditions={
        (partial(object_grabbed, object='banana'), 0.5),
        (partial(object_grabbed, object='rubiks_cube'), 0.5)
    },
    logical="any",
    name="weighted_set"
)
```

---

### Format 6: Dict with Single Callables

**Use Case**: Explicit group names, single condition per group

```python
Subtask(
    conditions={
        "banana": partial(object_grabbed, object='banana'),
        "cube": partial(object_grabbed, object='rubiks_cube')
    },
    logical="all",  # Both must be grabbed
    name="grab_both_explicit"
)
```

**Internal Structure**:
```python
{
    "banana": [(func1, 1.0)],
    "cube": [(func2, 1.0)]
}
```

---

### Format 7: Dict with List of Callables

**Use Case**: Multiple sequential conditions per group, equal scores

**This is the most common format for complex multi-step tasks.**

```python
Subtask(
    conditions={
        "banana": [
            partial(object_grabbed, object='banana'),
            partial(object_above_bottom, object='banana', reference_object='bowl'),
            partial(object_dropped, object='banana'),
            partial(object_in_container, object='banana', container='bowl')
        ],
        "cube": [
            partial(object_grabbed, object='rubiks_cube'),
            partial(object_above_bottom, object='rubiks_cube', reference_object='bowl'),
            partial(object_dropped, object='rubiks_cube'),
            partial(object_in_container, object='rubiks_cube', container='bowl')
        ]
    },
    logical="all",  # Both banana AND cube must complete all steps
    name="pick_and_place_both"
)
```

**Internal Structure**: Each callable gets equal score within its group
```python
{
    "banana": [(func1, 0.25), (func2, 0.25), (func3, 0.25), (func4, 0.25)],
    "cube": [(func1, 0.25), (func2, 0.25), (func3, 0.25), (func4, 0.25)]
}
```

**Key Feature**: Within each group, conditions are checked **sequentially** with regression checking.

---

### Format 8: Dict with List of Tuples (Callable, Score)

**Use Case**: Full control over groups, sequences, and scores

**This is the most flexible format.**

```python
Subtask(
    conditions={
        "banana": [
            (partial(object_grabbed, object='banana'), 0.1),
            (partial(object_above_bottom, object='banana', reference_object='bowl'), 0.2),
            (partial(object_dropped, object='banana'), 0.3),
            (partial(object_in_container, object='banana', container='bowl'), 0.4)
        ],
        "cube": [
            (partial(object_grabbed, object='rubiks_cube'), 0.25),
            (partial(object_above_bottom, object='rubiks_cube', reference_object='bowl'), 0.25),
            (partial(object_dropped, object='rubiks_cube'), 0.25),
            (partial(object_in_container, object='rubiks_cube', container='bowl'), 0.25)
        ]
    },
    logical="all",
    name="fully_weighted"
)
```

**Note**: Scores within each group are normalized to sum to 1.0.

---

### Format 9: Dict with Set of Callables

**Use Case**: Order-agnostic sequential conditions per group

```python
Subtask(
    conditions={
        "banana": {
            partial(object_grabbed, object='banana'),
            partial(object_in_container, object='banana', container='bowl')
        },
        "cube": {
            partial(object_grabbed, object='rubiks_cube'),
            partial(object_in_container, object='rubiks_cube', container='bowl')
        }
    },
    logical="any",  # Either banana OR cube completes
    name="pick_and_place_any"
)
```

**Note**: Converted to list internally.

---

## Logical Modes

The `logical` parameter determines when a subtask group is considered complete.

### Mode 1: "all" (Default)

**Semantics**: All groups must complete all their conditions

**Progress Tracking**:
```
Completed: 0/2 groups, Score: 0.00
  rubiks_cube: 0/4 conditions (0% complete)
  banana: 0/4 conditions (0% complete)

Completed: 0/2 groups, Score: 0.375
  rubiks_cube: 0/4 conditions (0% complete)
  banana: 3/4 conditions (75% complete)

Completed: 1/2 groups, Score: 0.75
  rubiks_cube: 2/4 conditions (50% complete)
  banana: 4/4 conditions (100% complete) ✓

Completed: 2/2 groups, Score: 1.0 ✓
  rubiks_cube: 4/4 conditions (100% complete) ✓
  banana: 4/4 conditions (100% complete) ✓
```

---

### Mode 2: "any"

**Semantics**: Success when any single group completes all its conditions


**Progress Tracking**:
```
Completed: 0/3 groups, Score: 0.00
  red_block: 0/4 conditions (0% complete)
  blue_block: 0/4 conditions (0% complete)
  green_block: 0/4 conditions (0% complete)

Completed: 0/3 groups, Score: 0.50
  red_block: 0/4 conditions (0% complete)
  blue_block: 2/4 conditions (50% complete)
  green_block: 0/4 conditions (0% complete)

Completed: 1/3 groups, Score: 1.0 ✓ (blue_block completed)
  red_block: 0/4 conditions (0% complete)
  blue_block: 4/4 conditions (100% complete) ✓
  green_block: 1/4 conditions (25% complete)
```

---

### Mode 3: "choose"

**Semantics**: Success when exactly K groups complete all their conditions

**Important**: The `K` parameter is **required** when using `logical="choose"`.

**Progress Tracking**:
```
Completed: 0/5 groups, Score: 0.30, Need K=2
  banana_01: 0/4 conditions (0% complete)
  banana_02: 3/4 conditions (75% complete)
  banana_03: 1/4 conditions (25% complete)
  banana_04: 0/4 conditions (0% complete)
  banana_05: 2/4 conditions (50% complete)
  Top 2: [0.75, 0.50] → avg = 0.625

Completed: 1/5 groups, Score: 0.75, Need K=2
  banana_01: 0/4 conditions (0% complete)
  banana_02: 4/4 conditions (100% complete) ✓
  banana_03: 1/4 conditions (25% complete)
  banana_04: 0/4 conditions (0% complete)
  banana_05: 2/4 conditions (50% complete)
  Top 2: [1.0, 0.50] → avg = 0.75

Completed: 2/5 groups, Score: 1.0 ✓ (K=2 reached!)
  banana_01: 0/4 conditions (0% complete)
  banana_02: 4/4 conditions (100% complete) ✓
  banana_03: 1/4 conditions (25% complete)
  banana_04: 0/4 conditions (0% complete)
  banana_05: 4/4 conditions (100% complete) ✓
  Top 2: [1.0, 1.0] → avg = 1.0
```

---

## Composite Functions

Composite functions automatically expand into multiple atomic subtasks, providing convenient shortcuts for common task patterns.

### pick_and_place()

The `pick_and_place()` function is a composite that creates a complete pick-and-place sequence:

**Signature**:
```python
@composite
def pick_and_place(
    object: str | list[str],
    container: str,
    logical: Literal["all", "any", "choose"] = "all",
    K: Optional[int] = None,
    score: float = 1.0
) -> Subtask:
```

**Returns**: A `Subtask` dataclass with the following structure:
```python
{
    "object1": [
        (partial(object_grabbed, object='object1'), 0.25),
        (partial(object_above_bottom, object='object1', reference_object='container'), 0.25),
        (partial(object_dropped, object='object1'), 0.25),
        (partial(object_in_container, object='object1', container='container'), 0.25)
    ],
    "object2": [
        # ... same sequence for object2
    ]
}
```

**Examples**:

```python
from robolab.core.task.conditionals import pick_and_place

# Example 1: Both objects must be placed
subtasks = [
    pick_and_place(
        object=["rubiks_cube", "banana"],
        container="bowl",
        logical="all",
        score=1.0
    )
]

# Example 2: Any one object is sufficient
subtasks = [
    pick_and_place(
        object=["red_block", "blue_block", "green_block"],
        container="bin",
        logical="any",
        score=0.5
    )
]

# Example 3: Exactly 2 out of 3 objects
subtasks = [
    pick_and_place(
        object=["banana_01", "banana_02", "banana_03"],
        container="bowl",
        logical="choose",
        K=2,
        score=1.0
    )
]
```

**NOTE** Do NOT use  for termination conditions (use `object_placed_in_container` instead). Terminations check final state only, while subtasks track intermediate progress.

---

## Sequential Subtasks

Multiple `Subtask` objects can be chained to create multi-stage tasks. Each stage must complete before the next begins.

### Basic Sequential Example

```python

subtasks = [
    # Stage 1:
    Subtask(...),
    # Stage 2:
    Subtask(...)
]
```

### Mixed Logical Modes

```python
subtasks = [
    # Stage 1: Pick any one block, must be complete before starting the next one
    pick_and_place(
        object=["red_block", "blue_block"],
        container="bowl",
        logical="any",
        score=0.3
    ),

    # Stage 2: Both fruits must be placed.
    pick_and_place(
        object=["banana", "apple"],
        container="bowl",
        logical="all",
        score=0.4
    ),

]
```

**Progress Tracking**:
```
Overall Progress: 0/2 stages complete (0%)
Current Stage 1: logical='any', score=0.3
  red_block: 2/4 (50%)
  blue_block: 0/4 (0%)

Overall Progress: 1/2 stages complete (50%)
Current Stage 2: logical='all', score=0.4
  banana: 0/4 (0%)
  apple: 0/4 (0%)

Overall Progress: 2/2 stages complete (100%) ✓
```

---

## Scoring System

### Score Normalization

**Within Each Group**: Condition scores are relative to each other, normalized to sum to 1.0

```python
# Before normalization
conditions = {
    "banana": [
        (func1, 0.1),
        (func2, 0.2),
        (func3, 0.3)
    ]
}
# Total: 0.6

# After normalization (automatic)
conditions = {
    "banana": [
        (func1, 0.167),  # 0.1 / 0.6
        (func2, 0.333),  # 0.2 / 0.6
        (func3, 0.500)   # 0.3 / 0.6
    ]
}
# Total: 1.0
```

**Across Subtasks**: Subtasks are also relative to each other, which will be normalized to 1.

```python
subtasks = [
    Subtask(..., score=0.2),  # 20% of total task
    Subtask(..., score=0.5),  # 50% of total task
    Subtask(..., score=0.3)   # 30% of total task
]
# These don't need to sum to 1.0, but they define relative weights
```

### A visual example of subtask state tracking for a group of objects:

```
Running RubiksCubeBananaTaskHomeOffice_0: 'put the cube and banana in the bowl'


 19%|█████████████████▏                                                                        | 90/470 [00:13<01:09,  5.45it/s]
--------------------------------------------------
SUBTASK STATE
--------------------------------------------------
Current Object Progress: {'rubiks_cube': 0, 'banana': 1}
Objects and their current subtasks:
  rubiks_cube (step 0/4):
     0. ❌ object_grabbed(object=rubiks_cube) <-- CURRENT
     1. ❌ object_above_bottom_surface(object=rubiks_cube, surface=bowl) (pending)
     2. ❌ object_dropped(object=rubiks_cube) (pending)
     3. ❌ object_in_container(object=rubiks_cube, container=bowl, tolerance=0.05) (pending)
  banana (step 1/4):
     0. ✅ object_grabbed(object=banana) (completed)
     1. ❌ object_above_bottom_surface(object=banana, surface=bowl) <-- CURRENT
     2. ❌ object_dropped(object=banana) (pending)
     3. ❌ object_in_container(object=banana, container=bowl, tolerance=0.05) (pending)
==================================================
Subtask status: i: 90, status: {'status': <SubtaskStatusCode.OBJECT_GRABBED_SUCCESS: 120>, 'completed': 1, 'total': 8, 'info': "success: object_grabbed(object='banana');"}


 30%|██████████████████████████▉                                                              | 142/470 [00:23<01:11,  4.56it/s]
--------------------------------------------------
SUBTASK STATE
--------------------------------------------------
Current Object Progress: {'rubiks_cube': 0, 'banana': 2}
Objects and their current subtasks:
  rubiks_cube (step 0/4):
     0. ❌ object_grabbed(object=rubiks_cube) <-- CURRENT
     1. ❌ object_above_bottom_surface(object=rubiks_cube, surface=bowl) (pending)
     2. ❌ object_dropped(object=rubiks_cube) (pending)
     3. ❌ object_in_container(object=rubiks_cube, container=bowl, tolerance=0.05) (pending)
  banana (step 2/4):
     0. ✅ object_grabbed(object=banana) (completed)
     1. ✅ object_above_bottom_surface(object=banana, surface=bowl) (completed)
     2. ❌ object_dropped(object=banana) <-- CURRENT
     3. ❌ object_in_container(object=banana, container=bowl, tolerance=0.05) (pending)
==================================================
Subtask status: i: 142, status: {'status': <SubtaskStatusCode.OBJECT_ABOVE_BOTTOM_SURFACE_SUCCESS: 160>, 'completed': 2, 'total': 8, 'info': "success: object_above_bottom_surface(object='banana', surface='bowl');"}


 39%|██████████████████████████████████▊                                                      | 184/470 [00:32<01:02,  4.57it/s]
--------------------------------------------------
SUBTASK STATE
--------------------------------------------------
Current Object Progress: {'rubiks_cube': 0, 'banana': 3}
Objects and their current subtasks:
  rubiks_cube (step 0/4):
     0. ❌ object_grabbed(object=rubiks_cube) <-- CURRENT
     1. ❌ object_above_bottom_surface(object=rubiks_cube, surface=bowl) (pending)
     2. ❌ object_dropped(object=rubiks_cube) (pending)
     3. ❌ object_in_container(object=rubiks_cube, container=bowl, tolerance=0.05) (pending)
  banana (step 3/4):
     0. ✅ object_grabbed(object=banana) (completed)
     1. ✅ object_above_bottom_surface(object=banana, surface=bowl) (completed)
     2. ✅ object_dropped(object=banana) (completed)
     3. ❌ object_in_container(object=banana, container=bowl, tolerance=0.05) <-- CURRENT
==================================================
Subtask status: i: 184, status: {'status': <SubtaskStatusCode.OBJECT_DROPPED_SUCCESS: 140>, 'completed': 3, 'total': 8, 'info': "success: object_dropped(object='banana');"}


 39%|███████████████████████████████████                                                      | 185/470 [00:32<00:57,  4.92it/s]
--------------------------------------------------
SUBTASK STATE
--------------------------------------------------
Current Object Progress: {'rubiks_cube': 0, 'banana': 4}
Objects and their current subtasks:
  rubiks_cube (step 0/4):
     0. ❌ object_grabbed(object=rubiks_cube) <-- CURRENT
     1. ❌ object_above_bottom_surface(object=rubiks_cube, surface=bowl) (pending)
     2. ❌ object_dropped(object=rubiks_cube) (pending)
     3. ❌ object_in_container(object=rubiks_cube, container=bowl, tolerance=0.05) (pending)
  banana (step 4/4):
     0. ✅ object_grabbed(object=banana) (completed)
     1. ✅ object_above_bottom_surface(object=banana, surface=bowl) (completed)
     2. ✅ object_dropped(object=banana) (completed)
     3. ✅ object_in_container(object=banana, container=bowl, tolerance=0.05) (completed)
==================================================
Subtask status: i: 185, status: {'status': <SubtaskStatusCode.OBJECT_IN_CONTAINER_SUCCESS: 110>, 'completed': 4, 'total': 8, 'info': "success: object_in_container(object='banana', container='bowl', tolerance=0.05). All subtasks complete for banana.;"}


 88%|██████████████████████████████████████████████████████████████████████████████▍          | 414/470 [01:11<00:10,  5.43it/s]
--------------------------------------------------
SUBTASK STATE
--------------------------------------------------
Current Object Progress: {'rubiks_cube': 1, 'banana': 4}
Objects and their current subtasks:
  rubiks_cube (step 1/4):
     0. ✅ object_grabbed(object=rubiks_cube) (completed)
     1. ❌ object_above_bottom_surface(object=rubiks_cube, surface=bowl) <-- CURRENT
     2. ❌ object_dropped(object=rubiks_cube) (pending)
     3. ❌ object_in_container(object=rubiks_cube, container=bowl, tolerance=0.05) (pending)
  banana (step 4/4):
     0. ✅ object_grabbed(object=banana) (completed)
     1. ✅ object_above_bottom_surface(object=banana, surface=bowl) (completed)
     2. ✅ object_dropped(object=banana) (completed)
     3. ✅ object_in_container(object=banana, container=bowl, tolerance=0.05) (completed)
==================================================
Subtask status: i: 414, status: {'status': <SubtaskStatusCode.OBJECT_GRABBED_SUCCESS: 120>, 'completed': 5, 'total': 8, 'info': "success: object_grabbed(object='rubiks_cube');"}


 93%|██████████████████████████████████████████████████████████████████████████████████▉      | 438/470 [01:17<00:07,  4.56it/s]
--------------------------------------------------
SUBTASK STATE
--------------------------------------------------
Current Object Progress: {'rubiks_cube': 2, 'banana': 4}
Objects and their current subtasks:
  rubiks_cube (step 2/4):
     0. ✅ object_grabbed(object=rubiks_cube) (completed)
     1. ✅ object_above_bottom_surface(object=rubiks_cube, surface=bowl) (completed)
     2. ❌ object_dropped(object=rubiks_cube) <-- CURRENT
     3. ❌ object_in_container(object=rubiks_cube, container=bowl, tolerance=0.05) (pending)
  banana (step 4/4):
     0. ✅ object_grabbed(object=banana) (completed)
     1. ✅ object_above_bottom_surface(object=banana, surface=bowl) (completed)
     2. ✅ object_dropped(object=banana) (completed)
     3. ✅ object_in_container(object=banana, container=bowl, tolerance=0.05) (completed)
==================================================
Subtask status: i: 438, status: {'status': <SubtaskStatusCode.OBJECT_ABOVE_BOTTOM_SURFACE_SUCCESS: 160>, 'completed': 6, 'total': 8, 'info': "success: object_above_bottom_surface(object='rubiks_cube', surface='bowl');"}


 97%|██████████████████████████████████████████████████████████████████████████████████████▎  | 456/470 [01:20<00:02,  4.73it/s]
--------------------------------------------------
SUBTASK STATE
--------------------------------------------------
Current Object Progress: {'rubiks_cube': 3, 'banana': 4}
Objects and their current subtasks:
  rubiks_cube (step 3/4):
     0. ✅ object_grabbed(object=rubiks_cube) (completed)
     1. ✅ object_above_bottom_surface(object=rubiks_cube, surface=bowl) (completed)
     2. ✅ object_dropped(object=rubiks_cube) (completed)
     3. ❌ object_in_container(object=rubiks_cube, container=bowl, tolerance=0.05) <-- CURRENT
  banana (step 4/4):
     0. ✅ object_grabbed(object=banana) (completed)
     1. ✅ object_above_bottom_surface(object=banana, surface=bowl) (completed)
     2. ✅ object_dropped(object=banana) (completed)
     3. ✅ object_in_container(object=banana, container=bowl, tolerance=0.05) (completed)
==================================================
Subtask status: i: 456, status: {'status': <SubtaskStatusCode.OBJECT_DROPPED_SUCCESS: 140>, 'completed': 7, 'total': 8, 'info': "success: object_dropped(object='rubiks_cube');"}


 97%|██████████████████████████████████████████████████████████████████████████████████████▋  | 458/470 [01:21<00:02,  5.18it/s]
--------------------------------------------------
SUBTASK STATE
--------------------------------------------------
Current Object Progress: {'rubiks_cube': 4, 'banana': 4}
Objects and their current subtasks:
  rubiks_cube (step 4/4):
     0. ✅ object_grabbed(object=rubiks_cube) (completed)
     1. ✅ object_above_bottom_surface(object=rubiks_cube, surface=bowl) (completed)
     2. ✅ object_dropped(object=rubiks_cube) (completed)
     3. ✅ object_in_container(object=rubiks_cube, container=bowl, tolerance=0.05) (completed)
  banana (step 4/4):
     0. ✅ object_grabbed(object=banana) (completed)
     1. ✅ object_above_bottom_surface(object=banana, surface=bowl) (completed)
     2. ✅ object_dropped(object=banana) (completed)
     3. ✅ object_in_container(object=banana, container=bowl, tolerance=0.05) (completed)
==================================================
Subtask status: i: 458, status: {'status': <SubtaskStatusCode.OBJECT_IN_CONTAINER_SUCCESS: 110>, 'completed': 8, 'total': 8, 'info': "success: object_in_container(object='rubiks_cube', container='bowl', tolerance=0.05). All subtasks complete for rubiks_cube.;"}
```

---

## Subtask Counts & Difficulty Scoring

Tasks are automatically assigned a **difficulty score** and **difficulty label** based on their subtask structure and attributes. This scoring is deterministic and computed from the task definition — no manual labeling is required.

### Subtask Counts (`num_subtasks`)

The **num_subtasks** metric counts the number of distinct manipulation actions the robot must perform, accounting for the subtask's `logical` mode:

| Logical Mode | Subtask Count |
|:--|:--|
| `"all"` | Number of object groups (every group must complete) |
| `"any"` | 1 (only one group needs to complete) |
| `"choose"` | K (exactly K groups must complete) |

The total is summed across all sequential stages in the task. For example, a task with two sequential stages — the first requiring `"all"` of 2 objects, the second `"any"` of 3 — has `2 + 1 = 3` subtasks.

See `count_subtasks()` in `robolab/core/task/subtask_utils.py`.

### Difficulty Score

The difficulty score combines manipulation volume with skill complexity:

```
difficulty_score = num_subtasks + max(skill_weight)
```

where `max(skill_weight)` is the highest weight among the task's attributes. The skill weights are:

| Weight | Attributes |
|:--|:--|
| 0 | color, semantics, size, conjunction, vague |
| +1 | spatial |
| +2 | counting, sorting, stacking, affordance |
| +3 | reorientation |

### Difficulty Labels

Labels are assigned based on the score using fixed thresholds:

| Label | Score Range |
|:--|:--|
| **simple** | score <= 2 |
| **moderate** | score 3–4 |
| **complex** | score >= 5 |

### Examples

| Task | Subtasks | Max Skill Weight | Score | Label |
|:--|:--|:--|:--|:--|
| `RubiksCubeTask` (1 pick-and-place, no special attrs) | 1 | 0 | 1 | simple |
| `BowlStackingLeftOnRightTask` (1 subtask, spatial) | 1 | 1 (spatial) | 2 | simple |
| `Stack3RubiksCubeTask` (2 subtasks, stacking) | 2 | 2 (stacking) | 4 | moderate |
| `BlockStackingSpecifiedOrderTask` (3 subtasks, stacking+color) | 3 | 2 (stacking) | 5 | complex |
| `ReorientAllMugsTask` (4 subtasks, reorientation) | 4 | 3 (reorientation) | 7 | complex |

### Implementation

The scoring constants and function live in `robolab/core/task/subtask_utils.py`:

- `SKILL_WEIGHTS` — attribute-to-weight mapping
- `DIFFICULTY_THRESHOLDS` — `(simple_max, moderate_max)` tuple
- `compute_difficulty_score(num_subtasks, attributes)` — returns `(score, label)`

The metadata pipeline (`robolab/tasks/_utils/load_task_info.py`) automatically populates `num_subtasks`, `difficulty_score`, and `difficulty_label` for each task. Summary statistics are available via:

```bash
python robolab/tasks/_utils/compute_task_statistics.py --difficulty
python robolab/tasks/_utils/compute_task_statistics.py --difficulty -v  # full task list
```
