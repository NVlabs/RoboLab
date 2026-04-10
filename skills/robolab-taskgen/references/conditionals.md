# Conditional Functions Reference

All functions are imported from `robolab.core.task.conditionals`. They are used in termination `success` conditions and in subtask definitions.

## Containment & Placement

| Function | Use when | Key params |
|----------|----------|------------|
| `object_in_container` | Object placed inside an open-top container (bowl, bin, box) | `object`, `container`, `require_contact_with`, `require_gripper_detached` |
| `object_on_top` | Object stably resting on a surface (plate, shelf, table) via contact force | `object`, `reference_object`, `require_gripper_detached` |
| `object_enclosed` | Object's bounding box fully enclosed inside container | `object`, `container` |
| `object_inside` | Object's centroid inside container's bounding box | `object`, `container` |
| `object_outside_of` | Object removed from a container | `object`, `container` |
| `object_outside_of_and_on_surface` | Object removed from container AND placed on surface | `object`, `container`, `surface` |

## Spatial Relations

| Function | Use when | Key params |
|----------|----------|------------|
| `object_left_of` | Object to the left of reference | `object`, `reference_object`, `frame_of_reference` |
| `object_right_of` | Object to the right of reference | `object`, `reference_object`, `frame_of_reference` |
| `object_in_front_of` | Object in front of reference | `object`, `reference_object`, `frame_of_reference` |
| `object_behind` | Object behind reference | `object`, `reference_object`, `frame_of_reference` |
| `object_above` | Object geometrically above reference (bounding box) | `object`, `reference_object`, `z_margin` |
| `object_below` | Object below bottom of reference | `object`, `reference_object` |
| `object_below_top` | Object below top surface of reference | `object`, `reference_object` |
| `object_on_bottom` | Object above bottom surface of reference | `object`, `reference_object` |
| `object_on_center` | Object centered on reference (XY) | `object`, `reference_object`, `tolerance` |
| `object_next_to` | Object within distance of reference | `object`, `reference_object`, `dist` |
| `object_between` | Object positioned between two references | `object`, `reference_obj1`, `reference_obj2` |
| `object_at` | Object at exact 3D position | `object`, `position`, `tolerance` |
| `objects_in_line` | Multiple objects arranged in a line | `objects`, `axis`, `tolerance` |

Spatial conditions support different frames of reference:
- **`frame_of_reference="robot"`** (default): Uses the robot's egocentric perspective
- **`frame_of_reference="world"`**: Uses global world coordinates
- **`mirrored=False`** (default): Robot's natural perspective. Set `True` for a flipped XY perspective.

## Orientation & State

| Function | Use when | Key params |
|----------|----------|------------|
| `object_upright` | Object standing upright | `object`, `tolerance`, `up_axis` |
| `object_grabbed` | Object held by gripper | `object`, `gripper_name` |
| `object_dropped` | Object released from gripper | `object`, `gripper_name` |
| `object_picked_up` | Object grabbed AND lifted above surface | `object`, `surface`, `distance` |
| `objects_stationary` | Objects have stopped moving | `object`, `linear_threshold` |

## Stacking

| Function | Use when | Key params |
|----------|----------|------------|
| `stacked` | Objects stacked in order | `objects` (list), `order` (`"bottom_to_top"` or `None`) |

## Multi-Group

| Function | Use when | Key params |
|----------|----------|------------|
| `object_groups_in_containers` | Multiple object groups each in different containers (sorting) | `groups` (list of dicts with `object`, `container`, `logical`) |

Each group dict has: `{"object": [...], "container": "...", "logical": "all", "require_contact_with": False, "require_gripper_detached": True}`

## Negative Conditions

| Function | Use when | Key params |
|----------|----------|------------|
| `wrong_object_grabbed` | Gripper holding any object other than the target | `object`, `gripper_name`, `ignore_objects` |

## Composite Functions (Subtasks Only)

These expand into multi-step sequences. Use them in `subtasks`, **not** in terminations.

| Function | Use when | Key params |
|----------|----------|------------|
| `pick_and_place` | Pick object(s) and place in container | `object`, `container`, `logical`, `score` |
| `pick_and_place_on_surface` | Pick object(s) and place on flat surface | `object`, `surface`, `logical`, `score` |

## Common Parameters

Most atomic conditionals share these parameters:

- **`object`**: `str | list[str]` -- Object name(s) to check. Must be in `contact_object_list`.
- **`logical`**: `"all" | "any" | "choose"` -- How to combine results when `object` is a list.
  - `"all"`: All objects must satisfy the condition.
  - `"any"`: At least one object must satisfy.
  - `"choose"`: Exactly `K` objects must satisfy (requires `K` param).
- **`require_contact_with`**: `bool | str | list[str]` -- Contact requirement.
  - `False`: No contact check (default).
  - `True`: Must be in contact with the reference object.
  - `str/list`: Must be in contact with the named object(s).
- **`require_gripper_detached`**: `bool` -- If True, object must NOT be held by the gripper. Use `True` for "place" conditions.
- **`gripper_name`**: `str` -- Gripper entity name (default: `"gripper"`).
- **`tolerance`**: `float` -- Distance tolerance for spatial checks (default varies by function).

## Quick Selection Guide

| Task description | Termination function | Subtask function |
|-----------------|---------------------|-----------------|
| "Put X in Y" (bowl, bin, box) | `object_in_container` | `pick_and_place` |
| "Put X on Y" (plate, shelf) | `object_on_top` | `pick_and_place_on_surface` |
| "Stack X, Y, Z" | `stacked` | `Subtask` with `partial(stacked, ...)` |
| "Move X left/right of Y" | `object_left_of` / `object_right_of` | `Subtask` with `partial(...)` |
| "Sort X into bins" | `object_groups_in_containers` | Multiple `pick_and_place` |
| "Take X out of Y" | `object_outside_of` | `Subtask` with `partial(...)` |
| "Stand X upright" | `object_upright` | `Subtask` with `partial(...)` |
| "Line up objects" | `objects_in_line` | `Subtask` with `partial(...)` |
