# Environment Generation

## Contact sensor creation

Contact sensors are automatically created during environment configuration to detect physical contacts between objects and the gripper.

### How sensors are created

When environments are generated, RoboLab dynamically creates contact sensors based on two configuration fields:

1. **`contact_object_list`**: List of object names to monitor (defined in the Task class)

    ```python
    @dataclass
    class MyTask(Task):
        contact_object_list = ["table", "apple", "plate"] # <-- contact sensors are created for these objects.
        scene = import_scene("my_scene.usda", contact_object_list)
    ```

2. **`contact_gripper`**: Dictionary mapping gripper names to prim path patterns (defined in robot configs, e.g., `robolab/robots/droid.py`)


Three types of sensors are created. In the code, objects are separated by two underscores `__`.

| Sensor Type | Naming Pattern | Purpose |
|-------------|----------------|---------|
| Batch gripper sensor | `{gripper}__all_objs` | Efficient batch queries for gripper vs all objects |
| Pairwise gripper-object | `{gripper}__{object}` | Individual gripper-object contact checks |
| Pairwise object-object | `{object1}__{object2}` | Object-to-object contact detection |

### Force direction convention

The sensor's `force_matrix_w` returns forces **on the sensor body from the filter body**. When retrieving contact forces via `WorldState.get_contact_force(body1, body2)`:
- If sensor exists as `body1__body2`: force is returned as-is
- If sensor exists as `body2__body1` (reversed): force is negated to give force on `body1` from `body2`

This ensures `get_contact_force(obj, surface)` returns an upward force (positive Z) when an object rests on a surface.

### Usage

This information is accessible via RoboLab's world state:
```python
# Querying contact forces (robolab/core/world/world_state.py)
world = get_world(env)
force = world.get_contact_force("apple", "table")  # Force on apple from table
supported = world.is_supported_on_surface("apple", "plate")  # Contact force cone check
```

### Key files

- `robolab/core/sensors/contact_sensor_utils.py`: Sensor creation and lookup functions
- `robolab/core/world/world_state.py`: `get_contact_force()`, `is_supported_on_surface()`


## Subtask and failure mode tracker creation
