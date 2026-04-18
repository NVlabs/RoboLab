# Debugging

RoboLab provides three global flags in `robolab.constants` for controlling debug output and visualization. All are `False` by default.

## Flags

### `VERBOSE`

Prints detailed operational information during environment setup and execution:

- **Environment registration** — Prints the full environment table after registration
- **Environment creation** — Logs event merging, env config saving
- **Episode recording** — Logs data export progress
- **Subtask state machine** — Prints the current subtask state and object tracker after each state change
- **Pose randomization** — Logs per-object randomization ranges, bounding radii, and before/after positions
- **Camera operations** — Logs camera pose randomization, resets, and warnings for missing sensors
- **Contact detection** — Logs which objects the gripper is in contact with, sensor mismatches

**Enable via CLI:**

```bash
python examples/policy/run_eval.py --enable-verbose
```

**Enable programmatically:**

```python
import robolab.constants
robolab.constants.VERBOSE = True
```

### `DEBUG`

Prints per-step conditional evaluation results. This is very detailed and produces output every simulation step — useful for diagnosing why a specific subtask condition is or isn't being satisfied.

Covers all conditional functions in `robolab.core.task.conditionals`:
- `object_grabbed`, `object_dropped`, `object_picked_up`
- `object_in_container`, `object_on_top`, `object_on_bottom`, `object_on_center`
- `object_left_of`, `object_right_of`, `object_in_front_of`, `object_behind`
- `object_above`, `object_below`, `object_below_top`
- `object_enclosed`, `object_inside`, `object_outside_of`
- `object_upright`, `object_at`, `object_between`, `object_next_to`
- `stacked`, `objects_in_line`, `objects_stationary`
- `wrong_object_grabbed`, `gripper_hit_table`, `gripper_fully_closed`
- Subtask state machine advancement and regression

**Enable via CLI:**

```bash
python examples/policy/run_eval.py --enable-debug
```

**Enable programmatically:**

```python
import robolab.constants
robolab.constants.DEBUG = True
```

### `VISUALIZE`

Renders bounding boxes and pose axes for all tracked objects in the viewport at every simulation step. Used inside the episode loop in `robolab/eval/episode.py`.

**Enable programmatically:**

```python
import robolab.constants
robolab.constants.VISUALIZE = True
```

When enabled, every step calls `get_world(env).visualize()`, which draws the oriented bounding box and coordinate axes for each object in the scene.

## World State Visualization

You can also call the visualization API directly at any point during execution, independently of the `VISUALIZE` flag:

```python
from robolab.core.world.world_state import get_world

world = get_world(env)

# Visualize all tracked objects
world.visualize()

# Visualize specific objects only
world.visualize(["bowl", "banana"])
```

This draws bounding boxes and coordinate axes in the viewport:

![Visualization Example](images/bbox_vis.gif)

## Combining Flags

You can enable both `VERBOSE` and `DEBUG` together for maximum diagnostics:

```bash
python examples/policy/run_eval.py --enable-verbose --enable-debug
```

| Flag | Scope | Volume |
|------|-------|--------|
| `VERBOSE` | Infrastructure (registration, env creation, recording, cameras, pose randomization) | Moderate — logs key operations |
| `DEBUG` | Task logic (conditional evaluations, subtask state transitions) | High — prints every step |
| `VISUALIZE` | Rendering (bounding boxes, pose axes in viewport) | Visual only — no text output |

## World State Inspection

`WorldState` (via `get_world(env)`) exposes methods useful for interactive debugging beyond visualization:

```python
from robolab.core.world.world_state import get_world

world = get_world(env)

# Object geometry
world.get_pose("banana")              # (position, quaternion)
world.get_velocity("banana")          # (linear, angular)
world.get_dimensions("banana")        # (x, y, z) extents
world.get_aabb("banana")              # Axis-aligned bounding box
world.get_bbox("banana")              # Oriented bounding box corners + centroid
world.get_centroid("banana")          # Center of mass

# Contact queries
world.in_contact("gripper", "banana")                 # bool
world.get_objects_in_contact_with("gripper")           # list of object names
world.get_contact_force("gripper", "banana")           # force magnitude
world.is_supported_on_surface("banana", "table")       # bool
world.get_objects_supported_on("table")                # list of object names
```

## Diagnostic Scripts

### Verify environment registration

After registration, print a table of all registered environments to confirm tasks were discovered correctly:

```python
from robolab.core.environments.factory import print_env_table
print_env_table()                       # All environments
print_env_table(tag="pick_place")       # Filter by tag
print_env_table(verbose=True)           # Include full env config details
```

Or use the standalone script:

```bash
python scripts/check_registered_envs.py
python scripts/check_registered_envs.py --task BananaInBowlTask
```

### Verify tasks are valid

Check that all task files load correctly, have valid fields, and no duplicate names:

```bash
python scripts/check_tasks_valid.py
python scripts/check_tasks_valid.py --tasks-folder /path/to/my_tasks/tasks
```

### Verify IsaacLab installation

Minimal smoke test that IsaacLab and IsaacSim launch correctly:

```bash
python scripts/check_isaaclab.py
```

### Inspect HDF5 data

View the HDF5 file structure with `h5glance` (install separately with `pip install h5glance`):

```bash
h5glance output/2026-01-24_15-35-59/BananaInBowlTask/run_0.hdf5
```

### Read subtask status from HDF5

Print subtask completion timeline, status codes, and scores from recorded episodes:

```bash
python scripts/read_subtask_status_from_hdf5.py output/.../run_0.hdf5
python scripts/read_subtask_status_from_hdf5.py output/.../run_0.hdf5 -e 0
```

### Check results integrity

Validate that episode results match the HDF5 data (every episode has a matching demo):

```bash
python analysis/check_results.py output/2026-01-24_15-35-59
python analysis/check_results.py output/2026-01-24_15-35-59 --verbose --diagnose
```

See [Analysis and Results Parsing](analysis.md) for the full set of analysis scripts.

## Known Issues

### GPU VRAM leak in non-headless mode across environment reloads

When running **without** `--headless` (i.e., with the GUI viewport enabled), GPU VRAM usage grows each time an environment is created and destroyed. After cycling through enough scenes, this will eventually cause an out-of-memory crash.

**Root cause:** In GUI mode, the Omniverse Kit viewport creates a Hydra render product and associated GPU textures to display the scene. When `env.close()` is called, IsaacLab clears the `SimulationContext` singleton and deletes the `ViewportCameraController`, but the **underlying Kit viewport and its Hydra render product persist** across stage reloads. Each subsequent `create_env()` call triggers `omni.usd.get_context().new_stage()`, which allocates new Hydra scene delegates and GPU-side textures for the viewport without fully releasing the previous ones. This is an IsaacLab 2.2.0 / Omniverse Kit issue — the viewport lifecycle is not tied to the `SimulationContext` lifecycle.

In headless mode, the viewport context and window are never created (`_viewport_context = None`, `_viewport_window = None`), so there is no viewport render product to leak.

**Workaround:** Always use `--headless` when running multi-task evaluations that cycle through many environments. If you need the GUI for debugging, limit your run to a small number of scenes (roughly fewer than 10, depending on GPU VRAM and camera resolution).
