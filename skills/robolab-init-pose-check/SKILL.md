---
name: robolab-init-pose-check
description: >
  Check that a RoboLab task's scene works with initial-pose randomization (--init-pose-noise):
  render randomized layouts from above, one image per env plus a GIF per scene, and review which
  objects move together, which stay put, and whether anything is pushed after placement. Use this
  skill whenever new tasks or scenes are added, when a scene's objects are edited or resettled,
  when a task's object_groups or fixed_objects field changes, or when the user asks whether a task can be
  randomized.
license: CC-BY-NC-4.0
compatibility: >
  Requires a RoboLab install with IsaacSim and a GPU. Runs no policy.
metadata:
  author: nvidia
  version: "1.0.0"
---

# Initial-Pose Randomization Check

Every new task or scene must pass this check before it is used with `--init-pose-noise`.
The check runs no policy: it resets each env with the randomization event, steps physics for
10 steps (1/120 s each), and renders a top-down image of every env.

See `docs/init_pose_randomization.md` for how the randomization works.

## 1. Run the check

Tasks that share a scene USD are rendered once, so pass every new task; the script skips
duplicates. Run headless. Each process needs about 10-12 GB of RAM with 16 envs: check
`free -g` and run at most two processes at once on a 64 GB machine.

```bash
python examples/check_init_pose.py --headless --task <NewTask1> <NewTask2> --task-dirs <folder> --init-pose-noise 0.1
```

For a whole folder, split the unique scenes across processes with `--shard INDEX COUNT`:

```bash
python examples/check_init_pose.py --headless --task-dirs <folder> --init-pose-noise 0.1 --shard 0 3
```

Output goes to `output/init_pose_check/<settings>/`:

- `<task>.gif` — first frame is the default layout, then one frame per env (episode). 600 px
  wide, 300 ms per frame (`--gif-width`, `--gif-frame-ms`); the caption gives the largest shift
  and turn allowed. Encoded with ffmpeg when installed, which keeps colors far better.
- `<task>/env<i>_run<r>.png`, `<task>/default.png` — the same frames as full-size images, no caption.

Add `--view viewport` to render from the eval runners' viewport camera instead, with no
footprints (output in `output/init_pose_check/<settings>_viewport/`). Use it to show layouts to
people; use the default top-down view to review footprints and groups.
- `<task>/init_pose_groups.json` — groups, which ones move, and why the others stay fixed.
- `summary_shard<k>.json` — per-task status, setup and per-reset timing, drift, errors.

Overlay colors: **green** = moved group, **red** = fixed object, **yellow** = workspace (by
default the table top). Each shape is the object's footprint: the convex outline of its
meshes seen from above at its default pose, turned with the group. Tall objects look larger
than their outline near the image edges because of the camera's perspective.

## 2. Review

For every task in the summary:

1. **Status.** `error` means the scene cannot be randomized as is: a `ValueError` names the
   missing table, an unknown object in `object_groups` or `fixed_objects`, or a requested
   object that must stay fixed. Report the message.
   `episodes_with_reduced_noise` lists episodes where the table was too crowded for the full
   noise, with the fraction used (frame captions show the reduced limits and "(crowded)"). Many entries mean the
   scene barely randomizes at that X; say so.
2. **Drift.** `drift_warning: true` means a moved object shifted more than 1 cm in xy during
   the 10 physics steps; `max_drift_m_by_object` names it and `episodes_with_drift` gives the
   frames to open. Usual causes: it was placed under an overhang or against something the
   footprint check does not know about, or it is tall and narrow and tipped over. Look at the
   frame. Drift at the default layout too (run with `--init-pose-noise 0`) means the scene
   itself is unstable; that is a scene problem, not a randomization one.
3. **Groups.** Open `init_pose_groups.json` and the GIF. Check that:
   - Stacks, objects inside containers or on movable risers, and objects touching each other
     move as one piece.
   - Objects on shelves or other fixed objects stay put (red).
   - Objects meant to keep an arrangement, but that do not touch (e.g. a place setting, a
     row of blocks with gaps), move as one group. If not, add them to the task's
     `object_groups` (step 3).
   - **Objects the instruction tells apart by position** ("the left bowl", "the right bin",
     "the middle cube") keep their order. Independent groups can each move up to X, so two
     such objects closer than 2X apart (center to center, along the direction that matters)
     can swap, while the success check still uses the original names. Group them with
     `object_groups`. Read every instruction variant (default, vague, specific), not only
     the default. A grouped pair can still swap if the group turns more than 90 degrees,
     which the default yaw allows from X = 0.15 m; say so if the task is meant for larger X.
   - Objects the task relies on staying where they were authored (a reference object the
     instruction or success check measures against) stay put. If not, list them in the
     task's `fixed_objects` (step 3).
   - The green outlines line up with the rendered objects in every frame.
4. **Coverage.** If every object is red, the task gets no randomization; say so.

Then show the user the GIF paths and a table: task, status, moving groups, fixed objects
with reasons, drift, setup and reset time.

## 3. Fix with `object_groups` or `fixed_objects` (only if needed)

`object_groups` is an optional data field on the task. Each inner list is a set of objects
that keep their relative poses. It only adds links; automatic grouping of touching objects
always applies. `fixed_objects` is a list of objects that never move; everything linked to
them (touching, or in the same `object_groups` entry) stays put too.

```python
@dataclass
class SetTheTableTask(Task):
    contact_object_list = ["fork", "plate", "knife", "table"]
    scene = import_scene("place_setting.usda", contact_object_list)
    object_groups = [["fork", "plate", "knife"]]
    fixed_objects = ["napkin_holder"]
    ...
```

Never add randomization code to a task file. After changing either field, run
`pytest tests/test_tasks_valid.py` (it checks the names exist in the scene) and re-run the
check for that task.

## 4. Optional: other settings

The check accepts the same settings as the eval runners:

```bash
python examples/check_init_pose.py --headless --task <Task> --init-pose-noise 0.2 90 --init-pose-objects banana --init-pose-workspace 0.3 0.8 -0.3 0.3
```

`X [YAW_DEG]` (yaw defaults to 180 × min(1, X / 0.3)), the objects that may move, and the
workspace `X_MIN X_MAX Y_MIN Y_MAX` in env-local meters.
