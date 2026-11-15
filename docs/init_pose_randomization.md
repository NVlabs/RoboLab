# Initial-Pose Randomization

Randomize where objects start on the table, with no task code. Every policy runner accepts:

```bash
python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask --init-pose-noise 0.1
python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask --init-pose-noise 0.1 45 --init-pose-objects banana bowl --init-pose-workspace 0.3 0.8 -0.3 0.3
```

| Flag | Meaning | Default |
|---|---|---|
| `--init-pose-noise X [YAW_DEG]` | Each group of objects moves up to ±X m in x and y and turns up to ±YAW_DEG about the vertical axis | off; yaw defaults to 180° × min(1, X / 0.3) |
| `--init-pose-objects NAME…` | Only these objects may move | every simulated object |
| `--init-pose-workspace X_MIN X_MAX Y_MIN Y_MAX` | Hard limit: every moved object stays entirely inside this rectangle | the table top |

All positions are env-local meters (the frame scenes are authored in; see [frames](frames.md)).
Height, roll and pitch never change. The settings are appended to the default output folder
name (e.g. `..._pi05_initpose0.1`).

<div align="center">
  <img src="images/init_pose_randomization.gif" alt="Randomized starting layouts in 25 RoboLab-120 scenes" width="800"/>
  <br><em>25 RoboLab-120 scenes with <code>--init-pose-noise 0.1</code>, seen from the eval viewport camera. The first frame is each scene's original layout; every frame after it is a new episode.</em>
</div>

From Python, pass the same event to `create_env`:

```python
from robolab.core.events.reset_pose import make_init_pose_event

env, env_cfg = create_env("BananaInBowlTask", num_envs=8, events=make_init_pose_event(noise_m=0.1))
```

## What moves

On the first reset, every object in the scene is reduced to its footprint: the convex outline
of its meshes seen from above, at its default pose (so a long figurine or a diagonal spoon
does not claim the space a box around it would). Then:

1. **Objects that touch move together.** Two objects closer than 1 cm at their default poses
   are linked; linked objects form a group that moves as one rigid piece, so stacks, objects
   in containers, objects on risers and objects leaning on each other keep their arrangement.
2. **The task can add links.** A task's optional `object_groups` field lists objects that must
   keep their relative poses even though they do not touch (see [Tasks](task.md)). It can only
   join groups, never split them. Use it when the instruction tells objects apart by position
   ("stack the left bowl on the right bowl"): independently moved objects closer than 2X apart
   can swap places. A group can still swap left and right if it turns more than 90 degrees,
   which the default yaw allows from X = 0.15 m.
3. **The task can hold objects in place.** A task's optional `fixed_objects` field lists
   objects that are never moved, e.g. `fixed_objects = ["shelf"]` (see [Tasks](task.md)). Use
   it for an object the task relies on staying where it was authored. It only affects the
   starting layout; the object can still be pushed during the episode.
4. **Some groups stay put.** A group is fixed if it contains a fixed object (one listed in
   `fixed_objects`, a static or kinematic scene object such as a shelf, an object the task
   does not simulate, or an object left out of `--init-pose-objects`), or if it already sticks
   out of the workspace by more than 3 cm. Everything linked to a fixed object stays with it,
   whether the link comes from touching or from `object_groups`: an object on a fixed shelf
   stays put, while an object on a movable riser moves with it, and an `object_groups` entry
   with one fixed member holds every member in place. Fixed objects are obstacles. A group that sticks out by less (authored
   scenes often let a bin or plate overhang the table edge by a few millimeters) moves, but
   never further out than it started on that side.
5. **The robot is an obstacle.** Every robot link over the table (at the default joint
   configuration) is a fixed obstacle. Footprints also record the height range an object
   occupies, and two footprints only collide if their height ranges come within 1 cm, so a mug
   can sit under the hovering gripper but a tall dish rack cannot.

The groups are printed once per task and saved as `init_pose_groups.json` next to the task's
results, with the reason each fixed group stays put ("fixed by the task", "static", "linked to
fixed object(s) shelf", ...). Naming an object in `--init-pose-objects` that has to stay fixed
is an error, so the task's `fixed_objects` always wins over the command line.

## How a layout is sampled

On every reset, for every env:

- Groups are placed largest first. Each gets one xy shift (its center lands within ±X of its
  default, inside the workspace) and one yaw; its members rotate about the group's center.
  It starts 5 mm above its authored height and drops into place: authored scenes often sink
  objects a few millimeters into the table, which is stable where they were settled but gets
  them ejected elsewhere.
- A candidate is accepted only if every member stays inside the workspace and keeps at least
  1 cm from every fixed object and every object already placed. The check uses the footprint
  outlines turned by the yaw, so it ignores height (an object cannot be placed under an
  overhang).
- A group gets 100 tries. The first 50 use the full ±X and yaw ranges; the last 50 shrink
  both toward the default pose (the last try is the default itself). If a group still cannot
  be placed (an earlier group took its default spot), the env's whole layout is resampled in
  a random order with smaller ranges, up to 10 times; the 10th uses the default layout. So
  crowded tables always get a layout, with less movement, and never stop the run. The check
  script below reports which episodes needed smaller ranges.
- Once all groups are placed, every pair of footprints involving a moved group is checked
  again.

Reading the outlines takes a few seconds once per task (it walks every mesh vertex);
sampling is plain numpy on the CPU, well under a second per reset for 16 envs.

## Reproducibility

Each episode draws from its own random generator, seeded by `(--seed, task name, episode
index)` with episode index = run index × num_envs + env id. Episode 17 of a task gets the
same layout for every policy, every `--num-envs` split and every resumed run; parallel envs get
different layouts; changing `--seed` gives a new set.

### Controlling the seed

```bash
# Default seed 0: the same layouts every time you run this command
python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask --init-pose-noise 0.1
# A different, equally reproducible set of layouts
python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask --init-pose-noise 0.1 --seed 7
```

- `--seed N` (default 0, must be nonnegative) is the same flag that seeds the simulator; there is
  no separate layout seed. From Python, pass `seed=N` to `create_env`.
- The task name is the task's `task_name` or class name, so every registered variant of a task
  (other backgrounds, lighting, cameras) gets the same layouts.
- The episode index comes from the run index, so `--num-envs 8 --num-runs 2` and
  `--num-envs 16` give the same 16 layouts. `--run-index-offset K` starts at run K, so workers
  that split one evaluation into disjoint runs get the layouts a single worker would have.
- From Python, set `env.run_index` before `env.reset()`; it defaults to 0.

### Where the starting poses are saved

Every episode's starting layout is written to the run's HDF5 file, so you can tell exactly
where each object started without re-running anything:

```
<output_folder>/<ENV_NAME>/run_<r>.hdf5
└ data/demo_<env id>/initial_state/
  ├ rigid_object/<object>/root_pose   [1 × 7]  x, y, z (env-local m), then quaternion w, x, y, z
  └ articulation/robot/...            joint positions and root pose
```

The snapshot is taken after the randomization, because Isaac Lab runs the reset events
(including this one) before the recorder captures the scene. It is the pose as placed,
including the 5 mm lift, before the object settles onto the table. Every later step is in
`data/demo_<env id>/states`. [Replay](replay.md) restores this initial state exactly, so
randomized episodes replay like any other. See [data](data.md) for the full HDF5 layout.

The sampled shift and turn of each group are not saved separately (`examples/check_init_pose.py`
writes them to `episode_motions.json`); recover them from the starting poses and the default
layout if needed. `init_pose_groups.json` next to the task's results lists the groups and the
settings used.

## Checking a scene

`examples/check_init_pose.py` renders randomized layouts from above (one image per env and a
GIF per scene, with each footprint drawn on top) without running a policy. Run it for every
new task or scene; the `robolab-init-pose-check` skill walks through the review. Add
`--view viewport` to render from the eval viewport camera instead, without footprints.

```bash
python examples/check_init_pose.py --headless --task BananaInBowlTask --init-pose-noise 0.1
```
