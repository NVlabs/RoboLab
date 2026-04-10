---
name: robolab-scenegen
description: >
  Generate USD scene files for RoboLab from natural language descriptions.
  Use this skill when a user wants to create a new scene with objects on a table,
  or asks to arrange objects for a robot manipulation task.
license: CC-BY-NC-4.0
compatibility: >
  Requires a RoboLab project with assets/scenes/base_empty.usda and assets/objects/object_catalog.json.
metadata:
  author: nvidia
  version: "1.0.0"
---

# Scene Generation

Generate USD scene files (`.usda`) from natural language descriptions of tabletop arrangements.

A **scene** is a USDA file that places objects on the table in `base_empty.usda`. Scenes are robot-agnostic — they define only what objects exist and where they are positioned.

## Reference Files

The `references/` directory contains detailed documentation loaded on-demand:

- `references/scene_format.md` — USD scene format, coordinate system, placement rules, and examples
- `references/object_catalog_guide.md` — Object catalog structure, classes, and naming conventions
- `references/predicates.md` — Predicate types, JSON format, and common patterns for the solver pipeline

## Prerequisites

- `assets/scenes/base_empty.usda` exists (the base scene template)
- `assets/objects/object_catalog.json` exists (the object catalog)

## When Invoked

When the user invokes this skill, display the following message **verbatim**:

---

I'll help you generate a USD scene file. I need a few things:

1. **Scene name** — What should the file be called? Must end in `.usda` (e.g., `fruits_bowl.usda`, `kitchen_sorting.usda`). Use snake_case.
2. **Scene description** — What should be on the table? (e.g., "a bowl with some fruits around it", "kitchen items for a cooking task", "3 blocks stacked near a container")
3. **Number of objects** — How many objects? (default: 3-5 for simple scenes)
4. **Output directory** — Where should I save the scene?
   - `assets/scenes/` (standard location)
   - `assets/scenes/generated/` (for generated scenes)
   - Or specify another path

**Objects I'll pick from:**

I use the object catalog at `assets/objects/object_catalog.json` (312 objects) built from these directories:

| Directory | Count | Examples |
|-----------|-------|---------|
| `assets/objects/vomp/` | 196 | containers, bins, totes, crates, kitchenware |
| `assets/objects/hope/` | 27 | condiments, canned goods, dairy, pantry items |
| `assets/objects/hot3d/` | 25 | mugs, bowls, electronics, toys |
| `assets/objects/ycb/` | 22 | fruits, bowls, cans, tools |
| `assets/objects/handal/` | 19 | hammers, spoons, utensils, ladles |
| `assets/objects/objaverse/` | 10 | bagels, bread, dumplings |
| `assets/objects/fruits_veggies/` | 9 | avocado, lemon, lime, orange, onion |
| `assets/objects/basic/` | 4 | colored blocks (red, blue, green, yellow) |

If the objects you need aren't in the catalog, you can:
- **Point me to a custom object directory** — give me a path and I'll scan it for USD files to include
- **Refresh the catalog** — if new objects have been added to the directories above, I can regenerate the catalog (requires a Python environment with `pxr`/IsaacSim installed)

---

## After the User Provides Information

After receiving the user's input:

1. **Always ask the user for the output directory on the first invocation.** Do not assume a default — you must wait for the user to confirm a directory before writing any file. Once the user has chosen a directory, reuse it for all subsequent scenes in the same session without asking again.
2. **Validate the scene name** ends with `.usda`. If not, append `.usda` and confirm with the user.
3. **Check for duplicate scene names.** If a file with the same name already exists at the output path, warn the user and ask them to choose a different name. Do not overwrite existing scene files.
4. **Resolve the object catalog.**
   - Read the default catalog at `assets/objects/object_catalog.json`.
   - If the user specified custom object directories, scan those directories for `.usd`/`.usda` files and include them as additional objects. For each custom object, extract the prim name from the filename and compute dimensions if possible (or ask the user).
   - If the user asked to refresh the catalog, find the IsaacSim Python interpreter (see [Step 0 under After Generating](#step-0-find-the-correct-python-interpreter)) and run `<ISAACSIM_PYTHON> assets/objects/_utils/generate_catalog.py`, then re-read the catalog.
   - **Tell the user which catalog you're using** and how many objects are available (e.g., "Using 312 objects from `assets/objects/object_catalog.json`" or "Using 312 default objects + 5 custom objects from `/path/to/custom/`").
5. **Read `references/scene_format.md`** for the exact USDA format, coordinate system, and placement rules.
6. **Select objects** that match the user's description from the catalog. Use exact `name` values from the catalog.
7. **Generate predicates and solve placements** using the predicate solver pipeline (see [Predicate Solver Pipeline](#predicate-solver-pipeline)).
8. **Compute the payload path** for each object: convert `usd_path` from catalog (e.g., `assets/objects/ycb/banana.usd`) to a path relative to the scene's output directory (e.g., `../objects/ycb/banana.usd` for `assets/scenes/`, or `../../objects/ycb/banana.usd` for `assets/scenes/generated/`)
9. **Generate the USDA file** using the solved positions (see [Scene File Generation](#scene-file-generation)).
10. **Tell the user what objects were placed and where**, with a summary table.

## Predicate Solver Pipeline

The skill uses the same predicate-based solver from `robolab/scene_gen/llm_scene_gen/` to compute collision-free placements. This is a 4-step pipeline that does **not** require IsaacSim — it runs with pure Python + numpy + scipy.

### Step 1: Generate predicates JSON

Based on the user's description, generate a JSON structure with object selections and predicates. This is the same format the LLM agent on `xuning/scene_task_gen` uses:

```json
{
  "objects": [
    {"name": "bowl"},
    {"name": "banana"},
    {"name": "orange_01"}
  ],
  "predicates": [
    {"type": "place-on-base", "object": "bowl", "x": 0.55, "y": 0.0},
    {"type": "random-rot", "object": "bowl"},
    {"type": "place-on-base", "object": "banana", "x": 0.40, "y": -0.20},
    {"type": "random-rot", "object": "banana"},
    {"type": "place-on-base", "object": "orange_01", "x": 0.70, "y": 0.15},
    {"type": "random-rot", "object": "orange_01"}
  ]
}
```

#### Available predicate types

**Spatial (2D placement on table):**
- `place-on-base` — Place on table at explicit (x, y). Params: `object`, `x`, `y`
- `left-of` / `right-of` / `front-of` / `back-of` — Relative to another object. Params: `object`, `reference`, `distance`
- `random-rot` — Random yaw rotation. Params: `object`
- `facing-left` / `facing-right` / `facing-front` / `facing-back` — Oriented direction. Params: `object`
- `align-left` / `align-right` / `align-center-lr` / `align-center-fb` — Alignment. Params: `object`, `reference`

**Physical (3D placement):**
- `place-on` — Stack on top of another object. Params: `object`, `support`
- `place-in` — Place inside a container. Params: `objects` (list), `container`

#### Coordinate system
- Table center: (0.55, 0.0)
- Safe bounds: X=[0.30, 0.80], Y=[-0.40, 0.40]
- Front = +X, Left = +Y, Right = -Y

Every object needs at least `place-on-base` + `random-rot` (2 predicates).

### Step 2: Parse predicates into ObjectStates

Run a Python script via Bash that:
1. Reads the object catalog
2. Parses the predicates JSON into `ObjectState` objects using `parse_predicates_from_dict`
3. Runs the spatial solver
4. Runs the physical solver
5. Checks grammar feedback
6. Outputs the final (x, y, z, yaw) for each object

Use this exact script pattern:

```python
import json, sys
from robolab.scene_gen.llm_scene_gen import (
    ObjectState, PlaceOnBasePredicate, parse_predicates_from_dict,
    SpatialSolver, PhysicalSolver, FeedbackSystem
)

catalog = json.load(open('assets/objects/object_catalog.json'))
llm_result = json.loads(sys.argv[1])  # Pass predicates JSON as argument

# Parse objects
object_states = {}
object_info = {}
for obj_data in llm_result['objects']:
    name = obj_data['name']
    entry = next((o for o in catalog if o['name'] == name), None)
    if entry:
        object_info[name] = entry
        object_states[name] = ObjectState(name=name)

# Parse predicates
for pred_data in llm_result['predicates']:
    pred = parse_predicates_from_dict(pred_data)
    if pred.target_object in object_states:
        object_states[pred.target_object].predicates.append(pred)

# Add default placement for objects without predicates
for name, state in object_states.items():
    if not state.predicates:
        state.predicates.append(PlaceOnBasePredicate(name))

# Solve spatial constraints
solver = SpatialSolver(table_bounds=(0.25, 0.85, -0.45, 0.45))
dims = {n: tuple(info['dims']) for n, info in object_info.items()}
success, msg = solver.solve(object_states, dims)
if not success:
    print(json.dumps({"error": f"Spatial solver failed: {msg}"}))
    sys.exit(1)

# Solve physical constraints
phys = PhysicalSolver()
phys_success, phys_msg = phys.solve(
    object_states, dims,
    {n: info['usd_path'] for n, info in object_info.items()},
    'assets/scenes/base_empty.usda'
)

# Check grammar
feedback = FeedbackSystem.generate_grammar_feedback(object_states)
if feedback:
    print(json.dumps({"error": f"Grammar issues: {feedback}"}))
    sys.exit(1)

# Output results
results = {}
for name, state in object_states.items():
    results[name] = {
        "x": state.x, "y": state.y,
        "z": object_info[name]['dims'][2] / 2 + 0.002,
        "yaw": state.yaw or 0.0,
        "usd_path": object_info[name]['usd_path']
    }
print(json.dumps(results, indent=2))
```

### Step 3: Handle solver failures

If the spatial solver fails (returns an error), this means objects can't fit without colliding. Options:
1. **Reduce object count** — try with fewer objects
2. **Adjust initial positions** — spread objects further apart in the predicates
3. **Report to user** — explain which objects collided and ask for guidance

If the solver succeeds, proceed to generate the USDA file using the solved positions.

## Scene File Generation

### Step 1: Read base_empty.usda

Read the entire `assets/scenes/base_empty.usda` file. This is your template.

### Step 2: Build object prim blocks

For each object with solved positions from the predicate solver, create a prim block:

```usda
    def "<object_name>" (
        prepend payload = @<relative_path_to_usd>@
    )
    {
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (<x>, <y>, <z>)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }
```

Where:
- `<object_name>` = the `name` field from the catalog (e.g., `banana`, `bowl`)
- `<relative_path_to_usd>` = the `usd_path` converted to be relative from the scene's directory
- `<x>, <y>` = solved positions from the spatial solver
- `<z>` = `dims[2] / 2 + 0.002` (from the solver output)

### Step 3: Insert into base scene

Take the base_empty.usda content and insert the object prim blocks just before the final `}` that closes the `def Xform "world"` block.

### Step 4: Write the file

Write the complete USDA content to the output path using the Write tool.

## Duplicate Object Names

If the scene needs multiple objects of the same type (e.g., two bananas), append a suffix:
- First: `banana`
- Second: `banana_1`
- Third: `banana_2`

Use the same payload path for all instances.

## After Generating the Scene File

After writing the USDA file, **automatically** run settle + screenshot:

### Step 0: Find the correct Python interpreter

The settle and screenshot scripts require IsaacSim (`isaacsim` package) and `pxr` (Pixar USD bindings), which are typically installed in a specific conda environment — **not** the system Python or the project `.venv`.

To find the right interpreter, search for a conda env that has `isaacsim`:

```bash
for env_dir in ~/miniforge3/envs/*/bin ~/miniconda3/envs/*/bin ~/anaconda3/envs/*/bin; do
    if [ -f "$env_dir/python" ]; then
        "$env_dir/python" -c "import isaacsim" 2>/dev/null && echo "Found: $env_dir/python" && break
    fi
done
```

If no conda env is found, fall back to checking `.venv/bin/python` or ask the user which Python has IsaacSim installed.

Cache the discovered interpreter path for the rest of the session.

### Step 1: Settle the scene (physics simulation)

Run the settle script to let objects fall into stable resting positions. Use `--replace` so the settled positions are written back into the same file, and `--screenshot` to capture a rendered image:

```bash
<ISAACSIM_PYTHON> assets/scenes/_utils/settle_scenes.py \
    --scene <scene_path> \
    --replace \
    --screenshot \
    --screenshot-dir <output_dir>/_images
```

Where `<ISAACSIM_PYTHON>` is the interpreter found in Step 0.

- Use a **timeout of 300 seconds** (5 min) — IsaacSim startup takes ~20s, settling takes ~5s, screenshot takes ~5s.
- The `--replace` flag overwrites the original USDA with settled positions.
- The `--screenshot` flag renders a 640x480 image and saves it as `<scene_name>.png` in the screenshot directory.
- Ignore warnings about GLFW, windowing, MDL parameters — these are expected in headless mode.

### Step 2: Show the screenshot to the user

After the settle script completes, read the generated screenshot image using the Read tool and display it to the user:

```
Read: <output_dir>/_images/<scene_name>.png
```

### Step 3: Display next steps

After showing the screenshot, display the following message to the user:

---

Scene settled and screenshot generated!

- **Settled scene**: `<scene_path>` (objects now in stable resting positions)
- **Screenshot**: `<output_dir>/_images/<scene_name>.png`

**Next steps:**

**1. Create a task** — use `/robolab-taskgen` to create a task file that references this scene.

**2. Verify in simulation** — run the empty demo (no policy) once you have a task:
```bash
python examples/demo/run_empty.py --task <TaskClassName>
```

---

## Naming Convention

Scene filenames should be descriptive and use snake_case:
- `banana_bowl.usda` — banana and bowl
- `fruits_plate_sorting.usda` — fruits on a plate for sorting
- `blocks_3_stacking.usda` — 3 blocks for stacking

## Validation Checklist

Before writing the file, verify:
- [ ] All object names exist in the catalog
- [ ] All payload paths are correct relative paths from the scene's directory (e.g., `../objects/` for `assets/scenes/`, or `../../objects/` for `assets/scenes/generated/`)
- [ ] All positions are within table bounds X=[0.30, 0.80], Y=[-0.40, 0.40]
- [ ] No two objects overlap (check pairwise distances)
- [ ] Z heights are computed correctly from object dimensions
- [ ] The USDA syntax is valid (proper quoting, brackets, indentation)
- [ ] The file starts with `#usda 1.0` and has `defaultPrim = "world"`

## Payload Path Computation

The payload path in the USDA must be relative from the scene file's directory to the object USD. The `usd_path` in the catalog is relative to the repo root.

- Scene in `assets/scenes/`: payload = `@../<usd_path minus "assets/">@` (e.g., `@../objects/ycb/bowl.usd@`)
- Scene in `assets/scenes/generated/`: payload = `@../../<usd_path minus "assets/">@` (e.g., `@../../objects/ycb/bowl.usd@`)
- Scene in `assets/scenes/foo/bar/`: count directory depth from `assets/` and prepend the right number of `../`

General formula: strip `assets/` prefix from `usd_path`, then prepend `../` repeated N times where N = depth of scene directory relative to `assets/`.
