# Scenes

A **scene** is a USD file containing objects and fixtures (tables, shelves, kitchen surfaces) arranged in a spatial layout. Each [task](task.md) references a scene that defines the physical environment for the episode. Scenes should **not** contain robots, lighting, backgrounds, etc.

RoboLab ships scenes in [`assets/scenes/`](../assets/scenes/). A visual catalog is in [`assets/scenes/README.md`](../assets/scenes/README.md) and metadata lives in `assets/scenes/_metadata/`.

## Creating a New USD Scene

> [!NOTE]
> Creating new USD scene via GUI assumes you have experience with IsaacSim GUI, and assumes you have [object](objects.md) USD files ready.

1. **Start from a template** — Open an existing scene (e.g., `base_empty.usda`) in IsaacSim as a starting point. Save-as the scene with a readable name. we recommend saving as `.usda` as these are human readable files.

   ![Start from base scene](images/start_from_base_scene.gif)
2. **Set `defaultPrim`** — The top-level prim must be set as `defaultPrim`.
3. **Add objects** — Drag object USD files from your object folder (e.g., `assets/objects/`) into the viewport. After you do this, make sure that the reference payload is _relative_, i.e., `../../assets/objects/object.usd`, instead of a global path.

   ![Drag and drop from assets](images/drag_and_drop_from_assets.gif)

   After adding objects, make sure that the reference payload is relative:

   ![Relative paths](images/relative_paths.png)
4. **Name prims carefully** — Click the object in the stage tree (right-hand panel) to rename it. These prim names are what tasks use in `contact_object_list` and conditionals.
5. **To move objects in the scene (position/orientation)** — Select objects in the right-hand panel to move them. Do **not** drag objects directly in the viewport; this moves the mesh away from the object's base `XFormPrim`.

   ![Do not click on object in scene](images/do_not_click_on_object_in_scene.gif)
6. You can click play and stop to observe the physics behavior of the objects. You do not need to place them exactly on the surface, we will run a settler to settle everything.

   ![Play scene](images/play_scene.gif)

### Settling scenes

After positioning objects, run the physics settler to let objects come to rest naturally and strip residual velocities:

```bash
# Settle a single scene
python assets/scenes/_utils/settle_scenes.py --scene my_scene.usda

# Settle all scenes in a directory
python assets/scenes/_utils/settle_scenes.py --scene /path/to/scenes/

# Replace originals in-place
python assets/scenes/_utils/settle_scenes.py --scene my_scene.usda --replace

# Settle and take screenshots
python assets/scenes/_utils/settle_scenes.py --scene my_scene.usda --replace --screenshot
```

It is recommended that you re-open the scenes and check that all the objects you intended to place looks good.

## AI Workflows: Scene Generation

We provide a Claude Code agent skill that you can use to generate scene files from natural language descriptions using the `/robolab-scenegen` skill. Describe the objects and arrangement you want, and the agent will produce a complete, valid USDA scene file, run physics settling, and generate a screenshot. See [`skills/robolab-scenegen/`](../skills/robolab-scenegen/) for details.

## To use the scenes in tasks

For how to use scenes in task definitions (via `import_scene` and `import_scene_and_contact_object_list`), see [Tasks](task.md#importing-scenes).

## Generating Scene Metadata

RoboLab provides utility scripts for generating metadata, screenshots, and statistics for scenes. These work with any scene directory.

### Generate metadata (JSON, CSV, and README)

Analyzes scene USDs, writes `scene_metadata.json`, `scene_table.csv`, and regenerates `README.md`:

```bash
python assets/scenes/_utils/generate_scene_metadata.py
python assets/scenes/_utils/generate_scene_metadata.py --scene-folder /path/to/scenes
python assets/scenes/_utils/generate_scene_metadata.py --scene my_scene.usda  # single scene
python assets/scenes/_utils/generate_scene_metadata.py --generate-images      # also take screenshots
```

### Generate screenshots only

Renders preview images without re-analyzing USD:

```bash
python assets/scenes/_utils/generate_scene_screenshots.py
python assets/scenes/_utils/generate_scene_screenshots.py --scene my_scene.usda
python assets/scenes/_utils/generate_scene_screenshots.py --view top   # top/front/angled
```

### Regenerate README only

Rebuilds `README.md` from existing `scene_table.csv` without running Isaac:

```bash
python assets/scenes/_utils/generate_scene_readme_only.py
```

### Compute scene statistics

Analyzes `scene_metadata.json` for object counts, dataset distribution, etc.:

```bash
python assets/scenes/_utils/compute_scene_statistics.py
python assets/scenes/_utils/compute_scene_statistics.py --save  # write scene_statistics.json
```

## Utility Script Reference

All scripts live in [`assets/scenes/_utils/`](../assets/scenes/_utils/README.md).

| Script | Purpose |
|--------|---------|
| `settle_scenes.py` | Run physics to settle objects, strip velocities, export clean USD |
| `generate_scene_metadata.py` | Analyze scene USDs and produce JSON, CSV, and README |
| `generate_scene_screenshots.py` | Render preview images (front/angled/top views) |
| `generate_scene_readme_only.py` | Rebuild README from existing CSV (no Isaac required) |
| `compute_scene_statistics.py` | Compute statistics from scene metadata |

## See Also

- [Objects](objects.md) — Creating and managing USD object assets
- [Tasks](task.md) — Defining tasks that bind scenes to instructions and termination criteria
- [Task Libraries](task_libraries.md) — Organizing tasks into collections
- [`assets/scenes/README.md`](../assets/scenes/README.md) — Visual catalog of all scenes
- [`assets/scenes/_utils/README.md`](../assets/scenes/_utils/README.md) — Quick reference for utility scripts
