# Objects

Objects are USD assets used in manipulation tasks. Each object is a self-contained `.usd` or `.usda` file with physics properties (rigid body, collision, mass, friction) that make it graspable and simulatable in IsaacSim.

RoboLab ships several object datasets in [`assets/objects/`](../assets/objects/):

| Dataset | Description |
|---------|-------------|
| `ycb` | YCB object set |
| `hope` | HOPE household objects |
| `hot3d` | HOT3D objects |
| `handal` | HANDAL objects |
| `vomp` | VOMP object set |
| `fruits_veggies` | Fruits and vegetables |

The full inventory is in [`assets/objects/object_catalog.json`](../assets/objects/object_catalog.json), and a visual table is in [`assets/objects/README.md`](../assets/objects/README.md).

## Creating New Objects

This requires experience with the IsaacSim GUI and assumes you have object meshes ready.

### File layout

Place objects under a dataset folder:

```
assets/objects/
  my_dataset/
    banana/
      banana.usd
      textures/
        banana_diffuse.png
    mug.usd
```

Use simple names (`object_name.usd`). Objects can either live directly in the dataset folder or in a subdirectory with a `textures/` folder alongside.

### USD requirements

- **`defaultPrim`** — Must be the top-level prim, named the same as the file (e.g., `/banana` for `banana.usd`).
- **RigidBodyAPI** — Applied to `defaultPrim`. Only one prim in the entire tree should have `RigidBodyAPI`.
- **Collision** — Use `physxSchema` collision API with `convexDecomposition` (vertices: 256, shrinkwrap: 0.01). Do **not** use `UsdPhysics` collision API.
- **Mass** — Apply `MassAPI` to `defaultPrim` (use either density or mass).
- **Friction** — Static and dynamic friction from 2.0–5.0 for reliable grasping.
- **Attributes** — `defaultPrim` should contain `description`, `class`, and `dataset` attributes describing the object.
- **Textures** — All texture paths must be relative to the USD file. Do not use absolute paths or paths pointing outside the repo.

### Workflow

After creating your objects:

1. **(Optional) Convert format** — Convert between binary and ASCII USD:

```bash
python assets/objects/_utils/convert_usd_format.py my_dataset --to-usda
python assets/objects/_utils/convert_usd_format.py my_dataset --to-usd
```

2. **Set physics properties** — Batch-update friction and restitution on all objects:

```bash
python assets/objects/_utils/update_object_properties.py assets/objects/my_dataset \
    --static 5.0 --dynamic 5.0
```

3. **Generate the catalog** — Scan objects and build `object_catalog.json`:

```bash
python assets/objects/_utils/generate_catalog.py
python assets/objects/_utils/generate_catalog.py --objects assets/objects/my_dataset  # specific directory
```

4. **(Optional) Generate screenshots** — Render preview images (requires IsaacSim):

```bash
python assets/objects/_utils/generate_object_screenshots.py --datasets my_dataset
```

5. **Generate the README** — Build the markdown table from the catalog:

```bash
python assets/objects/_utils/generate_readme.py --datasets my_dataset
```

### Inspecting the catalog

List all semantic class labels:

```bash
python assets/objects/_utils/generate_catalog.py --list-classes
python assets/objects/_utils/generate_catalog.py --list-classes --by-dataset
python assets/objects/_utils/generate_catalog.py --list-classes --by-dataset --verbose
```

## Using Objects in Scenes

Objects are placed into USD scenes by dragging them into the IsaacSim viewport (see [Scenes](scene.md)). When a task references objects via `contact_object_list`, the names must match the prim names in the scene.

## Utility Script Reference

All scripts live in [`assets/objects/_utils/`](../assets/objects/_utils/README.md).

| Script | Purpose |
|--------|---------|
| `generate_catalog.py` | Scan objects and build `object_catalog.json` |
| `generate_readme.py` | Build markdown README table from catalog |
| `generate_object_screenshots.py` | Render preview images (requires IsaacSim) |
| `update_object_properties.py` | Batch-update friction, restitution on USD files |
| `convert_usd_format.py` | Convert between `.usd` (binary) and `.usda` (ASCII) |
| `common.py` | Shared helpers (`iter_object_files`, `load_catalog`, etc.) |

## See Also

- [Scenes](scene.md) — Creating USD scenes that contain objects
- [Tasks](task.md) — Defining tasks that reference objects via `contact_object_list`
- [`assets/objects/README.md`](../assets/objects/README.md) — Visual catalog of all objects
- [`assets/objects/_utils/README.md`](../assets/objects/_utils/README.md) — Quick reference for utility scripts
