# Scene Utility Scripts

Scripts for managing scene assets, generating metadata, and rendering previews.
For full documentation, see [Scenes](../../../docs/scene.md).

### Settle scenes

Run physics to let objects come to rest, strip velocities and `PhysicsScene` prims, export clean USD:

```bash
python settle_scenes.py --scene my_scene.usda
python settle_scenes.py --scene /path/to/scenes/           # settle all scenes in a directory
python settle_scenes.py --scene my_scene.usda --replace     # overwrite original
python settle_scenes.py --scene my_scene.usda --screenshot  # also take screenshots
```

### Generate scene metadata (JSON, CSV, and README)

Analyze scene USDs, extract object info, and write `scene_metadata.json`, `scene_table.csv`, and `README.md`:

```bash
python generate_scene_metadata.py
python generate_scene_metadata.py --scene-folder /path/to/scenes
python generate_scene_metadata.py --scene my_scene.usda     # single scene
python generate_scene_metadata.py --generate-images          # also render screenshots
```

### Generate screenshots

Render preview images for scenes (front/angled/top views):

```bash
python generate_scene_screenshots.py
python generate_scene_screenshots.py --scene my_scene.usda
python generate_scene_screenshots.py --view top              # top, front, or angled
```

### Regenerate README only

Rebuild `README.md` from existing `scene_table.csv` without running Isaac:

```bash
python generate_scene_readme_only.py
```

### Compute scene statistics

Analyze `scene_metadata.json` for object counts, dataset distribution, etc.:

```bash
python compute_scene_statistics.py
python compute_scene_statistics.py --save   # write scene_statistics.json
```
