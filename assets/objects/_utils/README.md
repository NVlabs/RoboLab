# Object Asset Utilities

Scripts for managing the object catalog and generating documentation.

### Generate the object catalog
Scans USD files and extracts metadata (dimensions, physics properties, etc.) into `object_catalog.json`.
```bash
python generate_catalog.py
python generate_catalog.py --objects path/to/dir   # specific directory
python generate_catalog.py --verbose               # print details
```

### List semantic labels
Lists all unique object classes from the catalog.
```bash
python generate_catalog.py --list-classes
python generate_catalog.py --list-classes --by-dataset
python generate_catalog.py --list-classes --verbose          # show objects per class
python generate_catalog.py --list-classes --by-dataset -v    # show objects per class, grouped by dataset
```

### Generate the README table
Creates the markdown table in `assets/objects/README.md` from the catalog.
```bash
python generate_readme.py
python generate_readme.py --datasets hope ycb vomp
```

### Generate statistics
Outputs statistics about the object dataset (counts by class, dataset, dimensions, etc.).
```bash
python generate_object_statistics.py
python generate_object_statistics.py --format markdown
python generate_object_statistics.py --output stats.json --format json
```

### Generate screenshots
Renders preview images for objects (requires IsaacSim).
```bash
python generate_object_screenshots.py --datasets hope ycb
```

### Convert USD format
Converts between binary (.usd/.usdc) and ASCII (.usda) formats (requires IsaacSim).
```bash
python convert_usd_format.py --to-usda                       # convert all default folders to ASCII
python convert_usd_format.py ycb --to-usda                   # convert subfolder by name (resolves to assets/objects/ycb)
python convert_usd_format.py ycb hope hot3d --to-usda        # convert multiple subfolders by name
python convert_usd_format.py path/to/folder --to-usd         # convert explicit path to binary
python convert_usd_format.py path/to/file.usd --to-usda      # convert a single file
python convert_usd_format.py --to-usda --dry-run             # preview changes
python convert_usd_format.py --to-usda --overwrite           # overwrite existing output files
python convert_usd_format.py --to-usda --delete-original     # delete originals after conversion
```
