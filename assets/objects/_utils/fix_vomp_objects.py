from robolab.constants import OBJECT_DIR, OBJECT_CATALOG_PATH
from typing import List, Dict, Any
import json
from pathlib import Path
import sys
import os

def _iter_object_files(root: Path = Path(OBJECT_DIR)) -> List[Path]:
    """Return all USD files under the *root* directory, excluding subfolders starting with '_' and any 'materials' directories."""

    usd_extensions = ["*.usd", "*.usda", "*.usdc", "*.usdz"]
    usd_files = []

    for extension in usd_extensions:
        for file_path in root.glob(f"**/{extension}"):
            rel_parts = file_path.relative_to(root).parts[:-1]
            # Skip any path that has a folder starting with '_' OR a folder named 'materials'
            if any(part.startswith('_') for part in rel_parts):
                continue
            if any(part.lower() == 'materials' for part in rel_parts):
                continue
            usd_files.append(file_path)

    return usd_files

def _print_object_info(object_info: Dict[str, Any], usd_path: Path) -> None:
    """Print object info dictionary line by line in a readable format."""
    print(f"Object: {usd_path.name}, usd_path: {usd_path}")
    for key, value in object_info.items():
        if isinstance(value, (list, tuple)) and len(value) > 0:
            # Format arrays/tuples nicely
            if isinstance(value[0], (int, float)):
                formatted_value = f"[{', '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in value)}]"
            else:
                formatted_value = str(value)
        else:
            formatted_value = str(value)
        print(f"  {key:20s}: {formatted_value}")

def fix_absolute_paths_in_usd(usd_path: Path, dry_run: bool = False) -> tuple[bool, list[str]]:
    """
    Replace absolute paths with relative paths in USD file, only if the file exists locally.
    Checks if the file exists in OBJECT_DIR/vomp/<object_name>/... before replacing.
    Replaces patterns like '/home/rdagli/code/datasets/simready/simready_content/common_assets/props/<object_name>/'
    with './' only if the file exists locally.

    Args:
        usd_path: Path to the USD file
        dry_run: If True, only print what would be changed

    Returns:
        Tuple of (changes_made, missing_files)
    """
    from pxr import Usd, Sdf

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open stage: {usd_path}")
        return False, []

    # Get the object name and directory
    object_name = usd_path.parent.name
    object_dir = usd_path.parent

    # Pattern to search for
    old_pattern = f"/home/rdagli/code/datasets/simready/simready_content/common_assets/props/{object_name}/"

    changes_made = False
    missing_files = []

    # Helper function to check and replace path
    def check_and_replace_path(path_str: str):
        """Check if file exists and return (should_replace, new_path)"""
        if old_pattern not in path_str:
            return False, path_str

        # Extract the relative part after the old pattern
        relative_part = path_str.split(old_pattern, 1)[1]

        # Check if file exists in the object directory
        target_file = object_dir / relative_part
        if target_file.exists():
            new_path = path_str.replace(old_pattern, "./")
            return True, new_path
        else:
            # Store the full original path
            missing_files.append(path_str)
            return False, path_str

    # Traverse all prims and check all attributes
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            # Get the attribute value
            value = attr.Get()

            # Check if it's a string or asset path that contains the pattern
            if isinstance(value, str) and old_pattern in value:
                should_replace, new_value = check_and_replace_path(value)
                if should_replace:
                    if dry_run:
                        print(f"[DRY RUN] Would replace in {prim.GetPath()}.{attr.GetName()}:")
                        print(f"  Old: {value}")
                        print(f"  New: {new_value}")
                    else:
                        attr.Set(new_value)
                        print(f"Fixed path in {prim.GetPath()}.{attr.GetName()}")
                    changes_made = True
            elif hasattr(value, 'path'):
                # Handle Sdf.AssetPath
                asset_path = getattr(value, 'path', None)
                if asset_path and isinstance(asset_path, str) and old_pattern in asset_path:
                    should_replace, new_path = check_and_replace_path(asset_path)
                    if should_replace:
                        if dry_run:
                            print(f"[DRY RUN] Would replace asset path in {prim.GetPath()}.{attr.GetName()}:")
                            print(f"  Old: {asset_path}")
                            print(f"  New: {new_path}")
                        else:
                            from pxr import Sdf
                            attr.Set(Sdf.AssetPath(new_path))
                            print(f"Fixed asset path in {prim.GetPath()}.{attr.GetName()}")
                        changes_made = True

    if changes_made and not dry_run:
        stage.GetRootLayer().Save()
        print(f"Saved {usd_path.name} with fixed paths")

    return changes_made, missing_files

def set_default_prim_one_level_down(usd_path: Path, dry_run: bool = False, rename_root: bool = True) -> bool:
    """
    Move the child of the current default prim to the root level and set it as the new default prim.
    Renames it to match the USD file name.

    Args:
        usd_path: Path to the USD file to modify
        dry_run: If True, only print what would be changed without saving
        rename_root: If True, rename the prim to match the USD file name (default: True)

    Returns:
        True if the file was modified (or would be modified in dry_run), False otherwise
    """
    import omni.usd
    import omni.kit.commands
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open stage: {usd_path}")
        return False

    current_default_prim = stage.GetDefaultPrim()
    if not current_default_prim:
        print(f"No default prim found in: {usd_path}")
        return False

    # Get children of the current default prim
    children = list(current_default_prim.GetChildren())
    if not children:
        print(f"Default prim has no children in: {usd_path}")
        return False

    # Get the first child (this will be moved up to root level)
    first_child = children[0]
    old_child_path = first_child.GetPath()

    # Get the new name from the USD file name (without extension)
    new_root_name = usd_path.stem

    new_root_path = str(f"/{new_root_name}")
    temp_root_path = str(f"/{new_root_name}_temp")

    if dry_run:
        print(f"[DRY RUN] Would change:")
        print(f"  File: {usd_path.name}")
        print(f"  Move child to root: {old_child_path} -> {temp_root_path}")
        print(f"  Remove old root: {current_default_prim.GetPath()}")
        print(f"  New default prim: {new_root_path}")
        return True

    # Close any existing stage and open this USD file
    omni.usd.get_context().close_stage()
    success = omni.usd.get_context().open_stage(str(usd_path))
    if not success:
        print(f"ERROR: Failed to open stage in Isaac Sim: {usd_path}")
        return False

    # Get the stage from Isaac Sim context
    stage = omni.usd.get_context().get_stage()

    # Convert paths to strings for the command
    print(f"Moving prim from {old_child_path} to {temp_root_path}")

    # Step 1: Move the child prim to root level with new name
    omni.kit.commands.execute("MovePrim", path_from=old_child_path, path_to=temp_root_path)

    # Step 2: Remove the old root prim
    old_root_path_str = str(current_default_prim.GetPath())
    omni.kit.commands.execute("DeletePrims", paths=[old_root_path_str])

    omni.kit.commands.execute("MovePrim", path_from=temp_root_path, path_to=new_root_name)

    # Step 3: Set the moved prim as the new default prim
    new_root_prim = stage.GetPrimAtPath(new_root_path)
    if not new_root_prim or not new_root_prim.IsValid():
        print(f"ERROR: Failed to get prim at {new_root_path}")
        return False

    stage.SetDefaultPrim(new_root_prim)

    print(f"Updated {usd_path.name}: {old_child_path} -> {new_root_path} (new default prim)")
    for prim in stage.Traverse():
        print(prim.GetPath())

    # Step 4: Save the stage
    omni.usd.get_context().save_stage()

    return True

def update_friction_info(usd_path: Path, static_friction: float = 2.0, dynamic_friction: float = 2.0, restitution: float = 0.25) -> bool:
    """
    Update friction properties of the physics material in the USD file.
    """
    from robolab.core.utils.physics_utils import update_friction
    from pxr import Usd
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open stage: {usd_path}")
        return False

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        print(f"No default prim found in: {usd_path}")
        return False

    update_friction_info(default_prim, stage, static_friction=static_friction, dynamic_friction=dynamic_friction, restitution=restitution)

    stage.GetRootLayer().Save()

    return True

def update_mass_api(mass: float, usd_path: Path, dry_run: bool = False) -> bool:
    """
    Update mass of objects in the USD file. If a MassAPI exists, update its mass value.

    Args:
        mass: The mass value to set (typically 0.0)
        usd_path: Path to the USD file
        dry_run: If True, only print what would be changed

    Returns:
        True if the file was modified (or would be modified in dry_run), False otherwise
    """
    from pxr import Usd, UsdPhysics, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open stage: {usd_path}")
        return False

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        print(f"No default prim found in: {usd_path}")
        return False

    changes_made = False

    # Check if the root prim has MassAPI
    if UsdPhysics.MassAPI(default_prim):
        mass_api = UsdPhysics.MassAPI(default_prim)

        # Get density if available
        density = None
        density_attr = mass_api.GetDensityAttr()
        if density_attr and density_attr.IsValid():
            density = density_attr.Get()
            if density is not None:
                print(f"  Density on {default_prim.GetPath()}: {density}")

        mass_attr = mass_api.GetMassAttr()
        if mass_attr:
            old_mass = mass_attr.Get()

            # For the default prim (root level), always set mass to 0.0 when mass parameter is 0.0
            should_update = False
            if mass == 0.0:
                # Always set mass to 0 on default prim
                if old_mass is not None and old_mass != 0.0:
                    should_update = True
            else:
                # For nonzero mass values, always update if different
                if old_mass is not None and old_mass != mass:
                    should_update = True

            if should_update:
                if dry_run:
                    print(f"[DRY RUN] Would update mass on {default_prim.GetPath()} from {old_mass} to {mass}")
                else:
                    mass_attr.Set(mass)
                    print(f"Updated mass on {default_prim.GetPath()} from {old_mass} to {mass}")
                changes_made = True

    # Check all child meshes for MassAPI
    for prim in Usd.PrimRange(default_prim):
        if prim.IsA(UsdGeom.Mesh) and UsdPhysics.MassAPI(prim):
            mass_api = UsdPhysics.MassAPI(prim)

            # Get density if available
            density = None
            density_attr = mass_api.GetDensityAttr()
            if density_attr and density_attr.IsValid():
                density = density_attr.Get()
                if density is not None:
                    print(f"  Density on {prim.GetPath()}: {density}")

            mass_attr = mass_api.GetMassAttr()
            if mass_attr:
                old_mass = mass_attr.Get()

                # Determine if we should update the mass
                should_update = False
                if mass == 0.0:
                    # Only set mass to 0 if both current mass and density are nonzero
                    if old_mass is not None and old_mass != 0.0 and density is not None and density != 0.0:
                        should_update = True
                else:
                    # For nonzero mass values, always update if different
                    if old_mass is not None and old_mass != mass:
                        should_update = True

                if should_update:
                    if dry_run:
                        print(f"[DRY RUN] Would update mass on {prim.GetPath()} from {old_mass} to {mass}")
                    else:
                        mass_attr.Set(mass)
                        print(f"Updated mass on {prim.GetPath()} from {old_mass} to {mass}")
                    changes_made = True

    if changes_made and not dry_run:
        stage.GetRootLayer().Save()
        print(f"Saved {usd_path.name} with updated mass")

    return changes_made


def update_colliders_attributes(usd_path: Path, dry_run: bool = False) -> bool:
    """
    Update colliders attributes in the USD file.
    Sets contact_offset to -inf and rest_offset to -inf.
    """
    from pxr import Usd
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open stage: {usd_path}")
        return False

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        print(f"No default prim found in: {usd_path}")
        return False

    from robolab.core.utils.physics_utils import update_rigid_body_collider
    update_rigid_body_collider(default_prim, contact_offset=float('-inf'), rest_offset=float('-inf'))

    stage.GetRootLayer().Save()
    return True

def add_rigid_body_collider(usd_path: Path, dry_run: bool = False) -> bool:
    """
    Add rigid body collider to the USD file.
    """
    from pxr import Usd
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open stage: {usd_path}")
        return False

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        print(f"No default prim found in: {usd_path}")
        return False

    from robolab.core.utils.physics_utils import add_rigid_body_collider
    add_rigid_body_collider(default_prim, stage, enable_ccd=True)
    return True

def add_descriptions_to_objects(usd_path: Path, dry_run: bool = False) -> bool:
    """
    Add descriptions to the objects in the USD file.
    """
    from pxr import Usd
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"Failed to open stage: {usd_path}")
        return False

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        print(f"No default prim found in: {usd_path}")
        return False

    import json

    catalog_path = Path(OBJECT_DIR) / "object_catalog_old.json"

    if not catalog_path.exists():
        print(f"Catalog file not found at {catalog_path}")
        return False

    with open(catalog_path, "r") as f:
        catalog_data = json.load(f)

        for obj in catalog_data:
            if obj['name'] == default_prim.GetName():
                description = obj['description']
                class_name = obj['class']
                from robolab.core.utils.usd_utils import set_str_attribute_to_prim
                set_str_attribute_to_prim(default_prim, 'description', description)
                set_str_attribute_to_prim(default_prim, 'class', class_name)
                stage.GetRootLayer().Save()
                break
    return True

def fix_vomp_objects(objects_dir: Path = Path(OBJECT_DIR),
                     verbose: bool = False,
                     dry_run: bool = False,
                     filter_pattern: str = "vomp",
                     rename_root: bool = True,
                     fix_paths: bool = True,
                     update_friction: bool = False,
                     static_friction: float = 2.0,
                     dynamic_friction: float = 2.0,
                     restitution: float = 0.25,
                     update_colliders: bool = False,
                     mass: float = 0.0) -> None:
    """
    Fix vomp objects by moving the child prim to root level and renaming it to match the file name.
    Optionally also fix absolute texture paths.
    Example: /RootNode/redonion_inst -> /redonion

    Args:
        objects_dir: Root directory containing object USD files
        dry_run: If True, only print what would be changed without saving
        filter_pattern: Only process files whose path contains this pattern
        rename_root: If True, rename the prim to match the USD file name (default: True)
        fix_paths: If True, also fix absolute paths to relative paths (default: True)
        only_fix_paths: If True, only fix paths without changing prim structure
    """
    if objects_dir.is_dir():
        usd_files = _iter_object_files(objects_dir)
    else:
        usd_files = [objects_dir]

    # Filter for vomp objects
    filtered_files = [f for f in usd_files if filter_pattern in str(f)]

    print(f"Found {len(filtered_files)} files matching pattern '{filter_pattern}'")

    modified_count = 0
    all_missing_files = {}  # Dictionary to track missing files per object

    for usd_file in filtered_files:
        print(f"\nProcessing: {usd_file.name}")

        file_modified = False
        missing_files = []

        # Fix absolute paths if requested
        if fix_paths:
            changed, missing = fix_absolute_paths_in_usd(usd_file, dry_run=dry_run)
            if changed:
                file_modified = True
            if missing:
                all_missing_files[usd_file.parent.name] = missing

        # Move prim structure if not only fixing paths
        if rename_root:
            if set_default_prim_one_level_down(usd_file, dry_run=dry_run, rename_root=rename_root):
                file_modified = True

        # Add descriptions to the objects
        add_descriptions_to_objects(usd_file, dry_run=dry_run)

        if update_friction:
            if update_friction_info(usd_file, static_friction=static_friction, dynamic_friction=dynamic_friction, restitution=restitution):
                file_modified = True

        if update_colliders:
            if update_colliders_attributes(usd_file, dry_run=dry_run):
                file_modified = True

        if mass is not None:
            if update_mass_api(mass, usd_file, dry_run=dry_run):
                file_modified = True

        if file_modified:
            modified_count += 1

    print(f"\n{'Would modify' if dry_run else 'Modified'} {modified_count}/{len(filtered_files)} files")

    # Print summary of missing files
    if all_missing_files:
        print("\n" + "="*80)
        print("SUMMARY: Missing Files (absolute paths kept)")
        print("="*80)

        total_missing = sum(len(files) for files in all_missing_files.values())
        print(f"Total missing files: {total_missing} across {len(all_missing_files)} objects\n")

        for object_name, missing_files in sorted(all_missing_files.items()):
            print(f"\n{object_name}/ ({len(missing_files)} missing files):")
            for file_path in sorted(set(missing_files)):  # Remove duplicates and sort
                print(f"  - {file_path}")

        print("\n" + "="*80)
# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fix VOMP objects by moving child prim to root level and fixing absolute paths")
    parser.add_argument("--objects", type=Path, default=Path(os.path.join(OBJECT_DIR, "vomp")), help="Objects directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    parser.add_argument("--filter", type=str, default="vomp", help="Filter pattern for file paths (default: 'vomp')")
    parser.add_argument("--rename-root", action="store_true", help="Rename prim to match file name")
    parser.add_argument("--fix-paths", action="store_true", help="Fix absolute paths to relative paths")
    parser.add_argument("--update-friction", action="store_true", help="Update friction properties of the physics material")
    parser.add_argument("--static-friction", type=float, default=2.0, help="Static friction coefficient")
    parser.add_argument("--dynamic-friction", type=float, default=2.0, help="Dynamic friction coefficient")
    parser.add_argument("--restitution", type=float, default=None, help="Restitution coefficient")
    parser.add_argument("--update-colliders", action="store_true", help="Update colliders attributes")
    args = parser.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    fix_vomp_objects(
        objects_dir=args.objects,
        dry_run=args.dry_run,
        filter_pattern=args.filter,
        rename_root=args.rename_root,
        fix_paths=args.fix_paths,
        update_friction=args.update_friction,
        static_friction=args.static_friction,
        dynamic_friction=args.dynamic_friction,
        restitution=args.restitution,
        update_colliders=args.update_colliders,
        mass=None,
    )
    app.close()
