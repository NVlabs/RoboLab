#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Script to update physics material properties (friction, restitution) in USD files.

This script can:
- Update physics materials in individual USD files or entire directories
- Check existing values and only update if different
- Support dry-run mode to preview changes
- Batch process multiple files

Usage:
    # Update single file
    python update_object_properties.py path/to/object.usd --static 5.0 --dynamic 5.0 --restitution 0.25

    # Update directory of files
    python update_object_properties.py path/to/objects/ --static 5.0 --dynamic 5.0

    # Dry run
    python update_object_properties.py path/to/objects/ --static 5.0 --dry-run
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError as exc:
    sys.stderr.write("[ERROR] Pixar USD bindings not found. Please install the pxr Python package.\n")
    raise

def update_physics_material_in_usd(
    usd_path: Path,
    static_friction: Optional[float] = None,
    dynamic_friction: Optional[float] = None,
    restitution: Optional[float] = None,
    dry_run: bool = False,
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    Update physics material properties in a USD file.

    This function reads and writes USD files directly without requiring Isaac Sim.
    For runtime updates within Isaac Sim, use robolab.core.utils.physics_utils.modify_friction()

    Args:
        usd_path: Path to the USD file
        static_friction: New static friction value (None to skip)
        dynamic_friction: New dynamic friction value (None to skip)
        restitution: New restitution value (None to skip)
        dry_run: If True, only check differences without modifying
        verbose: If True, print detailed information

    Returns:
        Tuple of (changed, message) indicating if changes were made and a status message
    """
    if not usd_path.exists():
        return False, f"Error: File not found: {usd_path}"

    # Open the stage
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        return False, f"Error: Could not open USD stage: {usd_path}"

    # Get the default prim (root)
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        return False, f"Error: No default prim found in {usd_path}"

    root_prim_path = str(default_prim.GetPath())

    # Get current values using our utility function
    from robolab.core.utils.usd_utils import get_friction_info
    current_values = get_friction_info(default_prim, stage)

    if current_values is None:
        if verbose:
            print(f"  No physics material found at {root_prim_path}/physics_material")
        return False, "No physics material found"

    # Check what needs to be updated
    updates_needed = {}

    if static_friction is not None and current_values.get('static_friction') != static_friction:
        updates_needed['static_friction'] = (current_values.get('static_friction'), static_friction)

    if dynamic_friction is not None and current_values.get('dynamic_friction') != dynamic_friction:
        updates_needed['dynamic_friction'] = (current_values.get('dynamic_friction'), dynamic_friction)

    if restitution is not None and current_values.get('restitution') != restitution:
        updates_needed['restitution'] = (current_values.get('restitution'), restitution)

    if not updates_needed:
        if verbose:
            print(f"  No changes needed (current values match)")
        return False, "No changes needed"

    # Print what will be updated
    if verbose:
        for prop, (old_val, new_val) in updates_needed.items():
            print(f"  {prop}: {old_val} → {new_val}")

    if dry_run:
        return True, f"Would update {len(updates_needed)} properties"

    # Apply the updates
    material_prim_path = current_values['prim_path']
    material_prim = stage.GetPrimAtPath(material_prim_path)
    material_api = UsdPhysics.MaterialAPI(material_prim)

    if 'static_friction' in updates_needed:
        material_api.GetStaticFrictionAttr().Set(static_friction)

    if 'dynamic_friction' in updates_needed:
        material_api.GetDynamicFrictionAttr().Set(dynamic_friction)

    if 'restitution' in updates_needed:
        material_api.GetRestitutionAttr().Set(restitution)

    # Save the stage
    stage.Save()

    return True, f"Updated {len(updates_needed)} properties"


def find_usd_files(path: Path) -> List[Path]:
    """
    Find all USD files in a directory (recursively) or return single file.
    Excludes directories starting with '_'.

    Args:
        path: File or directory path

    Returns:
        List of USD file paths
    """
    if path.is_file():
        return [path]

    usd_extensions = ["*.usd", "*.usda", "*.usdc", "*.usdz"]
    usd_files = []

    for extension in usd_extensions:
        for file_path in path.glob(f"**/{extension}"):
            # Check if any part of the path contains a folder starting with '_'
            if not any(part.startswith('_') for part in file_path.relative_to(path).parts[:-1]):
                usd_files.append(file_path)

    return sorted(usd_files)


def main():
    parser = argparse.ArgumentParser(
        description="Update physics material properties in USD files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update single file with all properties
  python update_object_properties.py path/to/object.usd --static 5.0 --dynamic 5.0 --restitution 0.25

  # Update only friction, leave restitution unchanged
  python update_object_properties.py path/to/object.usd --static 8.0 --dynamic 8.0

  # Dry run to preview changes
  python update_object_properties.py path/to/objects/ --static 5.0 --dry-run

  # Update all objects in a directory
  python update_object_properties.py path/to/objects/ --static 5.0 --dynamic 5.0 --restitution 0.1

Note:
  This script operates on USD files directly. For runtime updates within Isaac Sim,
  use: from robolab.core.utils.physics_utils import modify_friction
        """
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to USD file or directory containing USD files"
    )

    parser.add_argument(
        "--static",
        "--static-friction",
        type=float,
        default=5.0,
        dest="static_friction",
        help="Static friction coefficient"
    )

    parser.add_argument(
        "--dynamic",
        "--dynamic-friction",
        type=float,
        default=5.0,
        dest="dynamic_friction",
        help="Dynamic friction coefficient"
    )

    parser.add_argument(
        "--restitution",
        type=float,
        help="Restitution coefficient (bounciness)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.path.exists():
        print(f"Error: Path does not exist: {args.path}")
        return 1

    if args.static_friction is None and args.dynamic_friction is None and args.restitution is None:
        print("Error: At least one property (--static, --dynamic, or --restitution) must be specified")
        return 1

    # Find USD files
    usd_files = find_usd_files(args.path)

    if not usd_files:
        print(f"No USD files found in {args.path}")
        return 1

    print(f"Found {len(usd_files)} USD file(s) to process")
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made\n")

    # Process files
    updated_count = 0
    skipped_count = 0
    error_count = 0

    for usd_file in usd_files:
        if not args.quiet:
            print(f"\nProcessing: {usd_file.name}")

        changed, message = update_physics_material_in_usd(
            usd_file,
            static_friction=args.static_friction,
            dynamic_friction=args.dynamic_friction,
            restitution=args.restitution,
            dry_run=args.dry_run,
            verbose=not args.quiet
        )

        if "Error" in message:
            error_count += 1
            if args.quiet:
                print(f"Error: {usd_file.name} - {message}")
        elif changed:
            updated_count += 1
            if args.quiet:
                print(f"Updated: {usd_file.name}")
        else:
            skipped_count += 1

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files: {len(usd_files)}")
    if args.dry_run:
        print(f"Would update: {updated_count}")
    else:
        print(f"Updated: {updated_count}")
    print(f"Skipped (no changes): {skipped_count}")
    print(f"Errors: {error_count}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
