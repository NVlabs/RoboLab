#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Convert USD files between binary (.usd/.usdc) and ASCII (.usda) formats.

This script opens IsaacSim headlessly to convert USD files. It can convert:
- .usd/.usdc -> .usda (binary to ASCII)
- .usda -> .usd (ASCII to binary)

Usage:
    # Convert all USDs in default folders (ycb, hope, hot3d, vomp, fruits_veggies, basic)
    python convert_usd_format.py --to-usda

    # Convert a subfolder by name (resolves to assets/objects/<name>)
    python convert_usd_format.py ycb --to-usda

    # Convert multiple subfolders by name
    python convert_usd_format.py ycb hope hot3d --to-usda

    # Convert all USDs in a specific folder path to USDA
    python convert_usd_format.py path/to/folder --to-usda

    # Convert all USDAs in a folder to USD (binary)
    python convert_usd_format.py path/to/folder --to-usd

    # Convert a single file
    python convert_usd_format.py path/to/file.usd --to-usda

    # Dry run (show what would be converted)
    python convert_usd_format.py --to-usda --dry-run

    # Overwrite existing output files (no prompts)
    python convert_usd_format.py --to-usda --overwrite

    # If output exists and --overwrite not specified, prompts:
    #   y - yes, overwrite this file
    #   n - no, skip this file (default)
    #   a - overwrite all remaining
    #   s - skip all remaining

    # Delete original files after conversion
    python convert_usd_format.py --to-usda --delete-original
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

from robolab.constants import OBJECT_DIR
from robolab.core.utils.file_utils import find_usd_files

DEFAULT_PATHS = [
    Path(OBJECT_DIR) / "ycb",
    Path(OBJECT_DIR) / "hope",
    Path(OBJECT_DIR) / "hot3d",
    Path(OBJECT_DIR) / "vomp",
    Path(OBJECT_DIR) / "fruits_veggies",
    Path(OBJECT_DIR) / "basic",
    Path(OBJECT_DIR) / "handal",
]


def convert_file(stage, input_path: Path, output_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Convert a single USD file to a different format.

    Args:
        stage: The USD stage (from pxr)
        input_path: Path to input file
        output_path: Path to output file
        dry_run: If True, don't actually convert

    Returns:
        Tuple of (success, message)
    """
    from pxr import Sdf

    if dry_run:
        return True, f"Would convert: {input_path} -> {output_path}"

    try:
        # Use Sdf.Layer instead of Usd.Stage to preserve relative asset paths.
        # Usd.Stage.Open() composes the stage, resolving relative paths to absolute.
        # Sdf.Layer operates at the layer level and keeps paths as authored.
        layer = Sdf.Layer.FindOrOpen(str(input_path))
        if not layer:
            return False, f"Failed to open: {input_path}"

        layer.Export(str(output_path))

        return True, f"Converted: {input_path.name} -> {output_path.name}"
    except Exception as e:
        return False, f"Error converting {input_path}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert USD files between binary and ASCII formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all .usd files in a subfolder by name (resolves to assets/objects/ycb)
  python convert_usd_format.py ycb --to-usda

  # Convert multiple subfolders by name
  python convert_usd_format.py ycb hope hot3d --to-usda

  # Convert all .usd files using a full folder path
  python convert_usd_format.py assets/objects/ycb --to-usda

  # Convert all .usda files to .usd (binary)
  python convert_usd_format.py hope --to-usd

  # Convert a single file
  python convert_usd_format.py assets/objects/ycb/banana.usd --to-usda

  # Preview changes without converting
  python convert_usd_format.py ycb --to-usda --dry-run
        """
    )

    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        default=DEFAULT_PATHS,
        help="Path(s) to USD files, directories, or subfolder names under assets/objects (e.g., 'ycb' resolves to assets/objects/ycb). Default: ycb, hope, hot3d, vomp, fruits_veggies, handal, basic"
    )

    format_group = parser.add_mutually_exclusive_group(required=True)
    format_group.add_argument(
        "--to-usda",
        action="store_true",
        help="Convert to ASCII format (.usda)"
    )
    format_group.add_argument(
        "--to-usd",
        action="store_true",
        help="Convert to binary format (.usd)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without converting"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip if output exists)"
    )

    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete original files after successful conversion"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    # Parse args before importing IsaacSim (which takes time)
    args, _ = parser.parse_known_args()

    # Resolve paths: single words without slashes are treated as subfolder names under OBJECT_DIR
    resolved_paths = []
    for path in args.paths:
        path_str = str(path)
        # Check if it's a single word (no path separators and not a USD file extension)
        if '/' not in path_str and '\\' not in path_str and not path_str.endswith(('.usd', '.usda', '.usdc', '.usdz')):
            # Treat as subfolder name under OBJECT_DIR
            resolved = Path(OBJECT_DIR) / path_str
            if resolved.exists():
                resolved_paths.append(resolved)
                print(f"Resolved '{path_str}' -> {resolved}")
            else:
                print(f"Error: Subfolder does not exist: {resolved}")
                return 1
        else:
            # Treat as direct path
            if not path.exists():
                print(f"Error: Path does not exist: {path}")
                return 1
            resolved_paths.append(path)

    args.paths = resolved_paths

    # Determine source and target extensions
    if args.to_usda:
        source_exts = ['.usd', '.usdc']
        target_ext = '.usda'
    else:
        source_exts = ['.usda']
        target_ext = '.usd'

    # Find files to convert from all paths
    files_to_convert = []
    for path in args.paths:
        for ext in source_exts:
            files_to_convert.extend(find_usd_files(path, extension=ext))
    files_to_convert = sorted(set(files_to_convert))

    if not files_to_convert:
        paths_str = ', '.join(str(p) for p in args.paths)
        print(f"No {'/'.join(source_exts)} files found in: {paths_str}")
        return 0

    print(f"Found {len(files_to_convert)} file(s) to convert")

    if args.dry_run:
        print("\nDRY RUN - No files will be modified\n")
        for f in files_to_convert:
            output_path = f.with_suffix(target_ext)
            exists_note = " (exists, would skip)" if output_path.exists() and not args.overwrite else ""
            print(f"  {f.name} -> {output_path.name}{exists_note}")
        return 0

    # Initialize IsaacSim
    print("\nInitializing IsaacSim (this may take a moment)...")
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})

    # Convert files
    success_count = 0
    skipped_count = 0
    error_count = 0
    deleted_count = 0
    skip_all_existing = False

    for input_path in files_to_convert:
        output_path = input_path.with_suffix(target_ext)

        # Handle existing output files
        if output_path.exists() and not args.overwrite:
            if skip_all_existing:
                skipped_count += 1
                if not args.quiet:
                    print(f"  - Skipped (output exists): {input_path.name}")
                continue

            # Prompt user
            while True:
                response = input(f"  Output exists: {output_path.name}. Overwrite? [y/N/a(ll)/s(kip all)]: ").strip().lower()
                if response in ('', 'n', 'no'):
                    skipped_count += 1
                    print(f"  - Skipped: {input_path.name}")
                    break
                elif response in ('y', 'yes'):
                    # Continue to conversion below
                    break
                elif response in ('a', 'all'):
                    # Set overwrite for this and all future
                    args.overwrite = True
                    break
                elif response in ('s', 'skip', 'skip all'):
                    skip_all_existing = True
                    skipped_count += 1
                    print(f"  - Skipped: {input_path.name}")
                    break
                else:
                    print("    Please enter y (yes), n (no), a (overwrite all), or s (skip all)")

            # If skipped, continue to next file
            if response in ('', 'n', 'no', 's', 'skip', 'skip all'):
                continue

        success, message = convert_file(None, input_path, output_path, dry_run=False)

        if success:
            success_count += 1
            if not args.quiet:
                print(f"  ✓ {message}")

            # Delete original if requested
            if args.delete_original and input_path != output_path:
                try:
                    input_path.unlink()
                    deleted_count += 1
                    if not args.quiet:
                        print(f"    Deleted: {input_path.name}")
                except Exception as e:
                    print(f"    Warning: Could not delete {input_path.name}: {e}")
        else:
            error_count += 1
            print(f"  ✗ {message}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Converted: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    if args.delete_original:
        print(f"Originals deleted: {deleted_count}")

    # Cleanup
    app.close()

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
