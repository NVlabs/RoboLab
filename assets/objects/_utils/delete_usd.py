#!/usr/bin/env python3
"""
Delete USD files (.usd/.usdc or .usda) from object asset directories.

This script finds and deletes USD files of a specified format, showing the user
a list of files before prompting for confirmation.

Usage:
    # Delete all .usd binary files in default folders
    python delete_usd.py --usd

    # Delete all .usda files in default folders
    python delete_usd.py --usda

    # Delete .usd files in a specific subfolder
    python delete_usd.py ycb --usd

    # Delete .usd files in multiple subfolders
    python delete_usd.py ycb hope hot3d --usd

    # Preview what would be deleted (no files removed)
    python delete_usd.py --usd --dry-run

    # Skip confirmation prompt
    python delete_usd.py ycb --usd --yes
"""

import argparse
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description="Delete USD files (.usd/.usdc or .usda) from object asset directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete all .usd binary files from default subfolders
  python delete_usd.py --usd

  # Delete all .usda files from ycb
  python delete_usd.py ycb --usda

  # Delete .usd files from multiple subfolders
  python delete_usd.py ycb hope hot3d --usd

  # Preview what would be deleted
  python delete_usd.py --usd --dry-run
        """
    )

    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        default=DEFAULT_PATHS,
        help="Path(s) to directories or subfolder names under assets/objects "
             "(e.g., 'ycb' resolves to assets/objects/ycb). "
             "Default: ycb, hope, hot3d, vomp, fruits_veggies, handal, basic"
    )

    format_group = parser.add_mutually_exclusive_group(required=True)
    format_group.add_argument(
        "--usd",
        action="store_true",
        help="Delete binary USD files (.usd, .usdc)"
    )
    format_group.add_argument(
        "--usda",
        action="store_true",
        help="Delete ASCII USD files (.usda)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files that would be deleted without removing them"
    )

    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt and delete immediately"
    )

    args = parser.parse_args()

    # Resolve paths: single words without slashes are treated as subfolder names
    resolved_paths = []
    for path in args.paths:
        path_str = str(path)
        if '/' not in path_str and '\\' not in path_str and not path_str.endswith(('.usd', '.usda', '.usdc', '.usdz')):
            resolved = Path(OBJECT_DIR) / path_str
            if resolved.exists():
                resolved_paths.append(resolved)
                print(f"Resolved '{path_str}' -> {resolved}")
            else:
                print(f"Error: Subfolder does not exist: {resolved}")
                return 1
        else:
            if not path.exists():
                print(f"Error: Path does not exist: {path}")
                return 1
            resolved_paths.append(path)

    # Determine which extensions to target
    if args.usd:
        target_exts = ['.usd', '.usdc']
        format_label = ".usd/.usdc"
    else:
        target_exts = ['.usda']
        format_label = ".usda"

    # Find files to delete
    files_to_delete = []
    for path in resolved_paths:
        for ext in target_exts:
            files_to_delete.extend(find_usd_files(path, extension=ext))
    files_to_delete = sorted(set(files_to_delete))

    if not files_to_delete:
        paths_str = ', '.join(str(p) for p in resolved_paths)
        print(f"\nNo {format_label} files found in: {paths_str}")
        return 0

    # Display list of files grouped by parent directory
    object_dir = Path(OBJECT_DIR)
    print(f"\nFound {len(files_to_delete)} {format_label} file(s) to delete:\n")

    current_parent = None
    for f in files_to_delete:
        try:
            rel = f.relative_to(object_dir)
        except ValueError:
            rel = f

        parent = rel.parent
        if parent != current_parent:
            current_parent = parent
            print(f"  {parent}/")

        print(f"    {rel.name}")

    # Dry run stops here
    if args.dry_run:
        print(f"\nDRY RUN - No files were deleted.")
        return 0

    # Prompt for confirmation
    if not args.yes:
        print()
        response = input(f"Delete these {len(files_to_delete)} file(s)? [y/N]: ").strip().lower()
        if response not in ('y', 'yes'):
            print("Aborted.")
            return 0

    # Delete files
    deleted_count = 0
    error_count = 0

    for f in files_to_delete:
        try:
            f.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"  Error deleting {f}: {e}")
            error_count += 1

    # Summary
    print(f"\nDeleted {deleted_count} file(s).", end="")
    if error_count > 0:
        print(f" Errors: {error_count}.", end="")
    print()

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
