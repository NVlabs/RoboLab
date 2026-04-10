#!/usr/bin/env python3
"""
Script to deduplicate consecutive error entries in log files.

For grasp/manipulation events, removes consecutive repetitions:
- Keep only the FIRST entry in each consecutive run
- If the same error appears again after a gap, that's a new occurrence to keep

Target error codes:
- WRONG_OBJECT_GRABBED (250)
- GRIPPER_HIT_TABLE (255)
- GRIPPER_FULLY_CLOSED (256)
- WRONG_OBJECT_DETACHED (257)
- OBJECT_BUMPED (258)
- OBJECT_MOVED (259)
- OBJECT_OUT_OF_SCENE (260)
- OBJECT_STARTED_MOVING (261)
- OBJECT_TIPPED_OVER (262)
- TARGET_OBJECT_DROPPED (263)
- GRIPPER_HIT_OBJECT (264)
- MULTIPLE_OBJECTS_GRABBED (265)

Example:
    Steps 500-509: Same error → keep only step 500
    Step 600: Same error again → keep step 600 (it's after a gap)
"""

import os
import json
import argparse
import glob
from collections import defaultdict


# Target error codes to deduplicate
TARGET_CODES = {
    250: "WRONG_OBJECT_GRABBED",
    255: "GRIPPER_HIT_TABLE",
    256: "GRIPPER_FULLY_CLOSED",
    257: "WRONG_OBJECT_DETACHED",
    258: "OBJECT_BUMPED",
    259: "OBJECT_MOVED",
    260: "OBJECT_OUT_OF_SCENE",
    261: "OBJECT_STARTED_MOVING",
    262: "OBJECT_TIPPED_OVER",
    263: "TARGET_OBJECT_DROPPED",
    264: "GRIPPER_HIT_OBJECT",
    265: "MULTIPLE_OBJECTS_GRABBED",
}


def deduplicate_log_file(log_data: list[dict], verbose: bool = False) -> tuple[list[dict], dict]:
    """
    Deduplicate consecutive error entries in a log file.

    Args:
        log_data: List of timestep dictionaries from log file
        verbose: Whether to print details about removals

    Returns:
        Tuple of (modified_log_data, stats_dict)
    """
    if not log_data:
        return log_data, {}

    # Track the last step where each unique error was recorded
    # Key: (info_string, code) for full matching
    last_seen_step: dict[tuple[str, int], int] = {}

    # Stats
    stats = defaultdict(lambda: {"kept": 0, "removed": 0})

    for step, entry in enumerate(log_data):
        all_status_codes = entry.get("all_status_codes", [])
        if not all_status_codes:
            continue

        filtered_codes = []
        for item in all_status_codes:
            # Handle both list and tuple formats
            if len(item) >= 2:
                info, code = item[0], item[1]
            else:
                filtered_codes.append(item)
                continue

            # Check if this is a target error code
            if code in TARGET_CODES:
                key = (info, code)

                # Check if this is a consecutive occurrence
                if key in last_seen_step:
                    prev_step = last_seen_step[key]
                    if step == prev_step + 1:
                        # This is a consecutive occurrence - skip it
                        stats[TARGET_CODES[code]]["removed"] += 1
                        if verbose:
                            print(f"  Removing step {step}: {info} (consecutive after step {prev_step})")
                        # Update last seen step even for removed entries to track the run
                        last_seen_step[key] = step
                        continue

                # Keep this entry (first in a new run or first ever)
                stats[TARGET_CODES[code]]["kept"] += 1
                last_seen_step[key] = step
                filtered_codes.append(item)
            else:
                # Not a target code - keep as-is
                filtered_codes.append(item)

        entry["all_status_codes"] = filtered_codes

    return log_data, dict(stats)


def process_directory(directory: str, dry_run: bool = True, backup: bool = True, verbose: bool = False) -> dict:
    """
    Process all log files in a directory.

    Args:
        directory: Path to the output directory (e.g., output/pi05_v2)
        dry_run: If True, don't actually modify files
        backup: If True, create .backup files before modifying
        verbose: Print detailed information

    Returns:
        Summary statistics
    """
    log_pattern = os.path.join(directory, "**", "log_*.json")
    log_files = glob.glob(log_pattern, recursive=True)

    total_stats = {
        "files_processed": 0,
        "files_modified": 0,
        "errors_by_type": defaultdict(lambda: {"kept": 0, "removed": 0}),
    }

    print(f"\nProcessing directory: {directory}")
    print(f"Found {len(log_files)} log files")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE (will modify files)'}")
    print("-" * 60)

    for log_file in sorted(log_files):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)

            if not isinstance(log_data, list):
                continue

            total_stats["files_processed"] += 1

            # Deduplicate
            modified_data, file_stats = deduplicate_log_file(log_data, verbose=verbose)

            # Check if any changes were made
            total_removed = sum(s["removed"] for s in file_stats.values())

            if total_removed > 0:
                total_stats["files_modified"] += 1

                # Aggregate stats
                for error_type, counts in file_stats.items():
                    total_stats["errors_by_type"][error_type]["kept"] += counts["kept"]
                    total_stats["errors_by_type"][error_type]["removed"] += counts["removed"]

                rel_path = os.path.relpath(log_file, directory)
                if verbose or not dry_run:
                    print(f"  {rel_path}: removed {total_removed} duplicate entries")

                if not dry_run:
                    # Create backup if requested
                    if backup:
                        backup_file = log_file + ".backup"
                        if not os.path.exists(backup_file):
                            with open(log_file, 'r') as f:
                                original_content = f.read()
                            with open(backup_file, 'w') as f:
                                f.write(original_content)

                    # Write modified data
                    with open(log_file, 'w') as f:
                        json.dump(modified_data, f, separators=(',', ': '))

        except Exception as e:
            print(f"  Error processing {log_file}: {e}")
            continue

    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate consecutive error entries in log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes without modifying)
  python deduplicate_error_logs.py output/pi05_v2 output/pi0_fast_v2

  # Actually modify files (with backups)
  python deduplicate_error_logs.py output/pi05_v2 output/pi0_fast_v2 --apply

  # Modify without backups
  python deduplicate_error_logs.py output/pi05_v2 --apply --no-backup

  # Verbose output
  python deduplicate_error_logs.py output/pi05_v2 --verbose
        """
    )
    parser.add_argument(
        "directories",
        nargs='+',
        help="Output directories to process (e.g., output/pi05_v2 output/pi0_fast_v2)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify files (default is dry-run)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files before modifying"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information about each removal"
    )

    args = parser.parse_args()

    dry_run = not args.apply
    backup = not args.no_backup

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No files will be modified")
        print("Use --apply to actually modify files")
        print("=" * 60)
    else:
        print("=" * 60)
        print("LIVE MODE - Files will be modified" + (" (with backups)" if backup else ""))
        print("=" * 60)

    grand_total = {
        "files_processed": 0,
        "files_modified": 0,
        "errors_by_type": defaultdict(lambda: {"kept": 0, "removed": 0}),
    }

    for directory in args.directories:
        # Handle relative paths
        if not os.path.isabs(directory):
            # Try from current directory first
            if os.path.exists(directory):
                pass
            # Try from workspace root
            elif os.path.exists(os.path.join(os.path.dirname(__file__), "..", directory)):
                directory = os.path.join(os.path.dirname(__file__), "..", directory)

        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue

        stats = process_directory(directory, dry_run=dry_run, backup=backup, verbose=args.verbose)

        grand_total["files_processed"] += stats["files_processed"]
        grand_total["files_modified"] += stats["files_modified"]
        for error_type, counts in stats["errors_by_type"].items():
            grand_total["errors_by_type"][error_type]["kept"] += counts["kept"]
            grand_total["errors_by_type"][error_type]["removed"] += counts["removed"]

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed: {grand_total['files_processed']}")
    print(f"Files with changes: {grand_total['files_modified']}")

    if grand_total["errors_by_type"]:
        print("\nBy error type:")
        for error_type, counts in sorted(grand_total["errors_by_type"].items()):
            total = counts["kept"] + counts["removed"]
            print(f"  {error_type}:")
            print(f"    Kept: {counts['kept']} (first in each consecutive run)")
            print(f"    Removed: {counts['removed']} (consecutive duplicates)")
            print(f"    Total original: {total}")

    if dry_run:
        print("\n[DRY RUN] No files were modified. Use --apply to apply changes.")


if __name__ == "__main__":
    main()
