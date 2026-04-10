#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Compile and merge experiment results.

Two modes of operation:

1. Compile results to a single file:
    python compile_results.py "pi05_batch*" -o results.jsonl
    python compile_results.py "pi05_batch*" -o results.json
    python compile_results.py "pi05_batch*" -o results          # defaults to .jsonl

2. Merge folders (moves task subdirectories + merges results):
    python compile_results.py "pi05_batch*" --merge output_folder
    python compile_results.py "pi05_batch*" --merge output_folder --keep
"""

import argparse
import os
import shutil
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from robolab.core.logging.results import (
    load_episode_results,
    save_episode_results_jsonl,
)
from robolab.core.utils.file_utils import (
    confirm_folders,
    expand_folder_patterns,
    get_folders_in_dir,
    save_json,
)


def episode_matches_task_filter(ep: dict, task_filter: str | None) -> bool:
    """Check if an episode matches the task filter.

    Args:
        ep: Episode dictionary with fields like 'reason', 'env_name', 'success'.
        task_filter: Filter type. Options:
            - None: No filtering, all episodes match.
            - "wrong object": Match episodes where reason contains "wrong object".

    Returns:
        True if episode matches the filter, False otherwise.
    """
    if task_filter is None:
        return True

    if task_filter == "wrong object":
        reason = ep.get("reason", "")
        if reason and "wrong object" in reason.lower():
            return True
        return False

    # Unknown filter, don't match
    return False


def get_task_folders(folder: Path) -> list[str]:
    """Get list of task folder names."""
    return get_folders_in_dir(str(folder.resolve()))


def merge_and_deduplicate_episodes(
    folders: list[Path],
    task_filter: str | None = None,
) -> list[dict]:
    """Load and deduplicate episode results from multiple folders.

    Args:
        folders: List of source folders.
        task_filter: Optional filter for episodes.

    Returns:
        Deduplicated list of episode dicts.
    """
    merged = []
    seen_episodes: set[tuple[str, int]] = set()  # (task, episode) pairs
    duplicates_skipped = 0
    filtered_out = 0

    for folder in folders:
        episodes = load_episode_results(str(folder))
        for ep in episodes:
            env_name = ep.get("env_name")
            episode = ep.get("episode")

            # Apply task filter
            if not episode_matches_task_filter(ep, task_filter):
                filtered_out += 1
                continue

            # Skip if we've already seen this (env_name, episode) pair
            key = (env_name, episode)
            if key in seen_episodes:
                duplicates_skipped += 1
                continue

            seen_episodes.add(key)
            merged.append(ep)

    filter_msg = f", {filtered_out} filtered out" if task_filter else ""
    print(f"Compiled {len(merged)} episodes ({duplicates_skipped} duplicates skipped{filter_msg})")

    return merged


def regenerate_results(
    episode_results: list[dict],
    output_folder: Path,
) -> None:
    """Regenerate results.json from episode results."""
    results = {
        "success": [],
        "failure": [],
    }
    env_results = defaultdict(lambda: {"success": [], "failure": []})

    for ep in episode_results:
        env_name = ep["env_name"]
        episode = ep["episode"]
        success = ep["success"]

        key = f"{env_name}_{episode}"

        if success:
            results["success"].append(key)
            env_results[env_name]["success"].append(episode)
        else:
            results["failure"].append(key)
            env_results[env_name]["failure"].append(episode)

    # Add per-env results to main results dict
    for env_name, env_data in sorted(env_results.items()):
        results[env_name] = {
            "success": sorted(env_data["success"]),
            "failure": sorted(env_data["failure"]),
        }

    output_path = output_folder / "results.json"
    save_json(results, output_path)
    print(f"Regenerated results.json: {len(env_results)} environments")


def move_env_folder(
    env_name: str,
    source_folder: Path,
    output_folder: Path,
) -> bool:
    """Move an environment folder from source to output. Returns True if successful."""
    source_env_dir = (source_folder / env_name).resolve()
    output_env_dir = (output_folder / env_name).resolve()

    if not source_env_dir.exists():
        return False

    output_env_dir.mkdir(parents=True, exist_ok=True)

    for file in source_env_dir.iterdir():
        if file.is_file():
            shutil.move(file, output_env_dir / file.name)

    # Remove the empty source directory
    try:
        source_env_dir.rmdir()
    except OSError:
        pass  # Directory not empty or other error, ignore

    return True


def check_conflicts(folders: list[Path]) -> set[str]:
    """Check for task folders that appear in multiple source folders.

    Returns:
        Set of conflicting task names. Empty if no conflicts.
    """
    task_counts: Counter[str] = Counter()
    for folder in folders:
        tasks = set(get_task_folders(folder))
        task_counts.update(tasks)

    return {task for task, count in task_counts.items() if count > 1}


def compile_to_file(
    folders: list[Path],
    output_path: Path,
    task_filter: str | None = None,
) -> None:
    """Compile episode results from multiple folders into a single output file.

    Output format determined by file extension: .json = JSON array, .jsonl or other = JSONL.
    """
    episodes = merge_and_deduplicate_episodes(folders, task_filter=task_filter)

    # Determine output format from extension
    ext = output_path.suffix.lower()
    if ext == ".json":
        save_json(episodes, output_path)
        print(f"Wrote {len(episodes)} episodes to {output_path} (JSON array)")
    else:
        # Default to JSONL
        if ext != ".jsonl":
            output_path = output_path.with_suffix(".jsonl")
        save_episode_results_jsonl(str(output_path), episodes)
        print(f"Wrote {len(episodes)} episodes to {output_path} (JSONL)")


def merge_folders(
    folders: list[Path],
    output_folder: Path,
    keep_sources: bool = False,
    task_filter: str | None = None,
) -> None:
    """Merge multiple experiment results folders into one.

    Aborts if any task folder appears in multiple sources (conflict).
    Moves task subdirectories and merges episode results into output.
    Removes source folders after successful merge (unless --keep).
    """
    # Check for conflicts
    conflicts = check_conflicts(folders)
    if conflicts:
        print("ERROR: Conflicting tasks found in multiple source folders:")
        for task in sorted(conflicts):
            print(f"  - {task}")
        print("\nAborting merge. Resolve conflicts before merging.")
        return

    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)

    # Print summary
    total_tasks = 0
    for i, folder in enumerate(folders):
        tasks = get_task_folders(folder)
        total_tasks += len(tasks)
        print(f"Folder {i + 1} ({folder.name}): {len(tasks)} tasks")
    print(f"Total tasks to move: {total_tasks}")

    # Move task folders
    for i, folder in enumerate(folders):
        tasks = get_task_folders(folder)
        for env_name in sorted(tasks):
            move_env_folder(env_name, folder, output_folder)
            print(f"  Moved from folder {i + 1}: {env_name}")

    # Merge episode results
    episodes = merge_and_deduplicate_episodes(folders, task_filter=task_filter)
    output_jsonl = output_folder / "episode_results.jsonl"
    save_episode_results_jsonl(str(output_jsonl), episodes)
    print(f"Wrote {output_jsonl}")

    # Regenerate results.json
    regenerate_results(episodes, output_folder)

    # Remove source folders (unless --keep)
    if not keep_sources:
        for folder in folders:
            if folder.resolve() != output_folder.resolve():
                shutil.rmtree(folder)
                print(f"Removed source folder: {folder}")

    print(f"\nMerge complete! Output: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile and merge experiment results"
    )
    parser.add_argument(
        "folders",
        type=str,
        nargs="+",
        help="Folders to compile/merge. Supports glob patterns (e.g., 'pi05_batch*')",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path for compile mode (e.g., results.jsonl, results.json). "
             "Extension determines format: .json = JSON array, .jsonl = JSONL (default).",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        default=None,
        metavar="OUTPUT_FOLDER",
        help="Merge mode: move task folders and results into OUTPUT_FOLDER.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep source folders after merge (default: remove them)",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt when using glob patterns",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["wrong object"],
        default=None,
        help="Filter episodes by task type.",
    )

    args = parser.parse_args()

    # Validate: must specify -o or --merge
    if args.output is None and args.merge is None:
        parser.error("Must specify either -o (compile) or --merge (merge folders)")
    if args.output is not None and args.merge is not None:
        parser.error("Cannot specify both -o and --merge")

    # Expand glob patterns
    folder_strs, pattern_expanded = expand_folder_patterns(args.folders)

    # Validate folders exist
    for f in folder_strs:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Folder not found: {f}")

    # Print expanded folders
    print(f"Folders ({len(folder_strs)}):")
    for f in folder_strs:
        print(f"  - {f}")

    # Confirm if patterns were expanded (unless -y flag is set)
    if pattern_expanded and not args.yes:
        folder_strs = confirm_folders(folder_strs, default_yes=False)
        if not folder_strs:
            return

    folders = [Path(f) for f in folder_strs]

    if args.output is not None:
        # Compile mode
        compile_to_file(folders, args.output, task_filter=args.task)
    else:
        # Merge mode
        merge_folders(
            folders,
            args.merge,
            keep_sources=args.keep,
            task_filter=args.task,
        )


if __name__ == "__main__":
    main()
