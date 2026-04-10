# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os
from contextlib import redirect_stdout

from robolab.constants import BENCHMARK_TASK_CATEGORIES, DEFAULT_OUTPUT_DIR
from robolab.core.logging.results import (
    filter_episodes_by_pattern,
    filter_episodes_by_task,
    load_and_merge_episode_data,
    summarize_experiment_results,
    summarize_experiments_by_category_with_attributes,
    summarize_experiments_by_instruction_type,
    summarize_experiments_by_remapped_attributes,
    summarize_experiments_by_scene,
    summarize_experiments_by_wrong_objects,
    summarize_task_results,
)
from robolab.core.utils.file_utils import confirm_folders, expand_folder_patterns, load_file  # noqa: F401 (load_file)


def main():
    parser = argparse.ArgumentParser(description="Read and summarize experiment results")
    parser.add_argument("folder", nargs='+', help="Folder name(s) or glob pattern(s) under robolab/output (e.g., 'pi0_*'), or absolute path(s). Glob patterns prompt for confirmation.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output (shows stddev columns, wrong objects, and episode IDs)")
    parser.add_argument("--show-episodes", action="store_true", default=False, help="Show the episodes in the results for each task")
    parser.add_argument("--task", type=str, nargs='+', default=None, help="Name(s) of the task(s) to summarize")
    parser.add_argument("--by-attributes", action="store_true", default=False, help="Summarize results in one table with categories (visual, relational, procedural) and their attributes grouped together")
    parser.add_argument("--by-remapped-attributes", action="store_true", default=False, help="Summarize results by remapped attribute categories (visual, relational, procedural)")
    parser.add_argument("--by-scene", action="store_true", default=False, help="Summarize results by scene instead of by task")
    parser.add_argument("--by-wrong-objects", action="store_true", default=False, help="Summarize each task with columns on wrong objects: success count, fail count, and which objects were grabbed")
    parser.add_argument("--by-instruction-type", action="store_true", default=False, help="Compare task success rates across instruction types (default, vague, specific, etc.) in a pivot table")
    parser.add_argument("--csv", action="store_true", default=False, help="If true, show the results in CSV format for copy and pasting")
    parser.add_argument("--csv-compact", action="store_true", default=False, help="CSV mode with stddev in same column as value, e.g., '-9.14 (± 4.72)' (implies --csv)")
    parser.add_argument("--output-csv", type=str, default=None, metavar="FILE", help="Write CSV output to the specified file (implies --csv)")
    parser.add_argument("--filter-pattern", type=str, default=None, help="Glob-style pattern to filter results by env_name (e.g., 'pick_*' or '*cube*')")
    parser.add_argument("--filter-field", type=str, default="env_name", help="Field to filter results by (e.g., 'env_name', 'task_name', 'scene', 'attributes')")
    parser.add_argument("--no-metrics", action="store_true", default=False, help="Hide trajectory metrics columns (EE SPARC, Path Length, Speed)")
    parser.add_argument("--timing", action="store_true", default=False, help="Show wall-clock timing columns (it/s, Wall(s))")
    parser.add_argument("--exclude-containers", action="store_true", default=False, help="Exclude container objects (bin, crate, box, etc.) from wrong object grabbed counts")
    args = parser.parse_args()

    folders, pattern_expanded = expand_folder_patterns(args.folder, base_dir=DEFAULT_OUTPUT_DIR)

    if not folders:
        parser.error("No folders found matching the given arguments.")

    if pattern_expanded:
        print(f"Found {len(folders)} folder(s):")
        for f in folders:
            print(f"  - {os.path.basename(f)}")

        folders = confirm_folders(folders)
        if not folders:
            return
        print()

    episode_results = []
    for folder_path in folders:

        # Load and merge episode results with metrics
        folder_episodes = load_and_merge_episode_data(folder_path)
        episode_results.extend(folder_episodes)

    # Filter episodes by pattern if provided

    if args.task is not None:
        episode_results = filter_episodes_by_task(episode_results, args.task)

    if args.filter_pattern is not None:
        episode_results = filter_episodes_by_pattern(episode_results, args.filter_pattern, field=args.filter_field)

    # Metrics shown by default, can be hidden with --no-metrics
    show_metrics = not args.no_metrics

    # If --output-csv or --csv-compact is specified, implies --csv
    use_csv = args.csv or args.output_csv is not None or args.csv_compact

    # If --output-csv is not an absolute path, default to the first data folder
    output_csv_path = args.output_csv
    if output_csv_path and not os.path.isabs(output_csv_path):
        output_csv_path = os.path.join(folders[0], output_csv_path)

    def run_summarization():
        if args.by_attributes:
            # Print combined category + attributes table
            summarize_experiments_by_category_with_attributes(episode_results=episode_results, remap=BENCHMARK_TASK_CATEGORIES, VERBOSE=args.verbose, csv=use_csv, show_metrics=show_metrics, csv_compact=args.csv_compact)
        elif args.by_remapped_attributes:
            summarize_experiments_by_remapped_attributes(episode_results=episode_results, remap=BENCHMARK_TASK_CATEGORIES, VERBOSE=args.verbose, csv=use_csv, csv_compact=args.csv_compact)
        elif args.by_scene:
            summarize_experiments_by_scene(episode_results=episode_results, VERBOSE=args.verbose, csv=use_csv, csv_compact=args.csv_compact)
        elif args.by_wrong_objects:
            summarize_experiments_by_wrong_objects(episode_results=episode_results, exclude_containers=args.exclude_containers, csv=use_csv)
        elif args.by_instruction_type:
            summarize_experiments_by_instruction_type(episode_results=episode_results, VERBOSE=args.verbose, csv=use_csv, csv_compact=args.csv_compact, show_metrics=show_metrics)
        else:
            summarize_experiment_results(episode_results=episode_results, VERBOSE=args.verbose, csv=use_csv, exclude_containers=args.exclude_containers, show_metrics=show_metrics, show_timing=args.timing, csv_compact=args.csv_compact)

        if args.show_episodes:
            summarize_task_results(episode_results, VERBOSE=args.verbose, csv=use_csv, csv_compact=args.csv_compact)

    # If --output-csv is specified, write to file
    if output_csv_path:
        with open(output_csv_path, 'w') as f:
            with redirect_stdout(f):
                run_summarization()
        print(f"CSV output written to: {output_csv_path}")
    else:
        run_summarization()


if __name__ == "__main__":
    main()
