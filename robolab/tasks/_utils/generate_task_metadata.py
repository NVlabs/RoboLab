# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Generate Task Metadata

This script scans a folder containing task definition files and generates metadata
summaries in multiple formats (JSON, CSV, and Markdown).

Usage:
    Run as a script to generate metadata for all tasks in the tasks folder:

    # Use default paths (scan TASK_DIR, output to TASK_DIR/_metadata)
    python generate_task_metadata.py

    # Specify custom paths
    python generate_task_metadata.py --tasks-folder /path/to/tasks --output-folder /path/to/output

    # Filter by specific subfolders
    python generate_task_metadata.py --subfolders ycb hope handal

    # Import as a module
    from robolab.tasks._utils.generate_task_metadata import generate_task_metadata
    generate_task_metadata("/path/to/tasks", "/path/to/output", subfolders=["ycb", "hope"])

Command Line Arguments:
    --tasks-folder: Path to the folder containing task definition files (default: TASK_DIR)
    --output-folder: Path where output files will be saved (default: TASK_DIR/_metadata)
    --subfolders: List of subfolder names to include (e.g., --subfolders ycb hope)
    --include-images: Include images in the markdown table (default: True)

Output Files:
    - task_metadata.json: Complete metadata for all tasks in JSON format
    - task_table.csv: Task metadata in CSV table format
    - README.md: Formatted markdown table saved to the tasks folder

The script extracts the following metadata from each task:
    - task_name: Name of the task class
    - instruction: Task description/instruction
    - episode_s: Episode duration in seconds
    - scene: Associated scene name
    - filename: Source file path (relative to tasks folder)
    - subfolder: Collection/subfolder the task belongs to
    - contact_objects: Objects involved in contact interactions
    - num_sequential_stages: Number of sequential stages
    - num_subtasks: Total number of subtasks (manipulation actions)
    - num_atomic_conditions: Total number of atomic condition checks
    - subtasks: List of all subtasks

Note: Files in folders named "not_used" or starting with "_" are automatically excluded.
"""

import os
import json
import csv
from typing import Dict, List, Any
from robolab.constants import SCENE_DIR

YELLOW = "\033[33m"
RESET = "\033[0m"

def _format_instruction_for_display(task_data: Dict[str, Any]) -> str:
    """Format instruction(s) for CSV/markdown display.

    If the task has instruction_variants (a dict), show all variants with
    type labels.  Otherwise return the plain instruction string.
    """
    variants = task_data.get('instruction_variants')
    if variants and isinstance(variants, dict):
        parts = []
        for key, text in variants.items():
            parts.append(f"**{key}:** {text}")
        return "<br>".join(parts)
    return str(task_data.get('instruction', ''))


def convert_task_results_to_csv(results: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Convert task results to CSV format.

    Args:
        results: List of task metadata dictionaries

    Returns:
        List of CSV rows
    """
    if not results:
        return []

    # Define headers in desired order
    headers = [
        'task_name',
        'scene',
        'instruction',
        'episode_s',
        'attributes',
        'num_subtasks',
        'difficulty_label',
    ]

    csv_rows = [headers]

    for task_data in results:
        # Extract subfolder from filename (subfolder name)
        filename = task_data.get('filename', '')

        subfolder = ''
        if '/' in filename:
            subfolder = filename.split('/')[0]

        row = []
        for header in headers:
            if header == 'subfolder':
                row.append(subfolder)
            elif header == 'task_name':
                task_name = str(task_data.get(header, ''))
                row.append(f"{task_name} ({filename})")
            elif header == 'instruction':
                row.append(_format_instruction_for_display(task_data))
            else:
                row.append(str(task_data.get(header, '')))
        csv_rows.append(row)

    return csv_rows


def generate_task_metadata(tasks_folder: str, output_folder: str = None, include_images: bool = False, subfolders: List[str] = None):
    """
    Generate task metadata for all tasks in the tasks folder.

    Args:
        tasks_folder: Path to the tasks folder
        output_folder: Path to save output files (defaults to tasks_folder)
        include_images: Whether to include images in the markdown table
        subfolders: List of subfolder names to include (if None, include all subfolders)
    """
    if output_folder is None:
        output_folder = tasks_folder

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    from robolab.tasks._utils.load_task_info import scan_tasks_folder
    # Use task_utils to scan the tasks folder, filtering by subfolders if specified
    if subfolders is not None:
        print(f"Filtering tasks by subfolders: {', '.join(subfolders)}")

    results = scan_tasks_folder(tasks_folder, subfolders=subfolders)

    if not results:
        print("No task classes found or processed successfully.")
        return

    # Sort results alphabetically by task name
    results.sort(key=lambda x: x.get('task_name', '').lower())

    # Check for duplicate task names
    from collections import Counter
    task_names = [r.get('task_name', '') for r in results]
    duplicates = {name: count for name, count in Counter(task_names).items() if count > 1}
    if duplicates:
        print(f"\n{YELLOW}WARNING: Found {len(duplicates)} duplicate task name(s):{RESET}")
        for name, count in duplicates.items():
            files = [r.get('filename', '?') for r in results if r.get('task_name') == name]
            print(f"  {YELLOW}{name} (x{count}): {', '.join(files)}{RESET}")
        print()

    # Save results to JSON file
    json_output_path = os.path.join(output_folder, "task_metadata.json")
    try:
        with open(json_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved to: {json_output_path}")
    except Exception as e:
        print(f"Error saving JSON results: {e}")

    # Convert to CSV format
    csv_rows = convert_task_results_to_csv(results)

    # Save CSV file
    csv_output_path = os.path.join(output_folder, "task_table.csv")
    try:
        with open(csv_output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"CSV results saved to: {csv_output_path}")
    except Exception as e:
        print(f"Error saving CSV results: {e}")

    image_dir = os.path.join(SCENE_DIR, '_images')
    if include_images and os.path.isdir(image_dir):
        from robolab.core.utils.csv_utils import add_images_to_csv
        csv_rows = add_images_to_csv(csv_output_path, image_dir=image_dir, column_name_to_img='scene', image_column_name='image', relative_dir=tasks_folder, replace_column=True, size=(400,None))

    markdown_output_path = os.path.join(tasks_folder, "README.md")
    try:
        # Create description with total task count
        total_tasks = len(results)

        # Build description with subfolder info if filtered
        if subfolders is not None:
            subfolder_list = ", ".join(subfolders)
            description = f"This table contains metadata for tasks in `{tasks_folder}`.\n\n**Filtered by subfolders:** {subfolder_list}\n\n**Total Tasks: {total_tasks}**"
        else:
            description = f"This table contains metadata for all tasks in `{tasks_folder}`.\n\n**Total Tasks: {total_tasks}**"

        save_markdown_table(
            csv_rows,
            markdown_output_path,
            title="Available Tasks",
            description=description,
            align="left",
            path_type="filename_only"
        )
    except Exception as e:
        print(f"Error saving markdown table: {e}")



if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    from robolab.core.utils.csv_utils import save_markdown_table
    import argparse
    from robolab.constants import TASK_DIR, DEFAULT_TASK_SUBFOLDERS
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate metadata table for all tasks in the tasks folder")
    parser.add_argument("--tasks-folder", default=TASK_DIR,
                       help="Path to the tasks folder")
    parser.add_argument("--output-folder", default=None,
                       help="Path to save output files (defaults to <tasks-folder>/_metadata)")
    parser.add_argument("--include-images", action="store_true", default=True, help="Include images in the markdown table")
    parser.add_argument("--subfolders", nargs="+", default=None,
                       help="List of subfolder names to include (e.g., --subfolders ycb hope). If not specified, all subfolders are included. When using the default --tasks-folder, defaults to DEFAULT_TASK_SUBFOLDERS.")
    args = parser.parse_args()

    subfolders = args.subfolders
    if subfolders is None and os.path.samefile(args.tasks_folder, TASK_DIR):
        subfolders = DEFAULT_TASK_SUBFOLDERS

    output_folder = args.output_folder
    if output_folder is None:
        output_folder = os.path.join(args.tasks_folder, "_metadata")

    generate_task_metadata(args.tasks_folder, output_folder, args.include_images, subfolders)

    simulation_app.close()
