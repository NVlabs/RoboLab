# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

import os
import json
import csv
from typing import Dict, List, Any
from tqdm import tqdm
from robolab.constants import SCENE_DIR


YELLOW = "\033[33m"
RESET = "\033[0m"

def verify_tasks_valid(tasks_folder: str):
    """
    Verify all tasks in the tasks folder are valid.

    Args:
        tasks_folder: Path to the tasks folder
    """
    from robolab.core.task.task_utils import find_task_files, load_task_from_file
    from robolab.core.task.task import verify_task_valid

    folder_name = os.path.basename(tasks_folder)
    task_files = find_task_files(tasks_folder)

    num_invalid = 0
    task_name_to_files = {}
    errors = []

    for task in tqdm(task_files, desc=f"Checking '{folder_name}'", unit="task"):
        try:
            task_class = load_task_from_file(task, allow_multiple=False)
        except Exception as e:
            errors.append(f"{YELLOW}Task {task} failed to load: {e}{RESET}")
            num_invalid += 1
            continue
        valid, error = verify_task_valid(task_class)
        if not valid:
            errors.append(f"{YELLOW}Task {task} is not valid: {error}{RESET}")
            num_invalid += 1

        name = getattr(task_class, '__name__', str(task_class))
        task_name_to_files.setdefault(name, []).append(task)

    # Print errors after progress bar completes
    for err in errors:
        print(f"\t{err}")

    # Check for duplicate task names
    duplicates = {name: files for name, files in task_name_to_files.items() if len(files) > 1}
    if duplicates:
        print(f"\t{YELLOW}WARNING: Found {len(duplicates)} duplicate task name(s):{RESET}")
        for name, files in duplicates.items():
            print(f"\t  {YELLOW}{name} (x{len(files)}):{RESET}")
            for f in files:
                print(f"\t    {YELLOW}{f}{RESET}")
        num_invalid += len(duplicates)

    if num_invalid > 0:
        print(f"\t{YELLOW}{num_invalid}/{len(task_files)} tasks are invalid.{RESET}")
    else:
        print(f"\tAll {len(task_files)} tasks verified successfully.")



if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    import argparse
    from robolab.constants import TASK_DIR, DEFAULT_TASK_SUBFOLDERS
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate metadata table for all tasks in the tasks folder")
    parser.add_argument("--tasks-folder", default=None,
                       help="Path to the tasks folder (if not provided, uses default task subfolders)")
    args = parser.parse_args()

    if args.tasks_folder:
        verify_tasks_valid(args.tasks_folder)
    else:
        for subfolder in DEFAULT_TASK_SUBFOLDERS:
            tasks_folder = os.path.join(TASK_DIR, subfolder)
            if os.path.isdir(tasks_folder):
                verify_tasks_valid(tasks_folder)
            else:
                print(f"Warning: subfolder '{subfolder}' not found at {tasks_folder}, skipping.")

    simulation_app.close()
