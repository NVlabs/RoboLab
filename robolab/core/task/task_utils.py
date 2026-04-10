# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import importlib.util
import os
from pathlib import Path
from typing import List, Optional

from robolab.core.task.task import Task

# Cache for resolve_task_path to avoid repeated expensive lookups
_resolve_task_cache: dict[tuple[str, str], tuple[str, str]] = {}

# Cache for loaded task modules to avoid re-importing the same file
_loaded_modules_cache: dict[str, object] = {}

# Cache for task classes extracted from modules
_task_classes_cache: dict[str, list] = {}


def load_task_from_file(task_file_path: str, allow_multiple: bool = False) -> Task | list[Task]:
    """
    Load a Task class from a Python file. If allow_multiple is True, return a list of Task classes contained in the file.
    Results are cached to avoid re-importing the same file multiple times.

    Args:
        task_file_path: Path to the task file (e.g., 'sauce_bottles_crate.py')

    Returns:
        The Task class from the file
    """
    # Normalize path for consistent cache keys
    normalized_path = os.path.abspath(task_file_path)

    # Check if we already have the task classes cached
    if normalized_path in _task_classes_cache:
        task_classes = _task_classes_cache[normalized_path]
        if not allow_multiple:
            return task_classes[0]
        return task_classes

    # Check if module is already loaded
    if normalized_path in _loaded_modules_cache:
        module = _loaded_modules_cache[normalized_path]
    else:
        # Get the module name from the file path
        module_name = os.path.splitext(os.path.basename(task_file_path))[0]

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, task_file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module from {task_file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Cache the loaded module
        _loaded_modules_cache[normalized_path] = module

    # Find the Task class in the module
    task_classes = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and
            issubclass(attr, Task) and
            attr != Task):
            task_classes.append(attr)

    if not task_classes:
        raise ValueError(f"No Task subclass found in {task_file_path} of type {type(Task)}")

    # Cache the task classes
    _task_classes_cache[normalized_path] = task_classes

    if not allow_multiple:
        return task_classes[0]
    return task_classes


def clear_task_cache():
    """Clear all task-related caches. Call this if task files have been modified."""
    global _resolve_task_cache, _loaded_modules_cache, _task_classes_cache
    _resolve_task_cache.clear()
    _loaded_modules_cache.clear()
    _task_classes_cache.clear()


def find_task_files(tasks_folder: str,
                    subfolders: List[str] = None,
                    exclude_patterns: Optional[List[str]] = None,
                    exclude_folders: list[str] = ['tmp', 'not_used', '__pycache__']
                    ) -> List[str]:
    """
    Find all Python task files in the tasks folder and its subfolders.

    Args:
        tasks_folder: Path to the tasks folder
        subfolders: List of subfolder names to include (if None, include all subfolders)
        exclude_patterns: List of patterns to exclude (defaults to common non-task files)
        exclude_folders: List of folder names to exclude

    Returns:
        List of task file paths
    """
    if exclude_patterns is None:
        exclude_patterns = ['__init__.py']

    # Don't exclude folders if they're explicitly in the subfolders list
    if subfolders is not None:
        exclude_folders = [f for f in exclude_folders if f not in subfolders]
    else:
        # Ensure default exclusions are in place
        for folder in ['tmp', 'not_used', '__pycache__']:
            if folder not in exclude_folders:
                exclude_folders.append(folder)

    task_files = []

    if not os.path.exists(tasks_folder):
        raise ValueError(f"Tasks folder '{tasks_folder}' does not exist.")

    # Walk through the directory tree recursively
    for root, dirs, files in os.walk(tasks_folder):
        # Calculate relative path from tasks_folder
        rel_path = os.path.relpath(root, tasks_folder)

        # If subfolders is specified, check if we're in an allowed subfolder
        if subfolders is not None and rel_path != '.':
            # Get the top-level subfolder name
            top_level_subfolder = rel_path.split(os.sep)[0]
            if top_level_subfolder not in subfolders:
                # Skip this entire directory tree
                dirs[:] = []
                continue

        # Skip __pycache__ directories and other unwanted folders
        dirs[:] = [d for d in dirs if not d.startswith('_') and
                   d not in exclude_folders]

        for filename in files:
            if (filename.endswith('.py') and
                not filename.startswith('.') and
                filename not in exclude_patterns):

                task_files.append(os.path.join(root, filename))

    return task_files


def get_task_class_name_from_file(task_file_path: str) -> str:
    """
    Get the Task class name from a task file.

    Args:
        task_file_path: Path to the task file

    Returns:
        Name of the first Task subclass found in the file

    Raises:
        ValueError: If no Task subclass is found
    """
    task_class = load_task_from_file(task_file_path, allow_multiple=False)
    return task_class.__name__  # type: ignore[union-attr]


def resolve_task_path(task: str, task_dir: str | Path) -> tuple[str, str]:
    """
    Resolve a task identifier to a full file path and Task class name.

    Handles three cases:
    1. Full file path (contains '/' or '\\') - use directly
    2. Filename ending in '.py' - attach to task_dir
    3. Task name - search recursively in task_dir for matching file

    Args:
        task: Task identifier (path, filename, or task name)
        task_dir: Directory to search for task files

    Returns:
        Tuple of (file_path, task_class_name)

    Raises:
        FileNotFoundError: If the task file cannot be found

    Examples:
        resolve_task_path("/path/to/BananaTask.py", task_dir)
        # Returns ("/path/to/BananaTask.py", "BananaTask")

        resolve_task_path("BananaTask.py", task_dir)
        # Returns ("/full/path/to/BananaTask.py", "BananaTask")

        resolve_task_path("BananaTask", task_dir)
        # Returns ("/full/path/to/BananaTask.py", "BananaTask")
    """
    # Check cache first
    cache_key = (task, str(task_dir))
    if cache_key in _resolve_task_cache:
        return _resolve_task_cache[cache_key]

    task_dir = Path(task_dir)

    # Case 1: Full file path
    if '/' in task or '\\' in task:
        if not Path(task).exists():
            raise FileNotFoundError(f"Task file not found: {task}")
        task_file_path = task
        task_class_name = get_task_class_name_from_file(task_file_path)
        result = (task_file_path, task_class_name)
        _resolve_task_cache[cache_key] = result
        return result

    # Case 2: Filename ending in .py (but not a full path)
    elif task.endswith('.py'):
        candidate = task_dir / task
        if candidate.exists():
            task_file_path = str(candidate)
        else:
            # Also try searching recursively
            matches = list(task_dir.rglob(task))
            if matches:
                task_file_path = str(matches[0])
            else:
                raise FileNotFoundError(f"Task file not found: {task} in {task_dir}")
        task_class_name = get_task_class_name_from_file(task_file_path)
        result = (task_file_path, task_class_name)
        _resolve_task_cache[cache_key] = result
        return result

    # Case 3: Task class name - searches in task_dir for all files, and loop through each file to find the class name.
    else:
        task_class_name_to_find = task
        all_task_files = find_task_files(str(task_dir))

        for candidate_file in all_task_files:
            try:
                task_classes = load_task_from_file(candidate_file, allow_multiple=False)
                if isinstance(task_classes, list):
                    for cls in task_classes:
                        if cls.__name__ == task_class_name_to_find:  # type: ignore[union-attr]
                            result = (candidate_file, task_class_name_to_find)
                            _resolve_task_cache[cache_key] = result
                            return result
                else:
                    if task_classes.__name__ == task_class_name_to_find:  # type: ignore[union-attr]
                        result = (candidate_file, task_class_name_to_find)
                        _resolve_task_cache[cache_key] = result
                        return result
            except (ValueError, Exception):
                # Skip files that can't be loaded
                continue

        raise FileNotFoundError(
            f"Task class '{task}' not found in any file in {task_dir}"
        )
