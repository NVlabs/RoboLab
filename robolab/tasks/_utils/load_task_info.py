# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
from functools import partial
from typing import Any

from robolab.constants import SCENE_DIR
from robolab.core.task.subtask_utils import (
    compute_difficulty_score,
    count_stages_and_conditions,
    count_subtasks,
    process_subtasks_as_str,
)
from robolab.core.task.task import Task, resolve_instruction
from robolab.core.task.task_utils import find_task_files, load_task_from_file


def extract_task_metadata(task_class: type[Task], filepath: str, tasks_folder: str) -> dict [str, Any]:
    """
    Extract metadata from a task class using the standard Task properties.

    Args:
        task_class: The task class object
        filepath: Full path to the task file
        tasks_folder: Root tasks folder path for relative path calculation

    Returns:
        Dictionary with task metadata
    """
    # Calculate relative path from tasks folder
    rel_path = os.path.relpath(filepath, tasks_folder)

    raw_instruction = getattr(task_class, 'instruction', '')
    metadata = {
        'task_name': task_class.__name__,
        'instruction': resolve_instruction(raw_instruction) if isinstance(raw_instruction, dict) else raw_instruction,
        'instruction_variants': raw_instruction if isinstance(raw_instruction, dict) else None,
        'episode_s': str(getattr(task_class, 'episode_length_s', '')),
        'scene': '',
        'filename': rel_path,
        'contact_objects': '',
        'terminations': '',
        'attributes': '',
        'subtasks': '',
        'num_sequential_stages': 0,
        'num_atomic_conditions': 0,
        'num_subtasks': 0,
        'difficulty_score': 0,
        'difficulty_label': 'simple',
    }

    try:
        # Extract contact objects
        contact_objects = getattr(task_class, 'contact_object_list', None)
        if contact_objects and isinstance(contact_objects, list):
            metadata['contact_objects'] = ', '.join(contact_objects)
        elif contact_objects:
            metadata['contact_objects'] = str(contact_objects)

        # Extract scene information
        scene = getattr(task_class, 'scene', None)
        if scene:
            if getattr(scene, 'scene', None):
                if hasattr(scene.scene, 'spawn'):
                    usd_path = scene.scene.spawn.usd_path
                    if usd_path.startswith(SCENE_DIR):
                        metadata['scene'] = os.path.relpath(usd_path, SCENE_DIR)
                    else:
                        metadata['scene'] = usd_path
            else:
                metadata['scene'] = scene.__name__

        # Extract subtasks information
        subtasks = getattr(task_class, 'subtasks', None)
        metadata['num_sequential_stages'], metadata['num_atomic_conditions'] = count_stages_and_conditions(subtasks)
        metadata['num_subtasks'] = count_subtasks(subtasks)
        metadata['subtasks'] = process_subtasks_as_str(subtasks)

        # Extract termination information
        terminations = getattr(task_class, 'terminations', None)
        if terminations:
            metadata['terminations'] = terminations.__name__

        # Extract attributes information
        attributes = getattr(task_class, 'attributes', None)
        if attributes and isinstance(attributes, list):
            metadata['attributes'] = ', '.join(attributes)
        elif attributes:
            metadata['attributes'] = str(attributes)

        # Compute difficulty score from num_subtasks + attribute skill weights
        attr_list = attributes if isinstance(attributes, list) else []
        score, label = compute_difficulty_score(metadata['num_subtasks'], attr_list)
        metadata['difficulty_score'] = score
        metadata['difficulty_label'] = label

    except Exception as e:
        print(f"Warning: Error extracting metadata from {task_class.__name__}: {e}")

    return metadata

def extract_task_metadata_from_file(task_file: str, tasks_folder: str) -> dict [str, Any]:
    """
    Extract metadata from a task file.
    """
    task_class = load_task_from_file(task_file, allow_multiple=False)
    return extract_task_metadata(task_class, task_file, tasks_folder)

def scan_tasks_folder(tasks_folder: str, subfolders: list[str] = None) -> list[dict [str, Any]]:
    """
    Scan a tasks folder and extract metadata from all task files.

    Args:
        tasks_folder: Path to the tasks folder
        subfolders: List of subfolder names to include (if None, include all subfolders)

    Returns:
        list of task metadata dictionaries
    """
    print(f"Scanning task files in: {tasks_folder}")

    # Find all task files
    task_files = find_task_files(tasks_folder, subfolders=subfolders)
    print(f"Found {len(task_files)} task files")

    results = []

    for task_file in task_files:
        rel_path = os.path.relpath(task_file, tasks_folder)
        print(f"Processing: {rel_path}")

        try:
            metadata = extract_task_metadata_from_file(task_file, tasks_folder)

            results.append(metadata)

            for key, value in metadata.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"  Warning: Could not process {rel_path}: {e}")
            continue

    print(f"\nProcessed {len(results)} tasks successfully.")
    return results
