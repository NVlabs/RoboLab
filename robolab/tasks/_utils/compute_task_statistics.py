# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Compute Task Statistics

This script analyzes task metadata and computes statistics on task attributes,
showing the number of tasks per attribute and their distribution.
It can also group tasks by scene to show task distribution across scenes.

Usage:
    # Use default paths and subfolders
    python compute_task_statistics.py

    # Specify custom metadata file
    python compute_task_statistics.py --metadata-file /path/to/task_metadata.json

    # Filter by specific subfolders
    python compute_task_statistics.py --subfolders examples2

    # Show detailed breakdown with all tasks per attribute
    python compute_task_statistics.py --verbose

    # Show tasks grouped by scene instead of attributes
    python compute_task_statistics.py --by-scene

    # Show tasks by scene with additional task details (subfolder, attributes)
    python compute_task_statistics.py --by-scene --verbose

    # Show full comprehensive report (attributes, objects, subtasks, episodes, scenes)
    python compute_task_statistics.py --verbose

    # Show individual analysis sections
    python compute_task_statistics.py --objects
    python compute_task_statistics.py --subtasks
    python compute_task_statistics.py --episodes

Output:
    A formatted table showing:
    - (Default) Total unique attributes and tasks scanned
    - (Default) Number of tasks per attribute
    - (With --verbose) Comprehensive report: attributes, categories, objects, subtasks, episodes, scenes
    - (With --by-scene) Total scenes and tasks, with task count per scene and list of tasks
    - (With --by-scene --verbose) Same as above, but includes subfolder and attributes for each task
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from robolab.constants import BENCHMARK_TASK_CATEGORIES, DIFFICULTY_THRESHOLDS, SKILL_WEIGHTS

########################################################
# Loading and filtering
########################################################

def load_metadata(metadata_file: str) -> List[Dict[str, Any]]:
    """
    Load metadata from JSON file.
    """
    with open(metadata_file, 'r') as f:
        tasks_data = json.load(f)
        print(f"Analyzing tasks in {metadata_file}")

    for task in tasks_data:
        # Add subfolders if exists
        filename = task.get('filename', '')
        if '/' in filename:
            task['subfolder'] = filename.split('/')[0]
        else:
            task['subfolder'] = ''
    return tasks_data


def filter_by_subfolders(tasks_data: List[Dict[str, Any]], subfolders: List[str] = None) -> List[Dict[str, Any]]:
    """Filter tasks to only include those in the specified subfolders."""
    if subfolders is None:
        return tasks_data
    subfolder_to_tasks = sort_tasks_by_subfolder(tasks_data)
    filtered = []
    for folder in subfolders:
        filtered.extend(subfolder_to_tasks.get(folder, []))
    return filtered


########################################################
# Grouping helpers
########################################################

def sort_tasks_by_scene(tasks_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Generate a dictionary of scene names to tasks."""
    scene_to_tasks = defaultdict(list)
    for task in tasks_data:
        scene = task.get('scene', '')
        scene_to_tasks[scene].append(task)
    return dict(scene_to_tasks)


def sort_tasks_by_subfolder(tasks_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Generate a dictionary of subfolder names to tasks."""
    subfolder_to_tasks = defaultdict(list)
    for task in tasks_data:
        subfolder = task.get('subfolder', '')
        subfolder_to_tasks[subfolder].append(task)
    return dict(subfolder_to_tasks)


########################################################
# Parsing helpers
########################################################

def parse_comma_separated_string_into_list(string: str) -> List[str]:
    """
    Parse comma-separated string into a list, filtering empty entries.

    Args:
        string: Comma-separated string of attributes

    Returns:
        List of stripped, non-empty strings
    """
    if not string or string.strip() == "":
        return []
    return [item.strip() for item in string.split(",") if item.strip()]


def normalize_object_name(obj: str) -> str:
    """
    Normalize object name by removing numeric suffixes.
    E.g., 'banana_01' -> 'banana', 'mug_01' -> 'mug', 'lime01_01' -> 'lime01'
    """
    return re.sub(r'_\d+$', '', obj)


def colorize_attributes(attributes_str: str) -> str:
    """
    Colorize specific attributes with ANSI color codes.

    Args:
        attributes_str: Comma-separated string of attributes

    Returns:
        String with colored attributes
    """
    if not attributes_str:
        return ""

    color_map = {
        'complex': '\033[1;31m',    # Bold red
        'moderate': '\033[1;33m',   # Bold yellow/orange
        'simple': '\033[1;32m'      # Bold green
    }
    reset = '\033[0m'

    attrs = [attr.strip() for attr in attributes_str.split(',')]
    colored_attrs = []

    for attr in attrs:
        if attr in color_map:
            colored_attrs.append(f"{color_map[attr]}{attr}{reset}")
        else:
            colored_attrs.append(attr)

    return ', '.join(colored_attrs)


########################################################
# Attribute analysis
########################################################

def analyze_task_attributes(tasks_data: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, str]]], int, List[Dict[str, str]]]:
    """
    Analyze task attributes from a list of task metadata dicts.

    Args:
        tasks_data: List of task metadata dicts (already loaded and filtered)

    Returns:
        Tuple of (attribute_to_tasks dict, total_tasks_count, untagged_tasks list)
        where attribute_to_tasks maps attribute names to lists of task info dicts
        containing 'task_name', 'path', and 'filename' keys.
    """
    attribute_to_tasks = defaultdict(list)
    untagged_tasks = []

    for task in tasks_data:
        task_name = task.get('task_name', 'Unknown')
        filename = task.get('filename', '')
        path = task.get('path', '')
        attributes = parse_comma_separated_string_into_list(task.get('attributes', ''))

        task_info = {
            'task_name': task_name,
            'path': path,
            'filename': filename
        }

        if not attributes:
            untagged_tasks.append(task_info)
        else:
            for attr in attributes:
                attribute_to_tasks[attr].append(task_info)

    return dict(attribute_to_tasks), len(tasks_data), untagged_tasks


def print_attribute_summary(attribute_to_tasks: Dict[str, List[Dict[str, str]]],
                            total_tasks: int,
                            untagged_tasks: List[Dict[str, str]],
                            verbose: bool = False,
                            print_header: bool = True):
    """
    Print formatted attribute summary table.

    Args:
        attribute_to_tasks: Dictionary mapping attributes to list of task info dicts
                           (each dict has 'task_name', 'path', 'filename' keys)
        total_tasks: Total number of tasks scanned
        untagged_tasks: List of task info dicts with no attributes
        verbose: If True, show detailed breakdown with all tasks per attribute
        print_header: If True, print the header section
    """
    # Sort attributes by category
    # Define priority categories (shown first)
    priority_attrs = ['simple', 'moderate', 'complex']

    # Separate into priority and other attributes
    priority_items = []
    other_items = []

    for attr, tasks in attribute_to_tasks.items():
        if attr in priority_attrs:
            priority_items.append((attr, tasks))
        else:
            other_items.append((attr, tasks))

    # Sort priority by the defined order
    priority_items.sort(key=lambda x: priority_attrs.index(x[0]))

    # Sort other attributes alphabetically
    other_items.sort(key=lambda x: x[0])

    # Calculate statistics
    total_unique_attributes = len(attribute_to_tasks)

    # Print header
    if print_header:
        print("\n" + "=" * 100)
        print("TASK ATTRIBUTES SUMMARY")
        print(f"Total unique attributes: {total_unique_attributes}")
        print(f"Total tasks scanned: {total_tasks}")
        if untagged_tasks:
            print(f"Untagged tasks: {len(untagged_tasks)}")
        print("=" * 100)

    # Calculate column widths
    max_attr_len = max(len(attr) for attr in attribute_to_tasks.keys()) if attribute_to_tasks else 10
    attr_col_width = max(max_attr_len + 2, len("Attribute"))  # +2 for indented sub-attributes

    # Print compact table header with just counts
    print(f"{'Attribute':<{attr_col_width}} | Count")
    print("-" * (attr_col_width + 10))

    # Print priority attributes (simple, moderate, complex) - just counts
    if priority_items:
        for attr, tasks in priority_items:
            count = len(tasks)
            print(f"{attr:<{attr_col_width}} | {count:5d}")

        # Print separator after priority items
        if other_items:
            print("-" * (attr_col_width + 10))

    # Group other attributes by category (visual, relational, procedural)
    category_order = ['visual', 'relational', 'procedural']
    category_groups = {cat: [] for cat in category_order}
    uncategorized = []

    for attr, tasks in other_items:
        category = BENCHMARK_TASK_CATEGORIES.get(attr)
        if category in category_groups:
            category_groups[category].append((attr, tasks))
        else:
            uncategorized.append((attr, tasks))

    for category in category_order:
        items = category_groups[category]
        if not items:
            continue
        # Collect unique tasks across all attributes in this category
        seen = set()
        category_total = 0
        for _, tasks in items:
            for t in tasks:
                if t['task_name'] not in seen:
                    seen.add(t['task_name'])
                    category_total += 1
        print(f"{category:<{attr_col_width}} | {category_total:5d}")
        for attr, tasks in items:
            print(f"  {attr:<{attr_col_width - 2}} | {len(tasks):5d}")

    # Print any uncategorized attributes
    if uncategorized:
        print("-" * (attr_col_width + 10))
        for attr, tasks in uncategorized:
            print(f"{attr:<{attr_col_width}} | {len(tasks):5d}")

    print("-" * (attr_col_width + 10))

    # Print detailed breakdown by attribute only if verbose
    if verbose:
        print()
        print("=" * 100)
        print("DETAILED BREAKDOWN BY ATTRIBUTE")
        print("=" * 100)

        # Print priority attributes with task lists
        if priority_items:
            for attr, tasks in priority_items:
                count = len(tasks)
                print(f"{attr.upper()} ({count} tasks):")
                sorted_tasks = sorted(tasks, key=lambda x: x['task_name'])
                for i, task_info in enumerate(sorted_tasks, 1):
                    print(f"  {i:2d}. {task_info['task_name']} ({task_info['filename']})")

        # Print other attributes with task lists
        for attr, tasks in other_items:
            count = len(tasks)
            print(f"{attr.upper()} ({count} tasks):")
            sorted_tasks = sorted(tasks, key=lambda x: x['task_name'])
            for i, task_info in enumerate(sorted_tasks, 1):
                print(f"  {i:2d}. {task_info['task_name']} ({task_info['filename']})")

        # Print untagged tasks if any
        if untagged_tasks:
            count = len(untagged_tasks)
            print(f"UNTAGGED ({count} tasks):")
            sorted_tasks = sorted(untagged_tasks, key=lambda x: x['task_name'])
            for i, task_info in enumerate(sorted_tasks, 1):
                print(f"  {i:2d}. {task_info['task_name']} ({task_info['filename']})")


########################################################
# Attribute category breakdown
########################################################

def analyze_attribute_categories(tasks_data: List[Dict[str, Any]],
                                 category_remap: Dict[str, str] = None) -> Dict[str, Counter]:
    """
    Break down attributes into higher-level categories
    (e.g., difficulty, visual, relational, procedural).

    Args:
        tasks_data: List of task metadata dicts
        category_remap: Mapping from attribute name to category name.
                        Defaults to BENCHMARK_TASK_CATEGORIES.

    Returns:
        Dict mapping category names to Counter of attribute counts
    """
    if category_remap is None:
        category_remap = BENCHMARK_TASK_CATEGORIES

    categories = defaultdict(Counter)
    for task in tasks_data:
        attributes = parse_comma_separated_string_into_list(task.get('attributes', ''))
        for attr in attributes:
            if attr in category_remap:
                category = category_remap[attr]
                categories[category][attr] += 1
    return dict(categories)


def print_attribute_category_summary(categories: Dict[str, Counter]):
    """Print attribute breakdown by higher-level category."""
    category_labels = {
        'difficulty': 'Difficulty',
        'visual': 'Visual',
        'relational': 'Relational',
        'procedural': 'Procedural',
    }

    print("\n" + "=" * 80)
    print("ATTRIBUTE BREAKDOWN BY CATEGORY")
    print("=" * 80)

    for category_key in ['difficulty', 'visual', 'relational', 'procedural']:
        counter = categories.get(category_key, Counter())
        if not counter:
            continue
        label = category_labels.get(category_key, category_key)
        total_in_category = sum(counter.values())
        print(f"\n  {label}:")
        for attr, count in counter.most_common():
            pct = (count / total_in_category * 100) if total_in_category > 0 else 0
            print(f"    {attr:15s}: {count:3d} ({pct:5.1f}%)")


########################################################
# Attribute reorganization
########################################################

def print_attribute_reorganization(attribute_to_tasks: Dict[str, List[Dict[str, str]]],
                                   remap: Dict[str, str],
                                   verbose: bool = False):
    """
    Print a reorganized view of attributes based on a remapping dictionary.

    Args:
        attribute_to_tasks: Dictionary mapping attributes to list of task info dicts
        remap: Dictionary mapping old attribute names to new category names
        verbose: If True, show detailed breakdown with all tasks per category
    """
    # Build new category to tasks mapping
    category_to_tasks = defaultdict(list)
    unmapped_attrs = {}

    for attr, tasks in attribute_to_tasks.items():
        if attr in remap:
            new_category = remap[attr]
            # Add tasks to the new category, avoiding duplicates
            for task_info in tasks:
                existing_task_names = [t['task_name'] for t in category_to_tasks[new_category]]
                if task_info['task_name'] not in existing_task_names:
                    category_to_tasks[new_category].append(task_info)
        else:
            # Keep unmapped attributes separate
            unmapped_attrs[attr] = tasks

    # Print header
    print("\n" + "=" * 100)
    print("REORGANIZED ATTRIBUTES")
    print(f"Mapping: {remap}")
    print("=" * 100)

    # Calculate column widths
    all_categories = list(category_to_tasks.keys()) + list(unmapped_attrs.keys())
    max_cat_len = max(len(cat) for cat in all_categories) if all_categories else 10
    cat_col_width = max(max_cat_len, len("Category"))

    # Print table header
    print(f"{'Category':<{cat_col_width}} | Count")
    print("-" * (cat_col_width + 10))

    # Print new categories first (sorted alphabetically)
    for category in sorted(category_to_tasks.keys()):
        tasks = category_to_tasks[category]
        count = len(tasks)
        print(f"{category:<{cat_col_width}} | {count:5d}")

    # Print separator if there are unmapped attributes
    if unmapped_attrs:
        print("-" * (cat_col_width + 10))
        print("(Unmapped attributes)")
        print("-" * (cat_col_width + 10))
        for attr in sorted(unmapped_attrs.keys()):
            tasks = unmapped_attrs[attr]
            count = len(tasks)
            print(f"{attr:<{cat_col_width}} | {count:5d}")

    print("-" * (cat_col_width + 10))

    # Print detailed breakdown if verbose
    if verbose:
        print()
        print("=" * 100)
        print("DETAILED BREAKDOWN BY NEW CATEGORY")
        print("=" * 100)

        for category in sorted(category_to_tasks.keys()):
            tasks = category_to_tasks[category]
            # Find which original attributes map to this category
            original_attrs = [attr for attr, cat in remap.items() if cat == category]
            print(f"\n{category.upper()} ({len(tasks)} tasks) - from: {', '.join(original_attrs)}")
            sorted_tasks = sorted(tasks, key=lambda x: x['task_name'])
            for i, task_info in enumerate(sorted_tasks, 1):
                print(f"  {i:2d}. {task_info['task_name']} ({task_info['filename']})")

        if unmapped_attrs:
            print("\n--- UNMAPPED ATTRIBUTES ---")
            for attr in sorted(unmapped_attrs.keys()):
                tasks = unmapped_attrs[attr]
                print(f"\n{attr.upper()} ({len(tasks)} tasks):")
                sorted_tasks = sorted(tasks, key=lambda x: x['task_name'])
                for i, task_info in enumerate(sorted_tasks, 1):
                    print(f"  {i:2d}. {task_info['task_name']} ({task_info['filename']})")


########################################################
# Instruction variant analysis
########################################################

def analyze_instruction_variants(tasks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze instruction variant coverage across tasks.

    Returns:
        Dict with keys:
          - 'tasks_with_variants': list of task metadata dicts that have instruction_variants
          - 'tasks_single': list of task metadata dicts with a single instruction string
          - 'variant_key_to_tasks': dict mapping each variant key to the list of task dicts that define it
    """
    tasks_with_variants = []
    tasks_single = []
    variant_key_to_tasks: Dict[str, List[Dict[str, Any]]] = {}

    for task in tasks_data:
        variants = task.get('instruction_variants')
        if variants and isinstance(variants, dict):
            tasks_with_variants.append(task)
            for key in variants:
                variant_key_to_tasks.setdefault(key, []).append(task)
        else:
            tasks_single.append(task)

    return {
        'tasks_with_variants': tasks_with_variants,
        'tasks_single': tasks_single,
        'variant_key_to_tasks': variant_key_to_tasks,
    }


def print_instruction_variant_summary(tasks_data: List[Dict[str, Any]], verbose: bool = False):
    """Print a summary of instruction variant coverage.

    Shows how many tasks define multiple instruction variants, which variant
    keys exist, and lists the tasks with their variant texts.
    """
    stats = analyze_instruction_variants(tasks_data)
    tasks_with = stats['tasks_with_variants']
    tasks_single = stats['tasks_single']
    key_to_tasks = stats['variant_key_to_tasks']

    total = len(tasks_data)
    n_with = len(tasks_with)
    n_single = len(tasks_single)

    print("\n" + "=" * 100)
    print("INSTRUCTION VARIANT SUMMARY")
    print(f"Total tasks: {total}")
    print(f"  Tasks with instruction variants: {n_with}")
    print(f"  Tasks with single instruction:   {n_single}")
    print("=" * 100)

    if not key_to_tasks:
        print("\nNo tasks define instruction variants.")
        return

    # Table of variant keys
    sorted_keys = sorted(key_to_tasks.keys())
    max_key_len = max(len(k) for k in sorted_keys)
    key_col_w = max(max_key_len, len("Variant Key"))

    print(f"\n{'Variant Key':<{key_col_w}} | Count")
    print("-" * (key_col_w + 10))
    for key in sorted_keys:
        print(f"{key:<{key_col_w}} | {len(key_to_tasks[key]):5d}")
    print("-" * (key_col_w + 10))

    # Detailed listing of tasks with variants
    print("\n" + "=" * 100)
    print("TASKS WITH INSTRUCTION VARIANTS")
    print("=" * 100)

    sorted_tasks = sorted(tasks_with, key=lambda t: t.get('task_name', ''))
    for i, task in enumerate(sorted_tasks, 1):
        task_name = task.get('task_name', 'Unknown')
        variants = task.get('instruction_variants', {})
        keys_str = ", ".join(sorted(variants.keys()))
        print(f"\n  {i:2d}. {task_name} [{keys_str}]")
        for key in sorted(variants.keys()):
            print(f"      {key}: \"{variants[key]}\"")

    if verbose and tasks_single:
        print("\n" + "=" * 100)
        print(f"TASKS WITH SINGLE INSTRUCTION ({n_single})")
        print("=" * 100)
        sorted_single = sorted(tasks_single, key=lambda t: t.get('task_name', ''))
        for i, task in enumerate(sorted_single, 1):
            task_name = task.get('task_name', 'Unknown')
            instruction = task.get('instruction', '')
            print(f"  {i:2d}. {task_name}: \"{instruction}\"")


########################################################
# Scene analysis
########################################################

def print_task_summary_by_scene(tasks_data: List[Dict[str, Any]], verbose: bool = False):
    """
    Print a table showing the number of tasks per scene and list of tasks for each scene.

    Args:
        tasks_data: List of task metadata dicts (already loaded and filtered)
        verbose: If True, also show additional details (subfolder, attributes) for each task
    """
    scene_to_tasks = sort_tasks_by_scene(tasks_data)

    total_scenes = len(scene_to_tasks)
    total_tasks = len(tasks_data)

    # Print header
    print("\n" + "=" * 100)
    print("TASKS BY SCENE SUMMARY")
    print(f"Total scenes: {total_scenes}")
    print(f"Total tasks: {total_tasks}")
    print("=" * 100)

    # Sort scenes alphabetically
    sorted_scenes = sorted(scene_to_tasks.items(), key=lambda x: x[0])

    # Calculate column width for scene names
    max_scene_len = max(len(scene) for scene in scene_to_tasks.keys()) if scene_to_tasks else 10
    scene_col_width = max(max_scene_len, len("Scene"))

    # Print table header
    print(f"{'Scene':<{scene_col_width}} | Count")
    print("-" * (scene_col_width + 10))

    # Print each scene with task count
    for scene, tasks in sorted_scenes:
        count = len(tasks)
        scene_name = scene if scene else "(no scene)"
        print(f"{scene_name:<{scene_col_width}} | {count:5d}")

    print("-" * (scene_col_width + 10))

    # Always print detailed breakdown by scene with task lists
    print()
    print("=" * 100)
    print("TASKS BY SCENE")
    print("=" * 100)

    for scene, tasks in sorted_scenes:
        scene_name = scene if scene else "(no scene)"
        count = len(tasks)
        print(f"\n{scene_name} ({count} tasks):")

        # Sort tasks by task name
        sorted_tasks = sorted(tasks, key=lambda x: x.get('task_name', ''))
        for i, task in enumerate(sorted_tasks, 1):
            task_name = task.get('task_name', 'Unknown')
            language_instruction = task.get('instruction', '')
            variants = task.get('instruction_variants')
            variant_info = f" [variants: {', '.join(variants.keys())}]" if variants else ""

            # In verbose mode, also show additional task details
            if verbose:
                attributes = task.get('attributes', '')
                colored_attributes = colorize_attributes(attributes)
                extra_info = f" ({colored_attributes})" if attributes else ""
                print(f"  {i:2d}. {task_name}: \"{language_instruction}\"{variant_info} {extra_info}")
            else:
                print(f"  {i:2d}. {task_name}: \"{language_instruction}\"{variant_info}")


########################################################
# Object analysis
########################################################

def analyze_objects(tasks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze objects across all tasks.

    Args:
        tasks_data: List of task metadata dicts (already loaded and filtered)

    Returns:
        Dict with keys: total_instances, unique_objects, normalized_unique,
        object_frequency, normalized_frequency, objects_per_task, repeat_objects
    """
    all_objects = []
    unique_objects = set()
    normalized_unique = set()
    object_frequency = Counter()
    normalized_frequency = Counter()
    objects_per_task = []

    for task in tasks_data:
        objects = parse_comma_separated_string_into_list(task.get('contact_objects', ''))
        # Filter out 'table' as it's a fixture, not a manipulable object
        objects = [obj for obj in objects if obj.lower() != 'table']

        all_objects.extend(objects)
        unique_objects.update(objects)
        objects_per_task.append(len(objects))

        for obj in objects:
            object_frequency[obj] += 1
            normalized = normalize_object_name(obj)
            normalized_unique.add(normalized)
            normalized_frequency[normalized] += 1

    repeat_objects = sorted(
        [(obj, count) for obj, count in object_frequency.items() if count > 1],
        key=lambda x: -x[1]
    )

    return {
        'total_instances': len(all_objects),
        'unique_objects': unique_objects,
        'normalized_unique': normalized_unique,
        'object_frequency': object_frequency,
        'normalized_frequency': normalized_frequency,
        'objects_per_task': objects_per_task,
        'repeat_objects': repeat_objects,
    }


def print_object_summary(object_stats: Dict[str, Any]):
    """Print formatted object analysis report."""
    print("\n" + "=" * 80)
    print("OBJECT STATISTICS")
    print("=" * 80)

    objects_per_task = object_stats['objects_per_task']
    print(f"Unique objects (exact): {len(object_stats['unique_objects'])}")
    print(f"Unique objects (normalized, excl. duplicates like mug_01/mug_02): {len(object_stats['normalized_unique'])}")
    print(f"Total object instances across all tasks: {object_stats['total_instances']}")

    if objects_per_task:
        avg = sum(objects_per_task) / len(objects_per_task)
        print(f"Average objects per task: {avg:.1f}")
        print(f"Min objects in a task: {min(objects_per_task)}")
        print(f"Max objects in a task: {max(objects_per_task)}")

    print("\n  Top 20 Most Used Objects (normalized names):")
    for obj, count in object_stats['normalized_frequency'].most_common(20):
        print(f"    {obj:25s}: {count:3d} tasks")

    if object_stats['repeat_objects']:
        print("\n  Objects Used in Multiple Tasks (top 30):")
        for obj, count in object_stats['repeat_objects'][:30]:
            print(f"    {obj:30s}: appears in {count:3d} tasks")


########################################################
# Subtask complexity analysis
########################################################

def analyze_subtask_complexity(tasks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze subtask complexity across all tasks.

    Args:
        tasks_data: List of task metadata dicts (already loaded and filtered)

    Returns:
        Dict with keys: stage_counts, atomic_condition_counts,
        avg_stages, avg_atomic_conditions
    """
    stage_counts = Counter()
    atomic_condition_counts = []

    for task in tasks_data:
        num_stages = task.get('num_sequential_stages', 0)
        atomic_count = task.get('num_atomic_conditions', 0)
        stage_counts[num_stages] += 1
        atomic_condition_counts.append(atomic_count)

    avg_stages = 0.0
    if tasks_data:
        total_stages = sum(k * v for k, v in stage_counts.items())
        avg_stages = total_stages / len(tasks_data)

    avg_atomic = 0.0
    if atomic_condition_counts:
        avg_atomic = sum(atomic_condition_counts) / len(atomic_condition_counts)

    return {
        'stage_counts': stage_counts,
        'atomic_condition_counts': atomic_condition_counts,
        'avg_stages': avg_stages,
        'avg_atomic_conditions': avg_atomic,
    }


def print_subtask_summary(subtask_stats: Dict[str, Any]):
    """Print formatted subtask complexity report."""
    print("\n" + "=" * 80)
    print("SUBTASK COMPLEXITY")
    print("=" * 80)

    print(f"Average sequential stages per task: {subtask_stats['avg_stages']:.2f}")
    print(f"Average atomic conditions per task: {subtask_stats['avg_atomic_conditions']:.2f}")

    print("\nSequential stage count distribution:")
    for count, num_tasks in sorted(subtask_stats['stage_counts'].items()):
        print(f"    {count} stage(s): {num_tasks} tasks")


########################################################
# Subtask count analysis
########################################################

def analyze_subtask_counts(tasks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze subtask counts across all tasks.

    Subtask count represents the number of distinct manipulation actions
    (e.g., pick-and-place) required, accounting for logical mode:
        - "all": every object group must complete
        - "any": only one object group must complete
        - "choose": exactly K object groups must complete

    Args:
        tasks_data: List of task metadata dicts (already loaded and filtered)

    Returns:
        Dict with keys: distribution (Counter), per_task (list of dicts),
        avg, min_val, max_val
    """
    distribution = Counter()
    per_task = []

    for task in tasks_data:
        num_subtasks = task.get('num_subtasks', 0)
        distribution[num_subtasks] += 1
        per_task.append({
            'task_name': task.get('task_name', 'Unknown'),
            'num_subtasks': num_subtasks,
            'filename': task.get('filename', ''),
        })

    values = [t['num_subtasks'] for t in per_task]
    avg = sum(values) / len(values) if values else 0.0

    return {
        'distribution': distribution,
        'per_task': per_task,
        'avg': avg,
        'min_val': min(values) if values else 0,
        'max_val': max(values) if values else 0,
    }


def print_subtask_count_summary(subtask_count_stats: Dict[str, Any], verbose: bool = False):
    """Print formatted subtask count report."""
    print("\n" + "=" * 80)
    print("SUBTASK COUNTS")
    print("=" * 80)
    print("num_subtasks = number of distinct manipulation actions required,")
    print("accounting for logical mode (all/any/choose).")

    print(f"\nAverage subtasks per task: {subtask_count_stats['avg']:.2f}")
    print(f"Min: {subtask_count_stats['min_val']}")
    print(f"Max: {subtask_count_stats['max_val']}")

    print("\nDistribution:")
    for count, num_tasks in sorted(subtask_count_stats['distribution'].items()):
        print(f"    {count} subtask(s): {num_tasks:3d} tasks")

    if verbose:
        print("\nTasks with most subtasks:")
        sorted_tasks = sorted(subtask_count_stats['per_task'],
                              key=lambda x: -x['num_subtasks'])
        for i, t in enumerate(sorted_tasks[:20], 1):
            print(f"  {i:2d}. {t['task_name']:<45s} {t['num_subtasks']} subtask(s)  ({t['filename']})")


########################################################
# Difficulty score analysis
########################################################

def analyze_difficulty_scores(tasks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze difficulty scores and labels across all tasks.

    Args:
        tasks_data: List of task metadata dicts (already loaded and filtered)

    Returns:
        Dict with keys: label_distribution, score_distribution, per_task,
        avg_score, min_score, max_score
    """
    label_distribution = Counter()
    score_distribution = Counter()
    per_task = []

    for task in tasks_data:
        score = task.get('difficulty_score', 0)
        label = task.get('difficulty_label', 'simple')
        num_subtasks = task.get('num_subtasks', 0)
        attributes = task.get('attributes', '')

        label_distribution[label] += 1
        score_distribution[score] += 1
        per_task.append({
            'task_name': task.get('task_name', 'Unknown'),
            'filename': task.get('filename', ''),
            'difficulty_score': score,
            'difficulty_label': label,
            'num_subtasks': num_subtasks,
            'attributes': attributes,
        })

    scores = [t['difficulty_score'] for t in per_task]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        'label_distribution': label_distribution,
        'score_distribution': score_distribution,
        'per_task': per_task,
        'avg_score': avg_score,
        'min_score': min(scores) if scores else 0,
        'max_score': max(scores) if scores else 0,
    }


def print_difficulty_score_summary(difficulty_stats: Dict[str, Any], verbose: bool = False):
    """Print formatted difficulty score analysis report."""
    print("\n" + "=" * 80)
    print("DIFFICULTY SCORING")
    print("=" * 80)
    print("score = num_subtasks + max(skill_weight)")
    print(f"Thresholds: simple <= {DIFFICULTY_THRESHOLDS[0]}, "
          f"moderate <= {DIFFICULTY_THRESHOLDS[1]}, "
          f"complex > {DIFFICULTY_THRESHOLDS[1]}")
    print(f"Skill weights: {dict(sorted(SKILL_WEIGHTS.items(), key=lambda x: x[1]))}")

    print(f"\nAverage difficulty score: {difficulty_stats['avg_score']:.2f}")
    print(f"Score range: {difficulty_stats['min_score']} – {difficulty_stats['max_score']}")

    print("\nDistribution by label:")
    total = sum(difficulty_stats['label_distribution'].values())
    for label in ['simple', 'moderate', 'complex']:
        count = difficulty_stats['label_distribution'].get(label, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"    {label:10s}: {count:3d} tasks ({pct:5.1f}%)")

    print("\nScore histogram:")
    for score, count in sorted(difficulty_stats['score_distribution'].items()):
        bar = "#" * count
        print(f"    score {score:2d}: {count:3d} {bar}")

    if verbose:
        print("\nAll tasks by difficulty (highest first):")
        sorted_tasks = sorted(difficulty_stats['per_task'],
                              key=lambda x: (-x['difficulty_score'], x['task_name']))
        for i, t in enumerate(sorted_tasks, 1):
            attrs = t['attributes']
            print(f"  {i:3d}. {t['task_name']:<45s} "
                  f"score={t['difficulty_score']:2d}  "
                  f"num_subtasks={t['num_subtasks']}  "
                  f"[{t['difficulty_label']}]  "
                  f"({attrs})")


########################################################
# Episode length analysis
########################################################

def analyze_episode_lengths(tasks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze episode lengths across all tasks.

    Args:
        tasks_data: List of task metadata dicts (already loaded and filtered)

    Returns:
        Dict with keys: lengths, by_difficulty
    """
    lengths = []
    by_difficulty = defaultdict(list)

    for task in tasks_data:
        try:
            ep_length = int(task.get('episode_s', 0))
            if ep_length > 0:
                lengths.append(ep_length)
                diff_label = task.get('difficulty_label', '')
                if diff_label in ('simple', 'moderate', 'complex'):
                    by_difficulty[diff_label].append(ep_length)
        except (ValueError, TypeError):
            pass

    return {
        'lengths': lengths,
        'by_difficulty': dict(by_difficulty),
    }


def print_episode_length_summary(episode_stats: Dict[str, Any]):
    """Print formatted episode length analysis report."""
    lengths = episode_stats['lengths']
    if not lengths:
        return

    print("\n" + "=" * 80)
    print("EPISODE LENGTH ANALYSIS")
    print("=" * 80)

    print(f"Average episode length: {sum(lengths) / len(lengths):.1f} seconds")
    print(f"Min episode length: {min(lengths)} seconds")
    print(f"Max episode length: {max(lengths)} seconds")

    print("\nAverage episode length by difficulty:")
    for diff in ['simple', 'moderate', 'complex']:
        diff_lengths = episode_stats['by_difficulty'].get(diff, [])
        if diff_lengths:
            avg = sum(diff_lengths) / len(diff_lengths)
            print(f"    {diff:15s}: {avg:6.1f} seconds (n={len(diff_lengths)})")


########################################################
# Full report
########################################################

def print_full_report(tasks_data: List[Dict[str, Any]], verbose: bool = False):
    """
    Print a comprehensive report covering all analysis dimensions:
    overview, attributes, attribute categories, objects, subtasks,
    episodes, and scenes.

    Args:
        tasks_data: List of task metadata dicts (already loaded and filtered)
        verbose: If True, show detailed breakdowns in each section
    """
    print("\n" + "=" * 80)
    print("ROBOLAB TASK STATISTICS — FULL REPORT")
    print(f"Total tasks: {len(tasks_data)}")
    print("=" * 80)

    # Tasks by subfolder
    subfolder_to_tasks = sort_tasks_by_subfolder(tasks_data)
    print("\n## TASKS BY SUBFOLDER")
    print("-" * 40)
    for subfolder in sorted(subfolder_to_tasks.keys()):
        count = len(subfolder_to_tasks[subfolder])
        pct = (count / len(tasks_data)) * 100 if tasks_data else 0
        name = subfolder if subfolder else "(root)"
        print(f"  {name:25s}: {count:3d} tasks ({pct:5.1f}%)")

    # Attribute summary
    attribute_to_tasks, total_tasks, untagged_tasks = analyze_task_attributes(tasks_data)
    print_attribute_summary(attribute_to_tasks, total_tasks, untagged_tasks, verbose=verbose)

    # Attribute category breakdown
    categories = analyze_attribute_categories(tasks_data)
    print_attribute_category_summary(categories)

    # Object analysis
    object_stats = analyze_objects(tasks_data)
    print_object_summary(object_stats)

    # Subtask complexity
    subtask_stats = analyze_subtask_complexity(tasks_data)
    print_subtask_summary(subtask_stats)

    # Subtask counts
    subtask_count_stats = analyze_subtask_counts(tasks_data)
    print_subtask_count_summary(subtask_count_stats, verbose=verbose)

    # Difficulty scoring
    difficulty_stats = analyze_difficulty_scores(tasks_data)
    print_difficulty_score_summary(difficulty_stats, verbose=verbose)

    # Episode lengths
    episode_stats = analyze_episode_lengths(tasks_data)
    print_episode_length_summary(episode_stats)

    # Scene summary
    scene_to_tasks = sort_tasks_by_scene(tasks_data)
    print("\n" + "=" * 80)
    print("SCENE USAGE")
    print("=" * 80)
    print(f"Total unique scenes: {len(scene_to_tasks)}")
    print("\nMost used scenes (top 15):")
    sorted_scenes = sorted(scene_to_tasks.items(), key=lambda x: -len(x[1]))
    for scene, tasks in sorted_scenes[:15]:
        scene_name = scene if scene else "(no scene)"
        print(f"    {scene_name:45s}: {len(tasks):3d} tasks")

    print("\n" + "=" * 80)


########################################################

if __name__ == "__main__":
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Compute task statistics from _metadata/task_metadata.json file. Shows number of tasks per attribute, per scene, etc. Make sure the metadata file is generated first by running generate_task_metadata.py."
    )
    parser.add_argument(
        "--metadata-file",
        default=os.path.join(TASK_DIR, "_metadata", "task_metadata.json"),
        help="Path to task_metadata.json file"
    )
    parser.add_argument(
        "--subfolders",
        nargs="+",
        default=None,
        help="List of subfolder names to include (e.g., --subfolders tasks1 tasks2). If not specified, includes all tasks. When using the default --metadata-file, defaults to DEFAULT_TASK_SUBFOLDERS."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show full comprehensive report (attributes, categories, objects, subtasks, episodes, scenes)"
    )
    parser.add_argument(
        "--by-scene",
        action="store_true",
        default=False,
        help="Group and display tasks by scene instead of by attributes"
    )
    parser.add_argument(
        "--objects",
        action="store_true",
        default=False,
        help="Show object analysis (unique objects, frequency, per-task stats)"
    )
    parser.add_argument(
        "--subtasks",
        action="store_true",
        default=False,
        help="Show subtask complexity analysis"
    )
    parser.add_argument(
        "--episodes",
        action="store_true",
        default=False,
        help="Show episode length analysis"
    )
    parser.add_argument(
        "--subtask-counts",
        action="store_true",
        default=False,
        help="Show subtask count analysis (manipulation actions per task)"
    )
    parser.add_argument(
        "--difficulty",
        action="store_true",
        default=False,
        help="Show difficulty score analysis (score distribution, labels, histogram)"
    )
    parser.add_argument(
        "--by-instruction-type",
        action="store_true",
        default=False,
        help="Show instruction variant analysis (which tasks define multiple instruction types)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save output to tasks/_metadata/task_report.txt in addition to printing to console"
    )

    args = parser.parse_args()

    # Check if metadata file exists
    if not os.path.exists(args.metadata_file):
        print(f"Error: Metadata file not found: {args.metadata_file}")
        print("Please run generate_task_metadata.py first to generate the metadata file.")
        exit(1)

    # Load data once
    tasks_data = load_metadata(args.metadata_file)
    default_metadata = os.path.join(TASK_DIR, "_metadata", "task_metadata.json")
    subfolders = args.subfolders
    if subfolders is None and os.path.exists(default_metadata) and os.path.samefile(args.metadata_file, default_metadata):
        subfolders = DEFAULT_TASK_SUBFOLDERS
    tasks_data = filter_by_subfolders(tasks_data, subfolders)

    # If --save, tee stdout to both console and file
    if args.save:
        report_path = os.path.join(os.path.dirname(args.metadata_file), "task_report.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report_file = open(report_path, 'w')

        class Tee:
            """Write to both console and file simultaneously."""
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()
            def flush(self):
                for s in self.streams:
                    s.flush()

        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, report_file)

    # Determine which reports to show
    individual_flags = args.by_scene or args.objects or args.subtasks or args.episodes or args.subtask_counts or args.difficulty or args.by_instruction_type

    if individual_flags:
        # Show only the requested sections
        if args.by_scene:
            print_task_summary_by_scene(tasks_data, args.verbose)
        if args.objects:
            object_stats = analyze_objects(tasks_data)
            print_object_summary(object_stats)
        if args.subtasks:
            subtask_stats = analyze_subtask_complexity(tasks_data)
            print_subtask_summary(subtask_stats)
        if args.subtask_counts:
            subtask_count_stats = analyze_subtask_counts(tasks_data)
            print_subtask_count_summary(subtask_count_stats, verbose=args.verbose)
        if args.difficulty:
            difficulty_stats = analyze_difficulty_scores(tasks_data)
            print_difficulty_score_summary(difficulty_stats, verbose=args.verbose)
        if args.episodes:
            episode_stats = analyze_episode_lengths(tasks_data)
            print_episode_length_summary(episode_stats)
        if args.by_instruction_type:
            print_instruction_variant_summary(tasks_data, verbose=args.verbose)
    elif args.verbose:
        # Full comprehensive report
        print_full_report(tasks_data, verbose=True)
    else:
        # Default: attribute summary + reorganization
        attribute_to_tasks, total_tasks, untagged_tasks = analyze_task_attributes(tasks_data)
        print_attribute_summary(attribute_to_tasks, total_tasks, untagged_tasks)
        print_attribute_reorganization(attribute_to_tasks, BENCHMARK_TASK_CATEGORIES)

    # Clean up file output
    if args.save:
        sys.stdout = original_stdout
        report_file.close()
        # Strip ANSI color codes from the saved file
        with open(report_path, 'r') as f:
            content = f.read()
        content = re.sub(r'\033\[[0-9;]*m', '', content)
        with open(report_path, 'w') as f:
            f.write(content)
        print(f"\nReport saved to: {report_path}")
