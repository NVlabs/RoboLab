# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
from datetime import datetime

# Get the robolab package root directory (repo root)
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # robolab (repo root)

# Get source directory (the robolab package itself)
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__)) # robolab/robolab

# Get children of package directory
DEFAULT_OUTPUT_DIR = os.path.join(PACKAGE_DIR, "output")
ASSET_DIR = os.path.join(PACKAGE_DIR, "assets")
BACKGROUND_ASSET_DIR = os.path.join(ASSET_DIR, "backgrounds")
OBJECT_DIR = os.path.join(ASSET_DIR, "objects")
FIXTURE_DIR = os.path.join(ASSET_DIR, "fixtures")
SCENE_DIR = os.path.join(ASSET_DIR, "scenes")
ROBOTS_DIR = os.path.join(ASSET_DIR, "robots")

# Get children of source directory
TASK_DIR = os.path.join(SOURCE_DIR, "tasks")
DEFAULT_TASK_SUBFOLDERS = [
    'benchmark',
]


# Object catalog
OBJECT_CATALOG_PATH = os.path.join(OBJECT_DIR, "object_catalog.json")


def resolve_catalog_path(relative_path: str) -> str:
    """
    Resolve a relative path from object_catalog.json to an absolute path.

    The catalog stores paths relative to PACKAGE_DIR (e.g., 'assets/objects/ycb/banana.usd').
    This function converts them to absolute paths.

    Args:
        relative_path: Path relative to PACKAGE_DIR

    Returns:
        Absolute path string
    """
    # If already absolute, return as-is
    if os.path.isabs(relative_path):
        return relative_path

    return os.path.join(PACKAGE_DIR, relative_path)

# Output directory
_output_dir = None

def set_output_dir(path: str):
    """Set the global output directory."""
    global _output_dir
    _output_dir = path
    if _output_dir is not None:
        os.makedirs(_output_dir, exist_ok=True)

def clear_output_dir():
    set_output_dir(None)

def get_output_dir() -> str:
    """Get the global output directory. Returns DEFAULT_OUTPUT_DIR if not set."""
    if _output_dir is None:
        set_output_dir(DEFAULT_OUTPUT_DIR)
    return _output_dir

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


DEBUG = False
VERBOSE = False
VISUALIZE = False
ENABLE_SUBTASK_PROGRESS_CHECKING = False
RECORD_IMAGE_DATA = False
DEVICE = "cuda:0"

# Difficulty scoring constants (authoritative source for compute_difficulty_score in subtask_utils.py)
SKILL_WEIGHTS: dict[str, int] = {
    'color': 0, 'semantics': 0, 'size': 0, 'conjunction': 0, 'vague': 0,
    'spatial': 1,
    'counting': 2, 'sorting': 2, 'stacking': 2, 'affordance': 2,
    'reorientation': 3,
}
DIFFICULTY_THRESHOLDS = (2, 4)  # simple <= 2, moderate <= 4, complex > 4

# Task category remap: maps fine-grained attributes to higher-level categories
BENCHMARK_TASK_CATEGORIES = {
    'size': 'visual',
    'color': 'visual',
    'semantics': 'visual',
    'spatial': 'relational',
    'conjunction': 'relational',
    'counting': 'relational',
    'stacking': 'procedural',
    'sorting': 'procedural',
    'reorientation': 'procedural',
    'affordance': 'procedural',
}
