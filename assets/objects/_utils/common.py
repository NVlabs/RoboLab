# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Shared utilities for object asset management scripts.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from robolab.constants import OBJECT_CATALOG_PATH, OBJECT_DIR
from robolab.core.utils.file_utils import find_usd_files


def iter_object_files(root: Path = Path(OBJECT_DIR)) -> List[Path]:
    """
    Return all USD files under the root directory.
    
    Excludes:
    - Subfolders starting with '_'
    - 'materials' directories
    
    Args:
        root: Root directory to search (defaults to OBJECT_DIR)
        
    Returns:
        List of paths to USD files
    """
    return find_usd_files(root, recursive=True, exclude_underscore_dirs=True, exclude_materials=True)


def load_catalog(catalog_path: Path = Path(OBJECT_CATALOG_PATH)) -> List[Dict[str, Any]]:
    """
    Load the object catalog JSON file.
    
    Args:
        catalog_path: Path to object_catalog.json
        
    Returns:
        List of object dictionaries
    """
    with open(catalog_path, 'r') as f:
        return json.load(f)


def get_dataset_from_path(usd_path: str) -> str:
    """
    Extract the dataset name from a USD file path.
    
    Given a path like '.../objects/vomp/object.usd', returns 'vomp'.
    
    Args:
        usd_path: Path to USD file
        
    Returns:
        Dataset name string
    """
    parts = str(usd_path).replace("\\", "/").split("/")
    if "objects" in parts:
        idx = parts.index("objects")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def print_object_info(object_info: Dict[str, Any], usd_path: Path) -> None:
    """Print object info dictionary in a readable format."""
    print(f"Object: {usd_path.name}, usd_path: {usd_path}")
    for key, value in object_info.items():
        if isinstance(value, (list, tuple)) and len(value) > 0:
            if isinstance(value[0], (int, float)):
                formatted_value = f"[{', '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in value)}]"
            else:
                formatted_value = str(value)
        else:
            formatted_value = str(value)
        print(f"  {key:20s}: {formatted_value}")


def resolve_usd_path(relative_path: str) -> str:
    """
    Resolve a relative usd_path from the catalog to an absolute path.
    
    The catalog stores paths relative to PACKAGE_DIR (e.g., 'assets/objects/ycb/banana.usd').
    This function converts them to absolute paths.
    
    Args:
        relative_path: Path relative to PACKAGE_DIR
        
    Returns:
        Absolute path string
    """
    from robolab.constants import resolve_catalog_path
    return resolve_catalog_path(relative_path)
