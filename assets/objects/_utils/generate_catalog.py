#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Generate object_catalog.json from USD files.

This script scans the objects directory for USD files and extracts metadata
(dimensions, physics properties, descriptions, etc.) into a JSON catalog.

Usage:
    python generate_catalog.py                          # Regenerate full catalog
    python generate_catalog.py --objects path/to/dir    # Scan specific directory
    python generate_catalog.py --list-classes           # List all semantic labels
    python generate_catalog.py --list-classes --by-dataset  # Labels by dataset
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common import get_dataset_from_path, iter_object_files, load_catalog, print_object_info

from robolab.constants import OBJECT_CATALOG_PATH, OBJECT_DIR


def generate_catalog(objects_dir: Path = Path(OBJECT_DIR), verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Generate object catalog by scanning USD files.
    
    Args:
        objects_dir: Directory containing object USD files
        verbose: Print details for each object
        
    Returns:
        List of object info dictionaries (with paths relative to PACKAGE_DIR)
    """
    from robolab.constants import PACKAGE_DIR
    from robolab.core.utils.usd_utils import get_usd_rigid_body_info
    
    catalog: List[Dict[str, Any]] = []
    
    if objects_dir.is_dir():
        usd_files = iter_object_files(objects_dir)
    else:
        usd_files = [objects_dir]
    
    print(f"Scanning {len(usd_files)} USD files...")
    
    for usd in usd_files:
        object_info = get_usd_rigid_body_info(str(usd))
        object_info['dataset'] = get_dataset_from_path(str(usd))
        
        # Convert usd_path to relative path (relative to PACKAGE_DIR)
        abs_path = object_info.get('usd_path', '')
        if abs_path and PACKAGE_DIR in abs_path:
            object_info['usd_path'] = abs_path.replace(PACKAGE_DIR + '/', '')
        
        if verbose:
            print_object_info(object_info, usd)
        
        catalog.append(object_info)
    
    return catalog


def save_catalog(catalog: List[Dict[str, Any]], output_path: Path = Path(OBJECT_CATALOG_PATH)) -> None:
    """Save catalog to JSON file."""
    output_path.write_text(json.dumps(catalog, indent=2))
    print(f"Saved catalog with {len(catalog)} objects to: {output_path}")


def list_classes(catalog: List[Dict[str, Any]], by_dataset: bool = False, verbose: bool = False) -> None:
    """
    Print all unique semantic labels (classes) from the catalog.
    
    Args:
        catalog: Object catalog list
        by_dataset: Group classes by dataset
        verbose: Show all objects under each class
    """
    def get_display_path(usd_path: str) -> str:
        """Get display path (strip 'assets/objects/' prefix if present)."""
        prefix = "assets/objects/"
        if usd_path.startswith(prefix):
            return usd_path[len(prefix):]
        return usd_path
    
    # Track objects without a class: dataset -> [(name, path)]
    no_class_objects: Dict[str, List[tuple]] = defaultdict(list)
    
    if by_dataset:
        # Group objects by dataset, then by class: dataset -> class -> [(name, path)]
        objects_by_dataset_class: Dict[str, Dict[str, List[tuple]]] = defaultdict(lambda: defaultdict(list))
        for obj in catalog:
            dataset = obj.get('dataset', 'unknown')
            obj_class = obj.get('class', '').strip()
            obj_name = obj.get('name', '')
            obj_path = get_display_path(obj.get('usd_path', ''))
            if obj_class:
                objects_by_dataset_class[dataset][obj_class].append((obj_name, obj_path))
            else:
                no_class_objects[dataset].append((obj_name, obj_path))
        
        print("\nSemantic Labels by Dataset:")
        print("=" * 60)
        for dataset in sorted(objects_by_dataset_class.keys()):
            classes = objects_by_dataset_class[dataset]
            print(f"\n{dataset} ({len(classes)} classes):")
            for cls in sorted(classes.keys()):
                obj_list = sorted(classes[cls], key=lambda x: x[0])
                if verbose:
                    print(f"  - {cls} ({len(obj_list)} objects):")
                    for name, path in obj_list:
                        print(f"      {name} ({path})")
                else:
                    print(f"  - {cls}")
    else:
        # Group objects by class: class -> [(name, path)]
        objects_by_class: Dict[str, List[tuple]] = defaultdict(list)
        for obj in catalog:
            obj_class = obj.get('class', '').strip()
            obj_name = obj.get('name', '')
            obj_path = get_display_path(obj.get('usd_path', ''))
            dataset = obj.get('dataset', 'unknown')
            if obj_class:
                objects_by_class[obj_class].append((obj_name, obj_path))
            else:
                no_class_objects[dataset].append((obj_name, obj_path))
        
        print("\nAll Semantic Labels:")
        print("=" * 60)
        for cls in sorted(objects_by_class.keys()):
            obj_list = sorted(objects_by_class[cls], key=lambda x: x[0])
            if verbose:
                print(f"\n{cls} ({len(obj_list)} objects):")
                for name, path in obj_list:
                    print(f"    {name} ({path})")
            else:
                print(f"  {cls}: {len(obj_list)} objects")
        
        print(f"\nTotal: {len(objects_by_class)} unique classes")
    
    # Show objects without a class
    total_no_class = sum(len(v) for v in no_class_objects.values())
    if total_no_class > 0:
        print(f"\n\nObjects WITHOUT a class attribute ({total_no_class} total):")
        print("=" * 60)
        for dataset in sorted(no_class_objects.keys()):
            obj_list = sorted(no_class_objects[dataset], key=lambda x: x[0])
            if verbose:
                print(f"\n{dataset} ({len(obj_list)} objects):")
                for name, path in obj_list:
                    print(f"    {name} ({path})")
            else:
                print(f"  {dataset}: {len(obj_list)} objects")


def main():
    parser = argparse.ArgumentParser(
        description="Generate object_catalog.json from USD files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--objects", 
        type=Path, 
        default=Path(OBJECT_DIR),
        help="Path to objects directory or single USD file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(OBJECT_CATALOG_PATH),
        help="Output path for catalog JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print details (objects per class with --list-classes, or object info during generation)"
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List all semantic labels instead of generating catalog"
    )
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        help="Group classes by dataset (use with --list-classes)"
    )
    
    args = parser.parse_args()
    
    if args.list_classes:
        # Load existing catalog and list classes
        if not args.output.exists():
            print(f"Error: Catalog not found at {args.output}")
            print("Run without --list-classes first to generate the catalog.")
            return 1
        
        catalog = load_catalog(args.output)
        list_classes(catalog, by_dataset=args.by_dataset, verbose=args.verbose)
    else:
        # Generate and save catalog
        catalog = generate_catalog(objects_dir=args.objects, verbose=args.verbose)
        save_catalog(catalog, output_path=args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
