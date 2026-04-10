#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Generate README.md with object table from object_catalog.json.

This script reads the object catalog and generates a markdown table with
object previews, descriptions, and physics properties.

Usage:
    python generate_readme.py                           # All datasets
    python generate_readme.py --datasets hope ycb       # Specific datasets
"""

import argparse
import os
import sys
from typing import List

from common import load_catalog

from robolab.constants import OBJECT_DIR


def generate_readme(datasets: List[str], objects_dir: str, output_path: str) -> str:
    """
    Generate a markdown table for object datasets.

    Args:
        datasets: List of dataset names to include (e.g., ['hope', 'ycb'])
        objects_dir: Path to the objects directory
        output_path: Path to save the markdown file

    Returns:
        Status message
    """
    from robolab.core.utils.csv_utils import get_markdown_image_text, save_markdown_table

    catalog_path = os.path.join(objects_dir, "object_catalog.json")

    if not os.path.exists(catalog_path):
        print(f"Error: Object catalog not found at {catalog_path}")
        print("Run generate_catalog.py first.")
        return ""

    catalog_data = load_catalog(catalog_path)
    print(f"Loaded {len(catalog_data)} objects from catalog")

    # Prepare table data
    table_data = []
    headers = [
        "USD Name", "Dataset", "Image Preview", "Description",
        "mass (kg)", "density (kg/m^3)", "dyn. friction", "stat. friction", "restitution"
    ]

    for obj in catalog_data:
        dataset = obj.get('dataset', None)

        # Skip if not in requested datasets
        if datasets and dataset not in datasets:
            continue

        name = obj.get('name', '')
        usd_path = obj.get('usd_path', None)
        description = obj.get('description', '').strip()
        mass = obj.get('mass', None)
        density = obj.get('density', None)
        dynamic_friction = obj.get('dynamic_friction', None)
        static_friction = obj.get('static_friction', None)
        restitution = obj.get('restitution', None)

        # Format numeric values
        mass_str = f"{mass:.2f}" if mass is not None else "N/A"
        density_str = f"{density:.2f}" if density is not None and density > 0 else "N/A"
        dyn_friction_str = f"{dynamic_friction:.1f}" if dynamic_friction is not None else "N/A"
        stat_friction_str = f"{static_friction:.1f}" if static_friction is not None else "N/A"
        restitution_str = f"{restitution:.1f}" if restitution is not None else "N/A"

        # Create image preview
        image_dir = os.path.join(objects_dir, "_images", dataset)
        usd_filename = os.path.basename(usd_path) if usd_path else ""

        image_preview = get_markdown_image_text(
            filename_to_img=usd_filename,
            relative_dir=objects_dir,
            image_dir=image_dir,
            image_ext='.png',
            size=(120, 120)
        )

        table_data.append([
            name, dataset, image_preview, description,
            mass_str, density_str, dyn_friction_str, stat_friction_str, restitution_str
        ])

    # Sort by dataset, then by name
    table_data.sort(key=lambda x: (x[1], x[0]))

    print(f"Generated table with {len(table_data)} objects")

    full_table_data = [headers] + table_data
    dataset_names = ", ".join(datasets)

    save_markdown_table(
        csv_input=full_table_data,
        output_path=output_path,
        title="Object Datasets",
        description=f"Available USD objects: {len(table_data)} objects across {len(datasets)} datasets. Datasets: {dataset_names}",
        align="left"
    )

    return f"Markdown table saved to {output_path}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate README.md with object table from catalog"
    )
    parser.add_argument(
        "--datasets",
        nargs='+',
        default=['hope', 'handal', 'hot3d', 'ycb', 'vomp', 'fruits_veggies'],
        help="List of dataset names to include"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(OBJECT_DIR, "README.md"),
        help="Output path for README.md"
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Generating README for datasets: {', '.join(args.datasets)}")
    result = generate_readme(
        datasets=args.datasets,
        objects_dir=OBJECT_DIR,
        output_path=args.output
    )
    print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
