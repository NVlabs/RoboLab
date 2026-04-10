# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Generate Scene README from Existing Metadata

This module generates a markdown README file from existing scene metadata CSV files.
Unlike generate_scene_metadata.py, this script does not analyze USD files or generate
metadata from scratch. Instead, it takes existing CSV data and creates markdown
documentation with scene preview images.

Use this script when:
- You've already generated scene metadata (scene_table.csv exists)
- You want to regenerate the README without re-analyzing scenes
- You've updated scene images and want to refresh the documentation

Usage:
    python generate_scene_readme.py --scene-folder /path/to/scenes

Dependencies:
    - robolab.core.utils.csv_utils
"""

import argparse
import os

if __name__ == "__main__":
    """
    Main entry point for generating markdown README from existing metadata.

    This script reads existing scene metadata CSV and generates a markdown table
    with scene preview images. It does NOT:
    - Launch Isaac Lab or simulation environment
    - Analyze USD files
    - Generate new metadata

    Command-line Arguments:
        --scene-folder: Directory containing scenes (default: SCENE_DIR)
        --csv-file: Path to existing CSV file (default: SCENE_DIR/_metadata/scene_table.csv)
        --output: Path for output README.md (default: SCENE_DIR/README.md)

    Workflow:
        1. Read existing CSV metadata file
        2. Add image references to CSV data
        3. Generate markdown table with scene previews
    """
    from robolab.constants import SCENE_DIR
    from robolab.core.utils.csv_utils import add_images_to_csv, save_markdown_table

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate scene README from existing metadata CSV"
    )
    parser.add_argument(
        "--scene-folder",
        default=SCENE_DIR,
        help="Path to the scenes directory"
    )
    parser.add_argument(
        "--csv-file",
        default=None,
        help="Path to the CSV metadata file (default: SCENE_DIR/_metadata/scene_table.csv)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for output README.md (default: SCENE_DIR/README.md)"
    )
    args = parser.parse_args()

    # Set defaults
    csv_file_path = args.csv_file or os.path.join(SCENE_DIR, '_metadata', 'scene_table.csv')
    output_path = args.output or os.path.join(SCENE_DIR, 'README.md')
    image_dir = os.path.join(SCENE_DIR, '_images')

    # Validate inputs
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Please run generate_scene_metadata.py first to create the metadata file.")
        exit(1)

    if not os.path.exists(image_dir):
        print(f"Warning: Image directory not found at {image_dir}")
        print("Images will not be included in the README.")

    print(f"Reading CSV from: {csv_file_path}")
    print(f"Using images from: {image_dir}")
    print(f"Generating README at: {output_path}")

    # Generate Markdown table with images
    csv_data_with_images = add_images_to_csv(
        csv_file_path=csv_file_path,
        image_dir=image_dir,
        column_name_to_img='scene',
        image_column_name='Preview',
        relative_dir=args.scene_folder
    )

    save_markdown_table(
        csv_input=csv_data_with_images,
        output_path=output_path,
        title='Available Scenes',
        description='A list of USD scenes available in the scene directory.'
    )

    print(f"\n✓ Successfully generated README at: {output_path}")
