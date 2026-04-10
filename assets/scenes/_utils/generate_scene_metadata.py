# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
USD Scene Metadata Generator

This module analyzes USD scene files to extract comprehensive metadata about objects contained
within each scene. It categorizes objects into rigid bodies and static bodies, providing
detailed analysis that can be used for scene validation, documentation, and task planning.

This script will automatically update the README.md file in the scenes folder with a table of the scenes and their metadata.
If you wish to update the README.md file only, use generate_scene_readme_only.py instead.

The module generates metadata in multiple formats:
- JSON format with detailed object information
- CSV format for tabular analysis and spreadsheet compatibility
- Markdown tables with scene previews (when combined with screenshot generation)

Key Features:
- Automatic detection and classification of rigid vs static bodies
- Batch processing of USD file directories
- Multiple output formats (JSON, CSV, Markdown)
- Integration with screenshot generation for visual documentation
- Support for Isaac Lab scene analysis workflows

Usage:
    python generate_scene_metadata.py --scene-folder /path/to/scenes --output-folder /path/to/output

Dependencies:
    - Isaac Lab (isaaclab.app)
    - robolab.core.utils.usd_utils
    - robolab.core.utils.file_utils
    - robolab.core.utils.csv_utils
"""

import json
import csv
import os

from typing import Dict, List, Tuple, Any, Optional

def convert_scene_results_to_csv(results: Dict[str, List[Dict]]) -> List[List[str]]:
    """
    Convert scene analysis results to CSV format for tabular representation.

    This function processes the scene metadata dictionary and creates a structured CSV
    representation that separates objects into dynamic bodies, kinematic bodies, and
    static bodies, providing comma-separated lists of object names for each scene.

    Args:
        results (Dict[str, List[Dict]]): Dictionary where keys are USD filenames and
            values are lists of object dictionaries containing scene analysis data.
            Each object dict should have 'name', 'rigid_body', and 'kinematic' fields.

    Returns:
        List[List[str]]: A list of CSV rows where:
            - First row is the header: ['scene', 'dynamic_bodies', 'kinematic_bodies', 'static_bodies']
            - Subsequent rows contain: [filename, dynamic_names, kinematic_names, static_names]

    Note:
        - Dynamic bodies: rigid_body=True and kinematic=False
        - Kinematic bodies: rigid_body=True and kinematic=True
        - Static bodies: rigid_body=False
        - Object names are joined with commas for the CSV representation
        - Empty categories are represented as empty strings
    """
    csv_rows = []

    csv_rows.append(['scene', 'dynamic_bodies', 'kinematic_bodies', 'static_bodies'])
    print(f"Converting scene results to CSV...")

    for filename, objects in results.items():
        dynamic_bodies = [obj.get('name', '') for obj in objects if obj.get('rigid_body', False) and not obj.get('kinematic', False)]
        kinematic_bodies = [obj.get('name', '') for obj in objects if obj.get('rigid_body', False) and obj.get('kinematic', False)]
        static_bodies = [obj.get('name', '') for obj in objects if not obj.get('rigid_body', False)]

        row = [
            filename,
            ', '.join(dynamic_bodies) if dynamic_bodies else '',
            ', '.join(kinematic_bodies) if kinematic_bodies else '',
            ', '.join(static_bodies) if static_bodies else '',
        ]
        csv_rows.append(row)

    return csv_rows

def generate_scene_metadata(scene_folder: str, output_folder: str, ignore_files: list[str] | None = None, ignore_folders: list[str] | None = None):
    """
    Generate comprehensive metadata for all USD scene files in a directory.

    This function analyzes all USD files in the specified folder, extracting detailed
    information about rigid bodies and static bodies contained in each scene. The results
    are saved in both JSON and CSV formats for different use cases.

    The analysis process:
    1. Discovers all USD files in the target directory
    2. Analyzes each file to identify and categorize objects
    3. Separates objects into rigid bodies vs static bodies
    4. Generates detailed reports in multiple formats
    5. Provides console output with analysis summaries

    Args:
        scene_folder (str): Path to directory containing USD scene files to analyze
        output_folder (str): Directory where metadata files will be saved
        ignore_files (list[str] | None, optional): List of file names to ignore.
            Files matching these names will be skipped. Defaults to None
        ignore_folders (list[str] | None, optional): List of folder names to ignore.
            Files in these folders will be skipped. Defaults to None

    Raises:
        ValueError: If the scene_folder does not exist

    Outputs:
        - scene_metadata.json: Detailed JSON with all object information
        - scene_table.csv: Tabular CSV format with object counts and lists
        - Console output: Progress updates and analysis summaries

    Note:
        - Only processes files with USD extensions (.usd, .usda, .usdc, .usdz)
        - Creates output directory if it doesn't exist
        - Handles errors gracefully and continues processing remaining files
        - Provides detailed console feedback during processing
    """

    from robolab.core.utils.file_utils import find_usd_files
    from robolab.core.utils.usd_utils import get_usd_objects_info

    # Validate folder path
    if not os.path.exists(scene_folder):
        raise ValueError(f"Error: Folder '{scene_folder}' does not exist.")
    print(f"Analyzing USD files in: {scene_folder}")

    usd_files = find_usd_files(scene_folder)
    # Also include scenes under llm_generated_settled subfolder (non-recursive)
    llm_settled_dir = os.path.join(scene_folder, "llm_generated_settled")
    if os.path.isdir(llm_settled_dir):
        usd_files += find_usd_files(llm_settled_dir)
    # de-duplicate and sort
    usd_files = sorted(set(usd_files))

    # Filter out ignored folders
    if ignore_folders:
        usd_files = [f for f in usd_files if not any(folder in str(f) for folder in ignore_folders)]

    # Filter out ignored files
    if ignore_files:
        usd_files = [f for f in usd_files if os.path.basename(str(f)) not in ignore_files]

    print(f"Found {len(usd_files)} USD files in {scene_folder}")

    results = {}

    for usd_file in usd_files:
        filename = os.path.basename(usd_file)
        print(f"Processing: {filename}")
        scene_analysis = get_usd_objects_info(usd_file)

        dynamic_bodies = [obj for obj in scene_analysis if obj.get('rigid_body') and not obj.get('kinematic', False)]
        kinematic_bodies = [obj for obj in scene_analysis if obj.get('rigid_body') and obj.get('kinematic', False)]
        static_bodies = [obj for obj in scene_analysis if not obj.get('rigid_body')]

        if scene_analysis:
            results[filename] = scene_analysis

            dynamic_names = [obj['name'] for obj in dynamic_bodies]
            kinematic_names = [obj['name'] for obj in kinematic_bodies]
            static_names = [obj['name'] for obj in static_bodies]

            print(f"\tDynamic bodies: {', '.join(dynamic_names) if dynamic_names else 'None'}")
            print(f"\tKinematic bodies: {', '.join(kinematic_names) if kinematic_names else 'None'}")
            print(f"\tStatic bodies: {', '.join(static_names) if static_names else 'None'}")


    if not results:
        print("No USD files found or processed successfully.")

    # Save results to JSON file
    json_output_path = os.path.join(output_folder, "scene_metadata.json")
    try:
        with open(json_output_path, 'w') as f:
            # Use default=str to coerce any non-serializable values
            json.dump(results, f, indent=2, default=str)
        print(f"\nJSON results saved to: {json_output_path}")
    except Exception as e:
        print(f"Error saving JSON results to {json_output_path}: {e}")

    # Save results to CSV file
    csv_output_path = os.path.join(output_folder, "scene_table.csv")
    try:
        csv_rows = convert_scene_results_to_csv(results)
        with open(csv_output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"CSV results saved to: {csv_output_path}")
    except Exception as e:
        print(f"Error saving CSV results to {csv_output_path}: {e}")

if __name__ == "__main__":
    """
    Main entry point for USD scene metadata generation.

    This script provides a comprehensive workflow for analyzing USD scene files:
    1. Optionally generates preview screenshots of scenes
    2. Analyzes scenes to extract object metadata (rigid/static bodies)
    3. Outputs results in JSON and CSV formats
    4. Generates markdown documentation with scene previews

    Command-line Arguments:
        --scene-folder: Directory containing USD files to analyze (default: SCENE_DIR)
        --output-folder: Directory for metadata output files (default: SCENE_DIR/_metadata)
        --generate-images: Flag to also generate screenshot images of scenes

    Workflow:
        1. Initialize Isaac Lab application in headless mode
        2. Optionally generate scene screenshots for documentation
        3. Extract and analyze object metadata from USD files
        4. Save results in JSON and CSV formats
        5. Generate markdown table with images for documentation
        6. Clean up and close the simulation application
    """
    import argparse


    # Set up argument parser (parsed before importing heavy dependencies to allow --help without waiting)
    parser = argparse.ArgumentParser(description="Analyze USD files in a folder to extract rigid bodies and static bodies")
    from robolab.constants import SCENE_DIR
    parser.add_argument("--scene-folder", default=SCENE_DIR, help="Path to the folder containing USD files to analyze")
    parser.add_argument("--scene", help="Specific USD scene file to process (overrides --scene-folder)")
    parser.add_argument("--output-folder", default=os.path.join(SCENE_DIR, "_metadata"), help="Path to the folder to save the results")
    parser.add_argument("--generate-images", action="store_true", help="Generate images for the scenes. If enabled, will also call generate_scene_screenshots.py")
    parser.add_argument("--ignore-files", nargs="*", default=['gaussian_kitchen.usda'], help="List of file names to ignore (e.g., gaussian_kitchen.usda)")
    parser.add_argument("--ignore-folders", nargs="*", default=['marble'], help="List of folder names to ignore (e.g., marble)")
    args = parser.parse_args()

    # If --scene is provided, use it as the scene_folder (single file mode)
    if args.scene:
        # If it's just a filename, look for it in SCENE_DIR
        if not os.path.dirname(args.scene):
            args.scene_folder = os.path.join(SCENE_DIR, args.scene)
        else:
            args.scene_folder = args.scene

        # Validate the scene file exists
        if not os.path.exists(args.scene_folder):
            print(f"Error: Scene file does not exist: {args.scene_folder}")
            import sys
            sys.exit(1)

    if args.generate_images:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True})
        from generate_scene_screenshots import generate_scene_screenshots

        # Generate screenshots for main folder
        generate_scene_screenshots(app, args.scene_folder, os.path.join(SCENE_DIR, '_images'), view="angled", ignore_files=args.ignore_files if args.ignore_files else None, ignore_folders=args.ignore_folders if args.ignore_folders else None)

        # Also generate screenshots for llm_generated_settled subfolder
        llm_settled_dir = os.path.join(args.scene_folder, "llm_generated_settled")
        if os.path.isdir(llm_settled_dir):
            print(f"Generating screenshots for llm_generated_settled...")
            generate_scene_screenshots(app, llm_settled_dir, os.path.join(SCENE_DIR, '_images'), view="angled", ignore_files=args.ignore_files if args.ignore_files else None, ignore_folders=args.ignore_folders if args.ignore_folders else None)

        app.close()
    else:
        print("Skipping image generation and using existing images. Use --generate-images to re-generate images.")

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    # Generate metadata
    generate_scene_metadata(args.scene_folder, args.output_folder, ignore_files=args.ignore_files if args.ignore_files else None, ignore_folders=args.ignore_folders if args.ignore_folders else None)

    # Assumes you've already ran `generate_scene_images.py` to generate the images.

    # Generate Markdown table with images
    csv_file_path = os.path.join(args.output_folder, 'scene_table.csv')

    from robolab.core.utils.csv_utils import add_images_to_csv, save_markdown_table
    csv_data_with_images = add_images_to_csv(
        csv_file_path=csv_file_path,
        image_dir=os.path.join(SCENE_DIR, '_images'),
        column_name_to_img='scene',
        image_column_name='Preview',
        relative_dir=args.scene_folder
    )



    # Load scene statistics to include in the description
    scene_stats_path = os.path.join(args.output_folder, 'scene_statistics.json')
    description = 'A list of USD scenes available in the scene directory.'
    if os.path.exists(scene_stats_path):
        with open(scene_stats_path, 'r') as f:
            scene_stats = json.load(f)
        total_scenes = scene_stats.get('total_scenes', 'N/A')
        total_unique_objects = scene_stats.get('total_unique_objects', 'N/A')
        avg_objects = scene_stats.get('average_objects_per_scene', 'N/A')
        description += (
            f'\n\n**Total scenes:** {total_scenes} | '
            f'**Total unique objects:** {total_unique_objects} | '
            f'**Average objects per scene:** {avg_objects}'
        )

    save_markdown_table(
        csv_input=csv_data_with_images,
        output_path=os.path.join(SCENE_DIR, 'README.md'),
        title='Available Scenes',
        description=description
    )

    # Close the simulation app
    simulation_app.close()
