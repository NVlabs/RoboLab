# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
USD Scene Screenshot Generator

This module provides functionality to render USD scene files to screenshot images using Isaac Sim.
It can process either individual USD files or entire directories of USD files, automatically
adding lighting and rendering high-quality screenshots for visualization purposes.

The generated screenshots are commonly used for:
- Scene documentation and visualization
- Creating preview images for scene libraries
- Generating markdown tables with scene previews
- Quality assurance and scene validation

Usage:
    python generate_scene_screenshots.py --usd-folder /path/to/scenes --output-folder /path/to/output
    python generate_scene_screenshots.py --scene kitchen.usda banana_bowl.usda
    python generate_scene_screenshots.py --scene kitchen/kitchen.usda /absolute/path/to/scene.usda

Dependencies:
    - Isaac Sim (isaacsim package)
    - robolab.core.utils.render_utils
    - robolab.core.utils.file_utils
"""

import argparse
import os

from robolab.constants import SCENE_DIR


def generate_scene_screenshots(
    app,
    usd_folder: str,
    output_folder: str,
    resolution: tuple[int, int] = (640, 480),
    skip_frames: int = 100,
    add_lighting: bool = True,
    view: str = "front",
    ignore_folders: list[str] | None = None,
    ignore_files: list[str] | None = None,
    scene_files: list[str] | None = None,
):
    """
    Generate screenshot images for USD scene files using Isaac Sim.

    This function can process a directory containing multiple USD files, or a specific list
    of scene files. For each USD file found, it renders a screenshot with optional lighting
    and saves it to the specified output folder.

    Args:
        app: Isaac Sim SimulationApp instance used for rendering
        usd_folder (str): Path to a directory containing USD files to render.
            Used when scene_files is not provided.
        output_folder (str): Directory where the generated screenshot images will be saved
        resolution (tuple[int, int], optional): Image resolution as (width, height).
            Defaults to (640, 480)
        skip_frames (int, optional): Number of simulation frames to skip before rendering
            to allow the scene to stabilize. Defaults to 100
        add_lighting (bool, optional): Whether to add default lighting to the scene
            for better visualization. Defaults to True
        ignore_folders (list[str] | None, optional): List of folder names to ignore
            when processing a directory. Files in these folders will be skipped.
            Defaults to None
        ignore_files (list[str] | None, optional): List of file names to ignore.
            Files matching these names will be skipped. Defaults to None
        scene_files (list[str] | None, optional): Explicit list of USD file paths to render.
            If provided, usd_folder is ignored. Defaults to None.

    Raises:
        ValueError: If usd_folder is not a valid directory (when scene_files is not provided)

    Note:
        - Supported USD formats: .usd, .usda, .usdc, .usdz
        - Generated images are saved with the same base name as the USD file
        - Default lighting is added to scenes that may not have proper illumination
    """
    from robolab.core.utils.file_utils import find_usd_files
    from robolab.core.utils.render_utils import render_stage_frame

    if scene_files:
        usds = scene_files
    elif os.path.isdir(usd_folder):
        usds = find_usd_files(usd_folder)
    else:
        raise ValueError(f"Invalid USD folder: {usd_folder}")

    # Filter out ignored folders
    if ignore_folders:
        usds = [u for u in usds if not any(folder in str(u) for folder in ignore_folders)]

    # Filter out ignored files
    if ignore_files:
        usds = [u for u in usds if os.path.basename(str(u)) not in ignore_files]

    if view == "front":
        camera_position = (1, 0, 0.7)
        camera_target = (0.5, 0.0, 0.0)
    elif view == "angled":
        camera_position = (-0.3, 0.3, 0.7)
        camera_target = (0.5, 0.0, 0.0)
    elif view == "top":
        camera_position = (0.5, 0, 1)
        camera_target = (0.5, 0.0, 0.0)
    else:
        raise ValueError(f"Invalid view: {view}")

    for usd_file in usds:
        output_path = render_stage_frame(
            app,
            usd_path=usd_file,
            output_dir=output_folder,
            resolution=resolution,
            skip_frames=skip_frames,
            camera_position=camera_position,
            camera_target=camera_target,
            add_lighting=add_lighting,
        )
        print(f"Rendered {usd_file} to {output_path}")

def main():
    """
    Main entry point for the USD scene screenshot generation script.

    This function handles command-line argument parsing and orchestrates the screenshot
    generation process. It initializes Isaac Sim in headless mode and processes the
    specified USD files or directories.

    Command-line Arguments:
        --usd-folder: Path to directory of USD files to process (default: SCENE_DIR)
        --scene: One or more scene paths to render. If a full path is specified, it is
            used directly. Otherwise, it is treated as a partial path from SCENE_DIR.
        --output-folder: Output directory for screenshots (default: SCENE_DIR/_images)

    The script automatically:
    - Initializes Isaac Sim in headless mode for batch processing
    - Processes all USD files in the specified path
    - Adds default lighting to scenes for better visualization
    - Saves screenshots with consistent naming convention
    """

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Render a USD file or a folder of USD files to an image")
    parser.add_argument("--usd-folder", default=SCENE_DIR, help="Path to the folder of USD files to render")
    parser.add_argument("--scene", nargs="+", help="One or more scene paths to render (full path used directly, otherwise relative to SCENE_DIR)")
    parser.add_argument("--output-folder", default=os.path.join(SCENE_DIR, "_images"), help="Folder to save the rendered image")
    parser.add_argument("--view", default="angled", help="View to render the scene from")
    parser.add_argument("--ignore-files", nargs="*", default=['gaussian_kitchen.usda'], help="List of file names to ignore (e.g., gaussian_kitchen.usda)")
    parser.add_argument("--ignore-folders", nargs="*", default=['marble'], help="List of folder names to ignore (e.g., marble)")

    # Parse args before importing IsaacSim to allow --help without waiting
    args = parser.parse_args()

    # Resolve scene paths if --scene is provided
    scene_files = None
    if args.scene:
        scene_files = []
        for scene in args.scene:
            if os.path.isabs(scene):
                resolved = scene
            else:
                resolved = os.path.join(SCENE_DIR, scene)
            if not os.path.exists(resolved):
                print(f"Error: Scene file does not exist: {resolved}")
                return
            scene_files.append(resolved)

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})

    generate_scene_screenshots(
        app,
        args.usd_folder,
        args.output_folder,
        view=args.view,
        ignore_files=args.ignore_files if args.ignore_files else None,
        ignore_folders=args.ignore_folders if args.ignore_folders else None,
        scene_files=scene_files,
    )

if __name__ == "__main__":
    main()
