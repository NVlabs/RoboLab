# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import random
import re

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from robolab.constants import BACKGROUND_ASSET_DIR


def find_background_files(folder_path: str = BACKGROUND_ASSET_DIR, filename: str = None, extensions=None):
    """
    Search folder_path recursively for background files,
    excluding folders that start with underscore.

    Args:
        folder_path: Directory to search in (defaults to BACKGROUND_ASSET_DIR)
        filename: Optional specific filename to search for (returns only matching files)
        extensions: List of file extensions to search for.
                   If None, defaults to ['.hdr', '.exr']

    Returns:
        If filename is specified: returns the absolute file path (str) or None if not found.
        If filename is None: returns list of all absolute file paths found.
    """
    if extensions is None:
        extensions = ['.hdr', '.exr']

    # Normalize extensions to lowercase with dots
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]

    background_files = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(folder_path):
        # Filter out directories that start with underscore
        dirs[:] = [d for d in dirs if not d.startswith('_')]

        # Check each file
        for file in files:
            # Skip files that start with underscore
            if file.startswith('_'):
                continue

            # If searching for specific filename, check if it matches
            if filename is not None and file != filename:
                continue

            # Check if file has a valid extension
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in extensions:
                # Get relative path from folder_path
                full_path = os.path.join(root, file)
                background_files.append(full_path)

                # If searching for specific filename and found it, return immediately
                if filename is not None:
                    return full_path

    # If searching for specific filename but not found, return None
    if filename is not None:
        return None

    return sorted(background_files)


# Some hardcoded examples
@configclass
class EmptyWarehouseBackgroundCfg:
    dome_light = AssetBaseCfg(
    prim_path="/World/background",
    spawn=sim_utils.DomeLightCfg(
        texture_file=find_background_files(BACKGROUND_ASSET_DIR, "empty_warehouse.hdr"),
        intensity=500.0,
        visible_in_primary_ray=True,
        texture_format="latlong",
        ),
    )

@configclass
class BilliardHallBackgroundCfg:
    dome_light = AssetBaseCfg(
    prim_path="/World/background",
    spawn=sim_utils.DomeLightCfg(
        texture_file=find_background_files(BACKGROUND_ASSET_DIR, "billiard_hall.hdr"),
        intensity=500.0,
        visible_in_primary_ray=True,
        texture_format="latlong",
        ),
    )

@configclass
class BrownPhotoStudioBackgroundCfg:
    dome_light = AssetBaseCfg(
    prim_path="/World/background",
    spawn=sim_utils.DomeLightCfg(
        texture_file=find_background_files(BACKGROUND_ASSET_DIR, "brown_photostudio.hdr"),
        intensity=500.0,
        visible_in_primary_ray=True,
        texture_format="latlong",
        ),
    )

@configclass
class HomeOfficeBackgroundCfg:
    dome_light = AssetBaseCfg(
    prim_path="/World/background",
    spawn=sim_utils.DomeLightCfg(
        texture_file=find_background_files(BACKGROUND_ASSET_DIR, "home_office.exr"),
        intensity=500.0,
        visible_in_primary_ray=True,
        texture_format="latlong",
        ),
    )


def find_and_generate_background_config(filename: str | None = None,
                                        folder_path: str = BACKGROUND_ASSET_DIR,
                                        class_name: str | None = None,
                                        intensity: float = 500.0):
    """
    Generate a configclass containing a background given the filename.

    Args:
        filename: The name of the background file (e.g., "my_background.hdr")
        folder_path: Directory to search for the background file (defaults to BACKGROUND_ASSET_DIR)
        class_name: Optional custom class name. If None, generates from filename.
        intensity: Light intensity value (default: 500.0)

    Returns:
        A configclass with dome light configuration for the specified background file.

    Raises:
        FileNotFoundError: If the specified filename is not found in folder_path.
    """
    # If no specific filename provided, choose a random background from the folder
    if filename is None:
        all_backgrounds = find_background_files(folder_path=folder_path)
        if not all_backgrounds:
            raise FileNotFoundError(f"No background files found in '{folder_path}'")
        background_path = random.choice(all_backgrounds)
    else:
        background_path = find_background_files(folder_path=folder_path, filename=filename)
        if background_path is None:
            raise FileNotFoundError(f"Background file '{filename}' not found in '{folder_path}'")

    return generate_background_config(background_path, class_name, intensity)


def generate_background_config(background_path: str, class_name: str = None, intensity: float = 500.0):
    """
    Generate a configclass containing a background given the filename.

    Args:
        background_path: The path to the background file.
        class_name: Optional custom class name. If None, generates from filename.
        intensity: Light intensity value (default: 500.0)

    Returns:
        A configclass with dome light configuration for the specified background file.

    Raises:
        FileNotFoundError: If the specified filename is not found in folder_path.
    """
    # Check that background_path exists and is a file
    if not os.path.isfile(background_path):
        raise FileNotFoundError(f"Background file '{background_path}' does not exist or is not a file.")

    # Generate class name if not provided
    if class_name is None:
        # Remove file extension and convert to CamelCase using background_path basename
        base_name = os.path.splitext(os.path.basename(background_path))[0]
        # Convert underscores and spaces to camelcase
        words = re.split(r'[_\s]+', base_name)
        class_name = ''.join(word.capitalize() for word in words) + "BackgroundCfg"

    # Create the configclass dynamically
    @configclass
    class GeneratedBackgroundConfig:
        dome_light = AssetBaseCfg(
            prim_path="/World/background",
            spawn=sim_utils.DomeLightCfg(
                texture_file=background_path,
                intensity=intensity,
                visible_in_primary_ray=True,
                texture_format="latlong",
            ),
        )

    # Set the class name for better debugging/introspection
    GeneratedBackgroundConfig.__name__ = class_name
    GeneratedBackgroundConfig.__qualname__ = class_name

    return GeneratedBackgroundConfig
