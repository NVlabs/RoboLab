# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Scene utilities for working with USD scenes.

This module provides utilities for:
- Finding scene files by name or path
- Scraping scene information from USD files
- Importing scenes as configclass objects
- Verifying objects exist in scenes
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import robolab.constants
from robolab.constants import SCENE_DIR
from robolab.core.utils.debug_utils import get_caller_info

ACCEPTED_SCENE_EXTENSIONS = (".usda", ".usdc", ".usdz", ".usd")


def find_scene_file(scene_path: str, scene_dir: str, ignore_directories: list[str] = ["not_used", "tmp"]) -> str:
    """Recursively search for a scene file matching the given path.

    This function searches through the scene directory and its subdirectories to find
    a file that ends with the specified scene_path. Only files with accepted USD
    extensions (.usda, .usdc, .usdz, .usd) are considered valid matches. scene_path can be a filename, or a relative path (e.g., "examples/banana_on_plate_task.usda")

    Args:
        scene_path (str): The scene file path to search for. If absolute, returns as-is.
                         If relative, searches for files ending with this path.
        scene_dir (str): The root directory to start searching from.
        ignore_directories (list[str], optional): List of directory names to skip during
                                                 search. Defaults to ["not_used", "tmp"].

    Returns:
        str: The full path to the matching scene file, or the original scene_path
             if no match is found.
    Note:
        - Directories starting with "_" are automatically ignored
        - Search is recursive and depth-first
        - Only the first match found is returned
        - If scene_path is already absolute, it's returned without searching
    """
    if os.path.isabs(scene_path):
        return scene_path

    for file in os.listdir(scene_dir):

        full_filepath = os.path.join(scene_dir, file)

        # Check if current file matches using '/' as the path separator and the file.
        if os.path.isfile(full_filepath) and full_filepath.endswith(ACCEPTED_SCENE_EXTENSIONS):
            if full_filepath.endswith(os.sep + scene_path) or full_filepath == scene_path:
                return full_filepath

        # If it's a directory, recursively search within it
        if os.path.isdir(full_filepath) and file not in ignore_directories and not file.startswith("_"):
            result = find_scene_file(scene_path, full_filepath)
            # Only return if we found a match (not just the original scene_path)
            if result != scene_path:
                return result

    return scene_path


@lru_cache(maxsize=256)
def _scrape_scene_cached(scene_path: str, objects_of_interest_tuple: tuple = None) -> tuple:
    """
    Cached internal implementation of scrape_scene.
    Uses tuple for objects_of_interest to enable caching (lists are not hashable).

    Returns a tuple of (scene_dict, contact_object_list, dynamic_body_names,
    kinematic_body_names, static_body_names) for cache compatibility.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    # Convert tuple back to list for internal processing
    objects_of_interest = list(objects_of_interest_tuple) if objects_of_interest_tuple else None

    scene_dict = {}
    scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn = sim_utils.UsdFileCfg(
                usd_path=str(scene_path),
                activate_contact_sensors=True,
                ),
            )
    from robolab.core.utils.usd_utils import get_usd_objects_info
    scene_objects = get_usd_objects_info(scene_path)
    dynamic_bodies = [obj for obj in scene_objects if obj['rigid_body'] and not obj.get('kinematic', False)]
    kinematic_bodies = [obj for obj in scene_objects if obj['rigid_body'] and obj.get('kinematic', False)]
    static_bodies = [obj for obj in scene_objects if not obj['rigid_body']]
    if robolab.constants.DEBUG:
        caller_info = get_caller_info()
        print(f"[{caller_info}] scene: {scene_path}")
        if objects_of_interest is not None:
            print(f"[{caller_info}] objects_of_interest: {objects_of_interest}")
        else:
            print(f"[{caller_info}] no objects_of_interest specified; adding all objects")

        print(f"[{caller_info}] Found scene_objects: {[obj.get('name') for obj in scene_objects]}")
        print(f"[{caller_info}]\t{len(dynamic_bodies)} dynamic bodies: {[obj.get('name') for obj in dynamic_bodies]}")
        print(f"[{caller_info}]\t{len(kinematic_bodies)} kinematic bodies: {[obj.get('name') for obj in kinematic_bodies]}")
        print(f"[{caller_info}]\t{len(static_bodies)} static bodies: {[obj.get('name') for obj in static_bodies]}")

    scene_dict["scene"] = scene

    contact_object_list = []
    dynamic_body_names = []
    kinematic_body_names = []
    static_body_names = []

    # Process dynamic rigid bodies as RigidObjectCfg
    for obj_info in dynamic_bodies:
        name = obj_info['name']

        if objects_of_interest is not None and name not in objects_of_interest:
            continue

        if objects_of_interest is None and name not in contact_object_list:
            contact_object_list.append(name)

        asset = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=obj_info['position'],
                rot=obj_info['rotation'],
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
        )
        if robolab.constants.DEBUG:
            caller_info = get_caller_info()
            print(f"[{caller_info}] adding RigidObjectCfg: {name} {asset.prim_path}")
        scene_dict[name] = asset
        dynamic_body_names.append(name)

    # Process kinematic bodies as AssetBaseCfg (avoids collision issues with
    # PhysX kinematic target timing while keeping them as scene geometry with
    # working collisions; pose is queryable via XFormPrimView)
    if objects_of_interest is not None:
        for obj_info in kinematic_bodies:
            name = obj_info['name']

            if name not in objects_of_interest:
                continue

            asset = AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                spawn=None,
            )
            if robolab.constants.DEBUG:
                caller_info = get_caller_info()
                print(f"[{caller_info}] adding AssetBaseCfg (kinematic): {name} {asset.prim_path}")
            scene_dict[name] = asset
            kinematic_body_names.append(name)

    # Process static bodies as AssetBaseCfg if objects_of_interest is specified
    if objects_of_interest is not None:
        for obj_info in static_bodies:
            name = obj_info['name']

            if name not in objects_of_interest:
                continue

            asset = AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                spawn=None,
            )
            if robolab.constants.DEBUG:
                caller_info = get_caller_info()
                print(f"[{caller_info}] adding AssetBaseCfg: {name} {asset.prim_path}")
            scene_dict[name] = asset
            static_body_names.append(name)


    return (
        scene_dict,
        tuple(contact_object_list),
        tuple(dynamic_body_names),
        tuple(kinematic_body_names),
        tuple(static_body_names),
    )


def scrape_scene(scene_path: str, objects_of_interest: list[str] = None) -> dict:
    """
    Scrape scene information from a USD file.
    Results are cached internally to avoid re-parsing the same scene configurations.

    Args:
        scene_path: Path to the USD scene file
        objects_of_interest: Optional list of object names to include. If None, includes all rigid bodies.

    Returns:
        dict with keys:
            - scene_dict: Mapping of object names to IsaacLab asset configurations.
            - contact_object_list: List of dynamic rigid body names for contact detection.
            - dynamic_bodies: List of dynamic rigid body names (RigidObjectCfg).
            - kinematic_bodies: List of kinematic rigid body names (AssetBaseCfg, pose queryable via XFormPrimView).
            - static_bodies: List of static body names (AssetBaseCfg).
    """
    objects_tuple = tuple(objects_of_interest) if objects_of_interest else None
    scene_dict, contact_tuple, dynamic_tuple, kinematic_tuple, static_tuple = _scrape_scene_cached(scene_path, objects_tuple)
    return {
        "scene_dict": dict(scene_dict),
        "contact_object_list": list(contact_tuple),
        "dynamic_bodies": list(dynamic_tuple),
        "kinematic_bodies": list(kinematic_tuple),
        "static_bodies": list(static_tuple),
    }


def clear_scene_cache():
    """Clear the scene scraping cache. Call this if USD files have been modified."""
    _scrape_scene_cached.cache_clear()


def import_scene(scene_path: str, objects_of_interest: list[str] = None) -> Any:
    """
    Import a scene file and create a configclass with scene objects.

    This function loads a USD scene file, extracts object information, and creates
    a dynamic configclass populated with Isaac Lab asset configurations for each
    scene object. Only objects specified in objects_of_interest are included.

    Args:
        scene_path (str): Path to the scene file. Can be absolute or relative.
                         If relative, will be searched for recursively in SCENE_DIR.
        objects_of_interest (list[str], optional): List of object names to include
                                                  in the scene configuration. If None,
                                                  only rigid body objects are included.

    Returns:
        Any: A configclass instance with attributes corresponding to scene objects.
             Each attribute contains Isaac Lab asset configuration.

    Raises:
        FileNotFoundError: If the scene file cannot be found.

    Note:
        - Dynamic rigid body objects are configured as RigidObjectCfg
        - Kinematic rigid body objects are configured as AssetBaseCfg (pose queryable via XFormPrimView)
        - Static body objects are configured as AssetBaseCfg (only if in objects_of_interest)
        - The scene itself is always included as a 'scene' attribute
    """
    from isaaclab.utils import configclass

    # Create a dynamic configclass
    @configclass
    class SceneConfig:
        pass

    if not os.path.isabs(scene_path):
        scene_path = find_scene_file(scene_path, SCENE_DIR)

    result = scrape_scene(scene_path, objects_of_interest)

    # Add attributes to the configclass based on scene_dict
    for name, asset in result["scene_dict"].items():
        setattr(SceneConfig, name, asset)

    return SceneConfig


def import_scene_and_contact_object_list(scene_path: str) -> tuple[Any, list[str]]:
    """
    Import a scene file and return both the configclass and list of contact objects.
    Same functionality as import_scene, but returns both the configclass and list of contact objects.

    This function loads a USD scene file and creates both a configclass with scene
    objects and a list of rigid body objects that can be used for contact detection.
    This is useful when you need to know which objects in the scene are interactive.

    Args:
        scene_path (str): Path to the scene file. Can be absolute or relative.
                         If relative, will be searched for recursively in SCENE_DIR.

    Returns:
        tuple[Any, list[str]]: A tuple containing:
            - A configclass instance with attributes for all rigid body scene objects
            - A list of rigid body object names suitable for contact detection

    Raises:
        FileNotFoundError: If the scene file cannot be found.

    Note:
        - Dynamic rigid body objects are included in both the configclass and contact list
        - Kinematic rigid body objects are configured as AssetBaseCfg
        - Static body objects are excluded from the configuration
        - The scene itself is always included as a 'scene' attribute in the configclass
    """
    from isaaclab.utils import configclass

    # Create a dynamic configclass
    @configclass
    class SceneConfig:
        pass

    if not os.path.isabs(scene_path):
        scene_path = find_scene_file(scene_path, SCENE_DIR)

    result = scrape_scene(scene_path, None)

    # Add attributes to the configclass based on scene_dict
    for name, asset in result["scene_dict"].items():
        setattr(SceneConfig, name, asset)

    return SceneConfig, result["contact_object_list"]


def get_scenes_from_folder(folder_path: str, recursive: bool = False, ext: list[str] = ACCEPTED_SCENE_EXTENSIONS, ignore_patterns: list[str] = ["base_empty.usda"]) -> list[str]:
    """Get all scene paths from a folder. Returns the absolute paths for strings. """
    folder = Path(folder_path)
    all_paths = []

    # Collect paths for each extension
    for extension in ext:
        if recursive:
            all_paths.extend(folder.rglob(f"*{extension}"))
        else:
            all_paths.extend(folder.glob(f"*{extension}"))

    filtered_paths = [path for path in all_paths if path.name not in ignore_patterns]
    return [str(path.resolve()) for path in filtered_paths]


def verify_objects_in_scene(object_list: list[str], usd_path: str) -> tuple[bool, str]:
    from robolab.core.utils.usd_utils import get_usd_objects_info

    # Get objects from the scene
    try:
        scene_objects = get_usd_objects_info(usd_path)
        scene_object_names = {obj['name'] for obj in scene_objects}

    except Exception as e:
        error = f"Error reading scene objects from '{usd_path}': {e}"
        return False, error

    # Check if all contact objects are in the scene
    missing_objects = []
    for obj_name in object_list:
        if obj_name not in scene_object_names:
            missing_objects.append(obj_name)

    if missing_objects:
        error = f"Objects in contact_object_list not found in scene: {missing_objects}"
        print(f"Objects in scene: {sorted(scene_object_names)}")
        print(f"Objects in contact_object_list: {object_list}")
        return False, error

    return True, None
