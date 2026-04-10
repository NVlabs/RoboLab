# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import glob
import json
import os
from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
import yaml

# from isaaclab.utils.datasets import episode_data

def find_usd_files(
    path: Union[str, Path], 
    extension: str = None,
    recursive: bool = True,
    exclude_underscore_dirs: bool = True,
    exclude_materials: bool = True
) -> List[Path]:
    """
    Find USD files in a directory.
    
    Args:
        path: File or directory path to search
        extension: Specific extension to find (e.g., '.usd', '.usda'). 
                   If None, finds all USD types (.usd, .usda, .usdc, .usdz)
        recursive: Search subdirectories recursively
        exclude_underscore_dirs: Skip directories starting with '_'
        exclude_materials: Skip 'materials' directories
        
    Returns:
        Sorted list of Path objects to USD files
    """
    path = Path(path)
    
    if path.is_file():
        if extension is None or path.suffix.lower() == extension.lower():
            return [path]
        return []
    
    if not path.is_dir():
        return []
    
    # Determine extensions to search
    if extension:
        extensions = [extension]
    else:
        extensions = ['.usd', '.usda', '.usdc', '.usdz']
    
    usd_files = []
    
    for ext in extensions:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue
                
            # Get relative path parts (excluding filename)
            try:
                rel_parts = file_path.relative_to(path).parts[:-1]
            except ValueError:
                rel_parts = ()
            
            # Apply exclusion filters
            if exclude_underscore_dirs and any(part.startswith('_') for part in rel_parts):
                continue
            if exclude_materials and any(part.lower() == 'materials' for part in rel_parts):
                continue
                
            usd_files.append(file_path)
    
    return sorted(set(usd_files))


# Helper function to convert file paths based on path_type
def convert_file_path(filepath: str, path_type: str, output_dir: str = None) -> str:
    """Convert file paths based on the specified path_type."""
    if not isinstance(filepath, str):
        return str(filepath)

    # Check if this looks like a file path (absolute or with file extension)
    is_file_path = (os.path.isabs(filepath) or
                    any(filepath.endswith(ext) for ext in ['.py', '.usd', '.usda', '.usdc', '.usdz', '.json', '.csv', '.md', '.txt', '.yaml', '.yml']))

    if not is_file_path:
        return filepath

    if path_type == "absolute":
        # Keep as-is
        return filepath
    elif path_type == "relative":
        # Convert absolute paths to relative paths
        if os.path.isabs(filepath):
            try:
                relative_path = os.path.relpath(filepath, output_dir)
                # Normalize path separators for markdown (use forward slashes)
                return relative_path.replace(os.sep, '/')
            except ValueError:
                # If conversion fails (e.g., different drives on Windows), return original
                return filepath
        return filepath
    elif path_type == "filename_only":
        # Extract just the filename
        return os.path.basename(filepath)
    else:
        # Invalid path_type, return as-is
        return filepath


def get_relative_path(input_path: str, relative_path: str) -> str:
    """Get the relative path of a file from a search directory to a relative directory."""
    # Get the absolute path of the search directory
    search_path = os.path.abspath(input_path)
    # Get the absolute path of the relative directory
    relative_path = os.path.abspath(relative_path)
    # Get the relative path of the file from the search directory to the relative directory
    return os.path.relpath(search_path, relative_path)

def load_hdf5_episode(filepath: str, episode: int) -> dict:
    """Load episode from an HDF5 file."""
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        data = f.get('data')
        if data is not None:
            ep = data.get(f'demo_{episode}')

            # Copy all sub-datasets into a dict of numpy arrays
            episode_data = {}
            for name, ds in ep.items():
                # ds[()] reads the full dataset into memory
                episode_data[name] = ds[()]
        else:
            raise ValueError(f"Episode {episode} not found in {filepath}")

    return episode_data

def load_hdf5_episode_data(filepath: str, episode: int, key: str) -> np.ndarray:
    """Load actions from an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        data = f.get('data')
        ep = data.get(f'demo_{episode}')
        if ep is not None:
            actions = ep.get(key)

            # Only copy "actions" into an np array
            actions = ep.get(key)[()]
        else:
            raise ValueError(f"Episode {episode} not found in {filepath}")

    return actions


def load_param_file(file_path: str, parent_dir: str = None) -> dict:
    file_path = validate_file_path(file_path, parent_dir)
    if file_path.endswith(".yaml"):
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

            if data is None:
                raise ValueError(f"File '{file_path}' is invalid")
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(
            f"File '{file_path}' is an unsupported type; supported types: [json/yaml]"
        )
    return data

def load_file(file_path: str):
    """Load a file from a path."""
    file_path = validate_file_path(file_path)
    if file_path.endswith(".json"):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"File '{file_path}' is not a valid JSON file")
    elif file_path.endswith(".yaml"):
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError:
            raise ValueError(f"File '{file_path}' is not a valid YAML file")
    else:
        raise ValueError(f"File '{file_path}' is an unsupported type; supported types: [json/yaml]")

def validate_file_extension(file_path: str, ext: str) -> str:
    """
    Ensures that the given filepath has the correct extension.

    Args:
        filepath (str): The file path or filename.
        ext (str): The required extension, with or without a leading dot.

    Returns:
        str: The filepath with the correct extension if it was missing or incorrect.
    """

    # Ensure extension starts with a dot
    if not ext.startswith("."):
        ext = "." + ext

    # Extract current file extension
    base, current_ext = os.path.splitext(file_path)

    # Check if the extension matches
    if current_ext.lower() != ext.lower():
        return base + ext

    return file_path


def validate_file_path(file_path: str, parent_dir: str = None) -> str:
    """
    Checks if the file exists. If the path is not absolute, prepends the parent directory.

    Args:
        file_path (str): The file path to check.
        parent_dir (str): The parent directory to use if the path is not absolute.

    Returns:
        str: The absolute file path if the file exists, otherwise raises an error.
    """
    if not os.path.isabs(file_path):
        if parent_dir is not None:
            file_path = os.path.join(parent_dir, file_path)
        else:
            raise ValueError(
                f"File is not an absolute filepath: {file_path} and no parent is specified"
            )
    if os.path.exists(file_path):
        return os.path.abspath(file_path)
    else:
        raise FileNotFoundError(f"File does not exist: {file_path}")


def get_filename_without_extension(filepath: str) -> str:
    """Returns the filename only, without the extension or the full filepath."""
    return os.path.basename(filepath).split(".")[0]


def find_ext(dir: str, ext: Union[str, List[str]], full_filepath=True, recursive=True):
    """Given a directory and an extension, return all valid files matching that extension.

    Args:
        dir: valid directory path
        ext (str or list of str): an extension, starting with the dot.
        full_filepath (bool): If the list of paths should specify the full filepath.
        recursive (bool): if the directory should be crawled recursively
    """
    # Check that directory is a valid directory
    if not os.path.isdir(dir):
        raise FileNotFoundError(
            f"Directory {dir} does not exist. Did you provide the correct path?"
        )

    # Check if a single extension or a list of extensions are provided
    if isinstance(ext, str):
        single_ext = True
    elif all(isinstance(e, str) for e in ext):
        single_ext = False
    else:
        raise ValueError("Argument 'ext' needs to be of type str or list of str.")

    # Iterate through directories
    ext_files = []
    iterator = os.walk(dir) if recursive else ((next(os.walk(dir))),)
    for root, dirs, files in iterator:
        for file in files:
            # Check for all the possible extensions.
            for e in ext if not single_ext else [ext]:
                if file.lower().endswith(e):
                    filepath = os.path.join(root, file) if full_filepath else file
                    ext_files.append(filepath)
    return ext_files


def check_ext(file: str, ext: Union[str, List[str]]) -> bool:
    """checks a full filepath is indeed a file and has the correct extension. Returns True if the file is of the type of extension specified.

    Args:
        file (str): a full filepath to the file to be examined.
        ext (str or list of str): acceptable extensions, wit or without the dot.

    """
    path = Path(file)

    # First check if file is valid
    if not path.is_file():
        raise FileNotFoundError(f"Specified file '{file}' is not a valid file.")

    # Check if a single extension or a list of extensions are provided
    if isinstance(ext, str):
        single_ext = True
    elif all(isinstance(e, str) for e in ext):
        single_ext = False
    else:
        raise ValueError("Argument 'ext' needs to be of type str or list of str.")

    # Check ext against a set of possible extensions.
    for e in ext if not single_ext else [ext]:
        if e.startswith("."):
            if path.suffix == e:
                return True
        else:
            if path.suffix == "." + e:
                return True

    return False


def h5tree(filepath):
    """Prints a tree of the data in the file."""

    with h5py.File(filepath, "r") as hf:
        print(f"{os.path.basename(filepath)}")
        h5tree_recurse(hf)


def h5tree_recurse(val, pre=""):
    """The recursive function for printing a data tree. You can call this function directly with the h5py file."""
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if isinstance(val, h5py._hl.group.Group):
                print(f"{pre}└── {key}")
                # print(pre + '└── ' + key)
                h5tree_recurse(val, pre + "    ")
            else:
                print(f"{pre}└── {key} {val.shape}")
                # print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if isinstance(val, h5py._hl.group.Group):
                print(f"{pre}├── {key}")
                # print(pre + '├── ' + key)
                h5tree_recurse(val, pre + "│   ")
            else:
                print(f"{pre}├── {key} {val.shape}")
                # print(pre + '├── ' + key + ' (%d)' % len(val))


def save_json(data: dict | list, filepath: str | Path, indent: int = 2) -> None:
    """Write data to a JSON file.

    Args:
        data: Serializable data (dict or list).
        filepath: Destination path (str or Path).
        indent: JSON indentation level.
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)


def write_dict_to_json(key: str, data_dict: dict, filepath: str):
    """Write a dictionary to file with a specific key.
    Note, the heading will be replaced if it already exists.

    The json file must be a single dictionary with subheadings:
    {
        "key1": {
            "subkey1": value1,
            "subkey2": value2
        },
        "key2": [
        ...
        ]
    }
    """
    data = dict()

    # Check if file exists. If so, load the existing data from it.
    if os.path.isfile(filepath):
        with open(filepath, "r") as json_file:
            data = json.load(json_file)
            if not isinstance(data, dict):
                raise ValueError(
                    "The json file provided must contain a single dictionary."
                )

    # Replace data with key with the current dict.
    data[key] = data_dict

    # Write to file.
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=2)
        print(f"'{key}' written to {filepath}")
    return


def get_latest_subdirectory(parent_dir) -> str:
    """Returns the latest modified subdirectory"""
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Parent directory does not exist: {parent_dir}")
    all_subdirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)

    return latest_subdir


def get_incremented_filepath(filepath):
    """
    Get an incremented filepath if the filepath already exists in that directory.

    Args:
    filepath (str): The base filepath (including extension).

    Returns:
    str: A unique filepath with an incremented suffix.
    """
    if not os.path.exists(filepath):
        return filepath

    base, ext = os.path.splitext(filepath)
    i = 1
    new_filepath = f"{base}_{i}{ext}"

    while os.path.exists(new_filepath):
        i += 1
        new_filepath = f"{base}_{i}{ext}"

    return new_filepath


def get_list_of_files_with_extension(
    dir: str, ext: Union[str, List], recursive: bool = False
) -> List[str]:
    if not os.path.isdir(dir):
        raise ValueError(f"Directory '{dir}' is not a valid directory")

    if isinstance(ext, str):
        ext = [ext]

    # Make sure all extensions start with a dot
    ext = [
        extension if extension.startswith(".") else "." + extension for extension in ext
    ]

    file_paths = []
    if recursive:
        # Walk through the directory and its subdirectories
        for root, dirs, files in os.walk(dir):
            for file in files:
                # Check if the file ends with any of the specified extensions
                if any(file.endswith(extension) for extension in ext):
                    # Construct the full path and add to the list
                    file_paths.append(os.path.join(root, file))
    else:
        # Only walk through the current level directory
        folder_items = os.listdir(dir)
        for file in folder_items:
            if any(file.endswith(extension) for extension in ext):
                file_paths.append(os.path.join(dir, file))

    return file_paths


def expand_folder_patterns(
    patterns: list[str],
    base_dir: str | None = None,
) -> tuple[list[str], bool]:
    """Expand glob patterns to matching directory paths.

    Each pattern can be a literal path or a glob (e.g., 'pi0_*'). Non-absolute
    patterns are resolved relative to *base_dir* when provided.

    Args:
        patterns: Folder names or glob patterns.
        base_dir: Optional base directory prepended to non-absolute patterns
            before globbing. When None, patterns are globbed as-is.

    Returns:
        (deduplicated list of directory paths, whether any pattern was expanded)
    """
    folders: list[str] = []
    pattern_expanded = False

    for pattern in patterns:
        if os.path.isabs(pattern):
            full_pattern = pattern
        elif base_dir is not None:
            full_pattern = os.path.join(base_dir, pattern)
        else:
            full_pattern = pattern

        matches = sorted(glob.glob(full_pattern))
        matches = [m for m in matches if os.path.isdir(m)]

        if matches:
            if len(matches) > 1 or matches[0] != full_pattern:
                pattern_expanded = True
            folders.extend(matches)
        else:
            folders.append(full_pattern)

    seen: set[str] = set()
    unique: list[str] = []
    for f in folders:
        real = os.path.realpath(f)
        if real not in seen:
            seen.add(real)
            unique.append(f)

    return unique, pattern_expanded


def confirm_folders(
    folders: list[str],
    default_yes: bool = True,
) -> list[str]:
    """Prompt user to confirm a folder selection.

    Args:
        folders: Candidate folder paths.
        default_yes: When True the default action (Enter) includes all folders;
            when False the default action aborts.

    Returns:
        Confirmed folders, or an empty list if the user aborts.
    """
    if default_yes:
        prompt = "\nInclude all? [Y/n/s] (s=select individually): "
    else:
        prompt = "Proceed with these folders? [y/N/s] (s=select individually): "

    while True:
        response = input(prompt).strip().lower()

        if response == "":
            if default_yes:
                return folders
            else:
                print("Aborted.")
                return []
        elif response == "y":
            return folders
        elif response == "n":
            print("Aborted.")
            return []
        elif response == "s":
            confirmed: list[str] = []
            for folder in folders:
                name = os.path.basename(folder)
                while True:
                    r = input(f"  Include '{name}'? [Y/n]: ").strip().lower()
                    if r in ("", "y"):
                        confirmed.append(folder)
                        break
                    elif r == "n":
                        break
                    else:
                        print("    Please enter 'y' or 'n'")

            if not confirmed:
                print("No folders selected.")
                return []

            print(f"\nSelected {len(confirmed)} folder(s):")
            for f in confirmed:
                print(f"  - {os.path.basename(f)}")
            return confirmed
        else:
            print("Please enter 'y', 'n', or 's'")


def get_folders_in_dir(dir: str) -> List[str]:
    if not os.path.isdir(dir):
        raise ValueError(f"Directory '{dir}' is not a valid directory")

    folder_items = os.listdir(dir)
    folders = [
        instance
        for instance in folder_items
        if os.path.isdir(os.path.join(dir, instance))
    ]

    return folders


def get_filename(filepath, without_extension: bool = True) -> str:
    filename = os.path.basename(filepath)
    if without_extension:
        return os.path.splitext(filename)[0]
    else:
        return filename


def get_filepath_with_extensions(dir: str, filename: str, ext: Union[str, List]):
    """
    Gets a file with names with a set of acceptable extensions. If both are available, it returns the first ext instead.
    """
    if isinstance(ext, str):
        ext = [ext]

    # Make sure all extensions start with a dot
    ext = [
        extension if extension.startswith(".") else "." + extension for extension in ext
    ]

    # Check if the current file exists, and if so, return the full file path directly
    if any(filename.endswith(extension) for extension in ext) and os.path.isfile(
        os.path.join(dir, filename)
    ):
        return os.path.join(dir, filename)

    # strip the extension and search for other files with the same name and same extension
    filename_wo_ext = get_filename(filename, without_extension=True)
    for extension in ext:
        # Check if each file exists
        path = os.path.join(dir, filename_wo_ext + extension)
        if os.path.isfile(path):
            print(f"Found alternative file '{filename_wo_ext + extension}'.")
            return path

    err = f"Provided file name {filename} cannot be found in directory {dir} with an valid extension. Valid extensions: {ext}"
    raise FileNotFoundError(err)


def write_class_params_to_file(obj, filepath):
    import json

    """
    Saves only the values of parameters defined in the __init__ method
    of the given object's class to a JSON file.

    Parameters:
    - obj: The object whose `__init__` parameters will be saved.
    - filepath: The JSON file to write the parameters to.
    """
    init_values = get_class_params_as_dict(obj)

    # Write the result to a JSON file
    with open(filepath, "w") as f:
        json.dump(init_values, f, indent=4)


def get_class_params_as_dict(obj) -> dict:
    """
    Saves only the values of parameters defined in the __init__ method
    of the given object's class to a JSON file.

    Parameters:
    - obj: The object whose `__init__` parameters will be saved.
    - filepath: The JSON file to write the parameters to.
    """
    import inspect

    # Get the `__init__` method of the class
    init_method = getattr(obj.__class__, "__init__", None)

    if not init_method:
        raise ValueError(
            f"The class {obj.__class__.__name__} does not have an __init__ method."
        )

    # Get the parameter names from the `__init__` method, excluding 'self'
    init_params = inspect.signature(init_method).parameters
    init_param_names = [param for param in init_params if param != "self"]

    # Filter instance attributes based on the `__init__` parameters
    # init_values = {key: getattr(obj, key) for key in init_param_names if hasattr(obj, key)}

    init_values = {}
    for key in init_param_names:
        if hasattr(obj, key):
            value = getattr(obj, key)
            try:
                # Test if the value is serializable
                json.dumps(value)
                init_values[key] = value
            except (TypeError, ValueError):
                # Skip unserializable values
                continue

    return init_values
