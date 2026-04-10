# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Environment factory for automatically creating and managing task environments.

This module provides a factory pattern for automatically creating environments
from task files, integrating with the existing environment creation system.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import robolab.constants
from robolab.constants import TASK_DIR
from robolab.core.environments.config import generate_env_cfg_from_task, print_env_cfg
from robolab.core.task.task_utils import resolve_task_path


class EnvFactory:
    """
    Factory class for automatically generating and managing task environments.

    This class provides a centralized way to create environments from task files
    and manage their registration with the environment system. It maintains
    structured information about all environments including their task origins,
    registration status, and tag memberships.

    Attributes:
        task_dir (Path): Directory containing task files
        _env_info (dict): Structured storage of environment information
        _tag_metadata (dict): Metadata for tags
    """

    def __init__(self, task_dir: str):
        """
        Initialize the factory.

        Args:
            task_dir: Directory containing task files
        """
        self.task_dir = Path(task_dir)

        # Structured environment storage
        self._env_info: dict[str, dict[str, Any]] = {}  # env_name -> env_info_dict

        # Tag metadata
        self._tag_metadata: dict[str, dict[str, Any]] = {}  # tag_name -> metadata

    def _store_env_info(self, env_name: str, task_file_name: str, env_cfg_class: type,
                       tags: set[str] | None = None) -> None:
        """
        Store environment information in structured format.

        Args:
            env_name: Environment name (same as Gymnasium registration name)
            task_file_name: Name of the task file (stem)
            env_cfg_class: Environment configuration class
            tags: Set of tag names this environment belongs to
        """
        # Get task_name from config class (base task name from Task class)
        task_name = getattr(env_cfg_class, '_task_name', None) or task_file_name

        self._env_info[env_name] = {
            'task_name': task_name,
            'task_file_name': task_file_name,
            'config_class_name': env_cfg_class.__name__,
            'config_class': env_cfg_class,
            'tags': tags or set(),
        }

    def create_env_cfg(self,
                       task: str,
                       tags: str | list[str] = "all",
                       env_name: str | None = None,
                       env_prefix: str = "",
                       env_postfix: str = "",
                       **env_kwargs) -> type:
        """
        Create an environment configuration from a task name or file path.

        This method auto-detects the task identifier type and resolves it:
        1. Full file path (contains '/' or '\\') - used directly
        2. Filename ending in '.py' - looked up in task_dir
        3. Task name - searched recursively in task_dir for matching .py file

        Args:
            task: Task identifier, can be:
                - Full path: "/path/to/BananaTask.py"
                - Filename: "BananaTask.py"
                - Task name: "BananaTask"
            tags: Tag name(s) to register the task to
            env_name: Name for the environment
            env_prefix: Prefix for environment name
            env_postfix: Postfix for environment name
            **env_kwargs: Additional environment configuration parameters

            Example env_kwargs:
                observations_cfg=ObservationCfg(),
                actions_cfg=DroidJointPositionActionCfg(),
                robot_cfg=DroidCfg,
                camera_cfg=OverShoulderLeftCameraCfg,  # could be a list of camera configurations
                lighting_cfg=SphereLightCfg,  # could be a list of lighting configurations
                background_cfg=HomeOfficeBackgroundCfg,
                contact_gripper=contact_gripper,
                dt=1 / (60 * 2),
                render_interval=8,
                decimation=8,
                seed=1,

        Returns:
            The generated environment configuration class

        Examples:
            # Using task name (looks in task_dir)
            factory.create_env_cfg("BananaTask")

            # Using full file path
            factory.create_env_cfg("/path/to/tasks/BananaTask.py")

            # With tags and postfix
            factory.create_env_cfg("BananaTask", tags="my_tag", env_postfix="v2")

            # With full configuration
            factory.create_env_cfg(
                "BananaTask",
                tags="background_variations",
                env_postfix="bg_warehouse",
                observations_cfg=ObservationCfg(),
                actions_cfg=DroidJointPositionActionCfg(),
                robot_cfg=DroidCfg,
                camera_cfg=[OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg],
                background_cfg=WarehouseBackgroundCfg,
            )
        """
        task_file_path, task_name = resolve_task_path(task, self.task_dir)

        # Create and register the environment
        env_cfg_class, actual_env_name = generate_env_cfg_from_task(
            task_file_path=task_file_path,
            env_name=env_name,
            env_prefix=env_prefix,
            env_postfix=env_postfix,
            register=True,
            **env_kwargs
        )

        # Prepare tags
        if isinstance(tags, str):
            tags_set = {tags}
        else:
            tags_set = set(tags)

        if "all" not in tags_set:
            tags_set.add("all")

        # Attributes will be automatically added to tags
        if hasattr(env_cfg_class, "_task_attributes") and env_cfg_class._task_attributes is not None:
            tags_set.update(env_cfg_class._task_attributes)

        # Store in structured format
        self._store_env_info(
            env_name=actual_env_name,
            task_file_name=task_name,
            env_cfg_class=env_cfg_class,
            tags=tags_set
        )

        return env_cfg_class

    def batch_create_env_cfgs(self,
                         tasks: list[str],
                         tags: str | list[str] = "all",
                         task_subdirs: list[str] | None = None,
                         env_prefix: str = "",
                         env_postfix: str = "",
                         **env_kwargs) -> dict[str, type]:
        """
        Create multiple environments from a list of task names or file paths.

        Args:
            tasks: List of task names or file paths to create environments for
            tags: Tag name(s) to register the tasks to
            task_subdirs: List of subdirectories to search (e.g., ["pick_place", "stacking_reorient"])
                         If None, searches the entire task directory recursively
            env_prefix: Prefix for environment names
            env_postfix: Postfix for environment names
            **env_kwargs: Additional environment configuration parameters

            Example env_kwargs:
                observations_cfg=ObservationCfg(),
                actions_cfg=DroidJointPositionActionCfg(),
                robot_cfg=DroidCfg,
                camera_cfg=OverShoulderLeftCameraCfg,  # could be a list of camera configurations
                lighting_cfg=SphereLightCfg,  # could be a list of lighting configurations
                background_cfg=HomeOfficeBackgroundCfg,
                contact_gripper=contact_gripper,
                dt=1 / (60 * 2),
                render_interval=8,
                decimation=8,
                seed=1,

        Returns:
            Dictionary mapping task names to generated environment classes

        Examples:
            # Create environments for multiple tasks
            factory.batch_create_env_cfgs(
                ["BananaTask", "AppleTask", "OrangeTask"],
                tags="fruit_tasks",
            )

            # With task_subdirs to limit search scope
            factory.batch_create_env_cfgs(
                ["BananaTask", "AppleTask"],
                tags="fruit_tasks",
                task_subdirs=["pick_place", "fruits"],
            )

            # Mixed task names and file paths
            factory.batch_create_env_cfgs(
                ["BananaTask", "/path/to/CustomTask.py"],
                tags="my_tag",
                env_postfix="v2",
            )

            # With full configuration
            factory.batch_create_env_cfgs(
                ["BananaTask", "AppleTask"],
                tags="evaluation",
                observations_cfg=ObservationCfg(),
                robot_cfg=DroidCfg,
                camera_cfg=[OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg],
            )
        """
        generated_envs = {}

        # If task_subdirs is provided, build a mapping of task names to file paths
        # by searching only within those subdirectories
        task_name_to_file = {}
        if task_subdirs is not None:
            task_files = self._discover_tasks_in_subdirs(task_subdirs, "*.py")
            for task_file in task_files:
                # Extract task class name from filename (e.g., banana_in_bowl_task.py -> BananaInBowlTask)
                task_name_to_file[task_file.stem] = str(task_file)
                # Also try to get the actual class name from the file
                try:
                    from robolab.core.task.task_utils import get_task_class_name_from_file
                    class_name = get_task_class_name_from_file(str(task_file))
                    task_name_to_file[class_name] = str(task_file)
                except Exception:
                    pass  # If we can't load the class name, just use file stem

        for task in tasks:
            task_name = Path(task).stem if ('/' in task or '\\' in task) else task

            # If task_subdirs was provided and we have a mapping, use the full path
            if task_subdirs is not None and task in task_name_to_file:
                task_to_resolve = task_name_to_file[task]
            else:
                task_to_resolve = task

            env_cfg_class = self.create_env_cfg(
                task_to_resolve,
                tags=tags,
                env_prefix=env_prefix,
                env_postfix=env_postfix,
                **env_kwargs
            )
            generated_envs[task_name] = env_cfg_class

        return generated_envs

    def auto_discover_and_create_cfgs(self,
                                pattern: str = "*_task.py",
                                add_tags: str | list[str] = "all",
                                task_subdirs: list[str] | None = None,
                                env_prefix: str = "",
                                env_postfix: str = "",
                                verbose_timing: bool = False,
                                **env_kwargs) -> dict[str, type]:
        """
        Enhanced to support folder-based tag organization.

        Args:
            pattern: Glob pattern to match task files
            add_tags: Tag name for discovered tasks, or list of tag names to file this under.
            task_subdirs: List of subdirectories to search (e.g., ["single_tasks", "composite_tasks"])
                         If None, searches the main task directory
            env_prefix: Prefix for environment names
            env_postfix: Postfix for environment names
            verbose_timing: If True, print timing information for each task registration
            **env_kwargs: Additional environment configuration parameters

            example args for env_kwargs:
                observations_cfg=ObservationCfg(),
                actions_cfg=DroidJointPositionActionCfg(),
                robot_cfg=DroidCfg,
                camera_cfg=OverShoulderLeftCameraCfg, # could be a list of camera configurations
                lighting_cfg=SphereLightCfg, # could be a list of lighting configurations
                background_cfg=HomeOfficeBackgroundCfg,
                contact_gripper=contact_gripper,
                dt=1 / (60 * 2),
                render_interval=8,
                decimation=8,
                seed=1,

        Returns:
            Dictionary mapping task names to generated environment classes
        """
        total_start = time.time()

        # Discover task files
        if not pattern:
            pattern = "*.py" # Force all task files to be python files.
        if task_subdirs is not None:
            task_files = self._discover_tasks_in_subdirs(task_subdirs, pattern)
        else:
            task_files = list(self.task_dir.glob(pattern))

        # Filter out files from the 'generated' folder
        task_files = [f for f in task_files if 'generated' not in f.parts]

        if verbose_timing:
            print(f"[EnvFactory] Discovered {len(task_files)} task files in {time.time() - total_start:.2f}s")

        generated_envs = {}

        for i, task_file in enumerate(task_files):
            task_name = task_file.stem

            if verbose_timing:
                task_start = time.time()

            env_cfg_class = self.create_env_cfg(
                str(task_file),
                tags=add_tags,
                env_prefix=env_prefix,
                env_postfix=env_postfix,
                **env_kwargs
            )
            generated_envs[task_name] = env_cfg_class

            if verbose_timing:
                print(f"[EnvFactory] ({i+1}/{len(task_files)}) Registered {task_name} in {time.time() - task_start:.3f}s")

        if verbose_timing:
            print(f"[EnvFactory] Total registration time: {time.time() - total_start:.2f}s for {len(task_files)} tasks")

        return generated_envs

    # ============================================================================
    # Environment Query Methods
    # ============================================================================

    def is_env(self, env_name: str) -> bool:
        """Check if an environment exists."""
        return env_name in self._env_info.keys()

    def get_env_info(self, env_name: str) -> dict[str, Any] | None:
        """Get structured information about a specific environment."""
        return self._env_info.get(env_name)

    def get_envs_by_filter(self,
                          task_name: str | None = None,
                          tag: str | None = None,
                          config_class_name: str | None = None) -> list[str]:
        """
        Get environment names filtered by various criteria.

        Args:
            task_name: Filter by task name (the base task class name)
            tag: Filter by tag name
            config_class_name: Filter by config class name

        Returns:
            List of environment names matching the criteria
        """
        matching_envs = []

        for env_name, info in self._env_info.items():
            # Check task name filter (base task class name)
            if task_name is not None and info['task_name'] != task_name:
                continue

            # Check tag filter
            if tag is not None and tag not in info['tags']:
                continue

            # Check config class name filter
            if config_class_name is not None and info['config_class_name'] != config_class_name:
                continue

            matching_envs.append(env_name)

        return matching_envs

    # ============================================================================
    # Task-based Query Methods
    # ============================================================================

    def get_envs_by_task(self, task_name: str) -> list[str]:
        """Get all environment names for a specific task."""
        return self.get_envs_by_filter(task_name=task_name)

    def get_env_cfgs_by_task(self, task_name: str) -> list[type]:
        """Get all environment configuration classes for a specific task."""
        env_names = self.get_envs_by_task(task_name)
        return [self._env_info[env_name]['config_class'] for env_name in env_names]

    # ============================================================================
    # Tag-based Query Methods
    # ============================================================================

    def get_envs_by_tag(self, tag_name: str) -> list[str]:
        """Get all environment names with a tag."""
        return self.get_envs_by_filter(tag=tag_name)

    def get_env_cfgs_by_tag(self, tag_name: str) -> dict[str, type]:
        """Get all environment configs for environments with a tag."""
        env_names = self.get_envs_by_tag(tag_name)
        return {env_name: self._env_info[env_name]['config_class'] for env_name in env_names}

    def get_tags_for_env(self, env_name: str) -> list[str]:
        """Get all tags for an environment."""
        env_info = self._env_info.get(env_name)
        if env_info is None:
            return []
        return list(env_info['tags'])

    # ============================================================================
    # Query Methods
    # ============================================================================

    def get_all_envs(self) -> list[str]:
        """Get all environment names."""
        return list(self._env_info.keys())

    def get_all_task_names(self) -> list[str]:
        """Get all unique task names (base task class names)."""
        return list(set(info['task_name'] for info in self._env_info.values()))

    # ============================================================================
    # Tag Management Methods
    # ============================================================================

    def register_env_to_tag(self, env_name: str, tag_name: str,
                            metadata: dict | None = None):
        """Register an environment to a specific tag."""
        # Update the environment info
        if env_name in self._env_info:
            self._env_info[env_name]['tags'].add(tag_name)

        # Update tag metadata
        if tag_name not in self._tag_metadata:
            self._tag_metadata[tag_name] = metadata or {}

    def register_tag(self, tag_name: str, env_names: list[str],
                     metadata: dict | None = None):
        """Register an entire tag of environments."""
        # Update environment info for each environment
        for env_name in env_names:
            if env_name in self._env_info:
                self._env_info[env_name]['tags'].add(tag_name)

        # Update tag metadata
        self._tag_metadata[tag_name] = metadata or {}

    # ============================================================================
    # Advanced Query Methods
    # ============================================================================

    def get_envs_by_tasks(self, task_names: list[str]) -> list[str]:
        """Get all environment names for multiple tasks."""
        env_names = []
        for task_name in task_names:
            env_names.extend(self.get_envs_by_task(task_name))
        return env_names

    def get_envs_by_tags(self, tag_names: list[str]) -> list[str]:
        """Get all environment names for multiple tags."""
        env_names = []
        for tag_name in tag_names:
            env_names.extend(self.get_envs_by_tag(tag_name))
        return list(set(env_names))  # Remove duplicates

    def get_envs_by_task_and_tag(self, task_name: str, tag_name: str) -> list[str]:
        """Get all environment names for a specific task and tag combination."""
        return self.get_envs_by_filter(task_name=task_name, tag=tag_name)

    def get_env_info_table_data(self,
                               task_name: str | None = None,
                               tag: str | None = None,
                               config_class_name: str | None = None) -> list[list[str]]:
        """
        Get table data for environments filtered by criteria.

        Args:
            task_name: Filter by task file name
            tag: Filter by tag name
            config_class_name: Filter by config class name

        Returns:
            List of table rows, each containing [task_name, env_name, config_class_name, tags]
        """
        filtered_envs = self.get_envs_by_filter(
            task_name=task_name,
            tag=tag,
            config_class_name=config_class_name
        )

        table_data = []
        for env_name in filtered_envs:
            info = self._env_info[env_name]
            tags_str = ", ".join(sorted(info['tags'])) if info['tags'] else "-"

            table_data.append([
                info['task_name'],
                env_name,
                info['config_class_name'],
                tags_str
            ])

        return table_data

    def list_tags(self) -> dict[str, dict]:
        """List all tags with their environments and metadata."""
        result = {}

        # Build tags from environment data
        tags = {}
        for env_name, env_info in self._env_info.items():
            for tag_name in env_info['tags']:
                if tag_name not in tags:
                    tags[tag_name] = set()
                tags[tag_name].add(env_name)

        for tag_name, env_names in tags.items():
            result[tag_name] = {
                'environments': list(env_names),
                'env_count': len(env_names),
                'metadata': self._tag_metadata.get(tag_name, {})
            }

        return result

    def print_env_table(self,
                       task_name: str | None = None,
                       tag: str | None = None,
                       config_class_name: str | None = None,
                       verbose: bool = False) -> None:
        """
        Print a formatted table of environments filtered by criteria.

        Args:
            task_name: Filter by task name
            tag: Filter by tag name
            config_class_name: Filter by config class name
            verbose: Whether to show detailed configuration info
        """
        table_data = self.get_env_info_table_data(
            task_name=task_name,
            tag=tag,
            config_class_name=config_class_name
        )

        if not table_data:
            print("No environments found matching the specified criteria.")
            return

        # Calculate column widths
        headers = ["Task Name", "Environment", "Config Class", "Tags"]
        col_widths = [len(header) for header in headers]

        for row in table_data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print table header
        header_row = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
        print(f"  {header_row}")
        print(f"  {'-' * len(header_row)}")

        # Print table rows
        for row in table_data:
            formatted_row = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
            print(f"  {formatted_row}")

            # Show detailed config if verbose
            if verbose:
                env_name = row[1]  # environment name column (index 1)
                env_info = self._env_info[env_name]
                print_env_cfg(env_info['config_class'](), prefix="    ")

        print()

    def _discover_tasks_in_subdirs(self, task_subdirs: list[str], pattern: str) -> list[Path]:
        """Discover task files in specified subdirectories."""
        task_files = []

        for subdir in task_subdirs:
            # Skip subfolders that start with '_' or are named 'generated'
            if subdir.startswith("_") or subdir == "generated":
                continue
            subdir_path = self.task_dir / subdir
            if subdir_path.exists() and subdir_path.is_dir():
                # Search recursively within the subdirectory
                discovered = list(subdir_path.rglob(pattern))
                # Filter out __init__.py and files from 'generated' subdirectories
                discovered = [f for f in discovered
                              if 'generated' not in f.parts
                              and f.name != '__init__.py']
                task_files.extend(discovered)

        return task_files


# ============================================================================
# Global Factory Instance
# ============================================================================

_global_factory = None

def get_global_env_factory(task_dir: str=TASK_DIR) -> EnvFactory:
    """
    Get or create the global auto environment factory.

    Args:
        task_dir: Directory containing task files

    Returns:
        The global AutoEnvFactory instance
    """
    global _global_factory
    if _global_factory is None:
        if task_dir is None:
            task_dir = TASK_DIR
        _global_factory = EnvFactory(task_dir)
    return _global_factory


# ============================================================================
# Convenience Functions for Auto Environment Creation.
# These functions are used to create environments automatically and register them.
#   1. create_env_cfg: Create an environment configuration from a task name or file path.
#   2. batch_create_env_cfgs: Create multiple environments from a list of task names or file paths.
#   3. auto_discover_and_create_cfgs: Auto-discover task files and create environment configurations.
# ============================================================================

def create_env_cfg(task: str, task_dir=None, **kwargs) -> type:
    """
    Create an environment configuration from a task name or file path.

    This function auto-detects the task identifier type and resolves it:
    1. Full file path (contains '/' or '\\') - used directly
    2. Filename ending in '.py' - looked up in task_dir
    3. Task name - searched recursively in task_dir for matching .py file

    Args:
        task: Task identifier, can be:
            - Full path: "/path/to/BananaTask.py"
            - Filename: "BananaTask.py"
            - Task name: "BananaTask"
        task_dir: Directory containing task files (uses default if None)
        tags: Tag name(s) to register the task to
        env_name: Name for the environment (if registering)
        env_prefix: Prefix for environment name
        env_postfix: Postfix for environment name
        register: Whether to register the environment
        **kwargs: Additional environment configuration parameters

        Example kwargs:
            observations_cfg=ObservationCfg(),
            actions_cfg=DroidJointPositionActionCfg(),
            robot_cfg=DroidCfg,
            camera_cfg=OverShoulderLeftCameraCfg,  # could be a list of camera configurations
            lighting_cfg=SphereLightCfg,  # could be a list of lighting configurations
            background_cfg=HomeOfficeBackgroundCfg,
            contact_gripper=contact_gripper,
            dt=1 / (60 * 2),
            render_interval=8,
            decimation=8,
            seed=1,

    Returns:
        The generated environment configuration class

    Examples:
        # Using task name (looks in task_dir)
        create_env_cfg("BananaTask")

        # Using full file path
        create_env_cfg("/path/to/tasks/BananaTask.py")

        # With tags and postfix
        create_env_cfg("BananaTask", tags="my_tag", env_postfix="v2")

        # With full configuration
        create_env_cfg(
            "BananaTask",
            tags="background_variations",
            env_postfix="bg_warehouse",
            observations_cfg=ObservationCfg(),
            actions_cfg=DroidJointPositionActionCfg(),
            robot_cfg=DroidCfg,
            camera_cfg=[OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg],
            background_cfg=WarehouseBackgroundCfg,
        )
    """
    factory = get_global_env_factory(task_dir)
    return factory.create_env_cfg(task, **kwargs)


def batch_create_env_cfgs(tasks: list[str], task_dir=None, **kwargs) -> dict[str, type]:
    """
    Create multiple environments from a list of task names or file paths.

    Args:
        tasks: List of task names or file paths to create environments for
        task_dir: Directory containing task files (uses default if None)
        tags: Tag name(s) to register the tasks to
        task_subdirs: List of subdirectories to search (e.g., ["pick_place", "stacking_reorient"])
                     If None, searches the entire task directory recursively
        env_prefix: Prefix for environment names
        env_postfix: Postfix for environment names
        register: Whether to register the environments
        **kwargs: Additional environment configuration parameters

        Example kwargs:
            observations_cfg=ObservationCfg(),
            actions_cfg=DroidJointPositionActionCfg(),
            robot_cfg=DroidCfg,
            camera_cfg=OverShoulderLeftCameraCfg,  # could be a list of camera configurations
            lighting_cfg=SphereLightCfg,  # could be a list of lighting configurations
            background_cfg=HomeOfficeBackgroundCfg,
            contact_gripper=contact_gripper,
            dt=1 / (60 * 2),
            render_interval=8,
            decimation=8,
            seed=1,

    Returns:
        Dictionary mapping task names to generated environment classes

    Examples:
        # Create environments for multiple tasks
        batch_create_env_cfgs(
            ["BananaTask", "AppleTask", "OrangeTask"],
            tags="fruit_tasks",
        )

        # With task_subdirs to limit search scope
        batch_create_env_cfgs(
            ["BananaTask", "AppleTask"],
            tags="fruit_tasks",
            task_subdirs=["pick_place", "fruits"],
        )

        # With full configuration
        batch_create_env_cfgs(
            ["BananaTask", "AppleTask"],
            tags="evaluation",
            observations_cfg=ObservationCfg(),
            robot_cfg=DroidCfg,
            camera_cfg=[OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg],
        )
    """
    factory = get_global_env_factory(task_dir)
    return factory.batch_create_env_cfgs(tasks, **kwargs)


def auto_discover_and_create_cfgs(task_dir=None, **kwargs) -> dict[str, type]:
    """
    Auto-discover task files and create environment configurations.

    Args:
        task_dir: Directory containing task files (uses default if None)
        pattern: Glob pattern to match task files (default: "*_task.py")
        add_tags: Tag name for discovered tasks, or list of tag names
        task_subdirs: List of subdirectories to search (e.g., ["single_tasks", "composite_tasks"])
                     If None, searches the main task directory
        env_prefix: Prefix for environment names
        env_postfix: Postfix for environment names
        verbose_timing: If True, print timing information for each task registration (default: False)
        **kwargs: Additional environment configuration parameters

        Example kwargs:
            observations_cfg=ObservationCfg(),
            actions_cfg=DroidJointPositionActionCfg(),
            robot_cfg=DroidCfg,
            camera_cfg=OverShoulderLeftCameraCfg,  # could be a list of camera configurations
            lighting_cfg=SphereLightCfg,  # could be a list of lighting configurations
            background_cfg=HomeOfficeBackgroundCfg,
            contact_gripper=contact_gripper,
            dt=1 / (60 * 2),
            render_interval=8,
            decimation=8,
            seed=1,

    Returns:
        Dictionary mapping task names to generated environment classes
    """
    factory = get_global_env_factory(task_dir)
    return factory.auto_discover_and_create_cfgs(**kwargs)


# ============================================================================
# Convenience Query Functions
# ============================================================================

def get_all_envs() -> list[str]:
    """Get all environment names."""
    factory = get_global_env_factory()
    return factory.get_all_envs()

# ============================================================================
# Query Convenience Functions
# ============================================================================

def get_envs_by_task(task_name: str) -> list[str]:
    """Get all environment names for a specific task.

    Args:
        task_name: The base task class name (e.g., "BananaInBowlTask")

    Returns:
        List of environment names for this task
    """
    factory = get_global_env_factory()
    return factory.get_envs_by_filter(task_name=task_name)


def get_envs_by_tag(tag_name: str) -> list[str]:
    """Get all environment names with a tag.

    Args:
        tag_name: The tag name (e.g., "pick_place", "all")

    Returns:
        List of environment names with this tag
    """
    factory = get_global_env_factory()
    return factory.get_envs_by_filter(tag=tag_name)


def get_envs(task: str | list[str] | None = None,
             env: str | None = None,
             tag: str | list[str] | None = None) -> list[str]:
    """Get environment names filtered by task, env, or tag.

    Only one filter argument should be provided at a time.

    Args:
        task: Task name (base task class name) or list of task names - returns all environment variants
        env: Exact environment name - returns a single-element list with this environment
        tag: Tag name or list of tag names - returns all environments with the tag(s)

    Returns:
        List of environment names matching the filter criteria

    Raises:
        ValueError: If multiple filter arguments are provided, or if the filter value is not found

    Examples:
        # Get all environments
        envs = get_envs()

        # Get all variants of a task
        envs = get_envs(task="BananaInBowlTask")

        # Get variants of multiple tasks
        envs = get_envs(task=["BananaInBowlTask", "AppleInBowlTask"])

        # Get a specific environment
        envs = get_envs(env="BananaInBowlTaskHomeOffice")

        # Get all environments with a tag
        envs = get_envs(tag="pick_place")
    """
    # Check that only one argument is provided
    provided_args = sum(arg is not None for arg in [task, env, tag])
    if provided_args > 1:
        raise ValueError("Only one of 'task', 'env', or 'tag' can be specified at a time.")
    if provided_args == 0:
        return get_all_envs()

    factory = get_global_env_factory()

    if env is not None:
        # Exact environment name lookup
        if not factory.is_env(env):
            available_envs = factory.get_all_envs()
            envs_list = "\n  ".join(sorted(available_envs))
            raise ValueError(f"Environment '{env}' not found.\n\nAvailable environments ({len(available_envs)}):\n  {envs_list}")
        return [env]

    if task is not None:
        # Filter by task name(s)
        if isinstance(task, str):
            task = [task]

        matching_envs = []
        for t in task:
            envs = get_envs_by_task(t)
            if not envs:
                available_task_names = factory.get_all_task_names()
                tasks_list = "\n  ".join(sorted(available_task_names))
                raise ValueError(f"Task '{t}' not found.\n\nAvailable task names ({len(available_task_names)}):\n  {tasks_list}")
            matching_envs.extend(envs)
        return matching_envs

    if tag is not None:
        # Filter by tag(s)
        if isinstance(tag, str):
            tag = [tag]

        matching_envs = []
        for t in tag:
            envs = get_envs_by_tag(t)
            if not envs:
                available_tags = list(factory.list_tags().keys())
                tags_list = "\n  ".join(sorted(available_tags))
                raise ValueError(f"Tag '{t}' not found or empty.\n\nAvailable tags ({len(available_tags)}):\n  {tags_list}")
            matching_envs.extend(envs)
        return matching_envs

    return []

# Display convenience functions
def print_env_table(task_name: str | None = None, tag: str | None = None,
                   config_class_name: str | None = None, verbose: bool = False) -> None:
    """Quick function to print environment table with filters."""
    factory = get_global_env_factory()
    factory.print_env_table(task_name=task_name, tag=tag,
                           config_class_name=config_class_name, verbose=verbose)
