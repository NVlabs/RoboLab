# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Environment configuration generation and parsing utilities.

This module provides functions for:
- Generating scene environment configurations from tasks
- Generating complete task environment configurations
- Auto-generating environments from task files
- Registering environments with gymnasium
- Parsing environment configurations from the registry
"""

from typing import Any, Type

import gymnasium as gym
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import load_cfg_from_registry

from robolab.core.environments.base import RobolabDefaultEnvCfg
from robolab.core.sensors.contact_sensor_utils import create_contact_sensors
from robolab.core.task.task import Task
from robolab.core.task.task_utils import load_task_from_file

# ============================================================================
# Scene Environment Configuration Generation
# ============================================================================

def generate_scene_env_cfg(task_class: Task,
                          robot_cfg,
                          camera_cfg=None,
                          lighting_cfg=None,
                          background_cfg=None) -> InteractiveSceneCfg:
    """
    Generate a scene environment configuration class from a task.

    Args:
        task_class: The Task class
        robot_cfg: Robot configuration class to include
        camera_cfg: Camera configuration class to include, could be a list of configurations
        lighting_cfg: Lighting configuration class to include, could be a list of configurations

    Returns:
        A configclass that combines the task scene with robot, camera, and lighting
    """
    bases = [task_class.scene, robot_cfg, InteractiveSceneCfg]

    # Add optionals only if not None
    if camera_cfg is not None:
        if isinstance(camera_cfg, list):
            for cam_cfg in camera_cfg:
                bases.append(cam_cfg)
        else:
            bases.append(camera_cfg)
    if lighting_cfg is not None:
        if isinstance(lighting_cfg, list):
            for light_cfg in lighting_cfg:
                bases.append(light_cfg)
        else:
            bases.append(lighting_cfg)
    if background_cfg is not None:
        bases.append(background_cfg)

    # Dynamically create the class with a meaningful name
    class_name = f"{task_class.__name__}SceneEnvCfg"
    cfg_cls = type(class_name, tuple(bases), {})

    # Apply the configclass decorator
    return configclass(cfg_cls)


# ============================================================================
# Task Environment Configuration Generation
# ============================================================================

def generate_task_env_cfg(task_class: Task,
                         scene_env_cfg: Type,
                         observations_cfg: Any,
                         actions_cfg: Any,
                         contact_gripper: str,
                         dt: int,
                         render_interval: int,
                         decimation: int,
                         seed: int = 0,
                         num_envs: int = 1,
                         eye: tuple[float, float, float] = (1.5, 0.0, 1.0),
                         lookat: tuple[float, float, float] = (0.2, 0.0, 0.0),
                         env_spacing: float = 5.0) -> Type[RobolabDefaultEnvCfg]:
    """
    Generate a complete task environment configuration class.

    Args:
        task_class: The Task class
        scene_env_cfg: The scene environment configuration class
        observations_cfg: Observations configuration (defaults to standard Droid observations)
        actions_cfg: Actions configuration (defaults to DroidJointPositionActionCfg)
        subtasks: A list of subtasks to be completed
        episode_length_s: Episode length in seconds
        decimation: Decimation factor
        seed: Random seed
        num_envs: Number of environments
        env_spacing: Environment spacing

    Returns:
        A complete environment configuration class
    """
    from robolab.core.task.subtask_utils import compute_difficulty_score, count_subtasks
    attributes = getattr(task_class, "attributes", []) or []
    subtasks_raw = getattr(task_class, "subtasks", None)
    num_subtasks = count_subtasks(subtasks_raw)
    _, difficulty_label = compute_difficulty_score(num_subtasks, attributes)
    attributes = list(attributes) + [difficulty_label]

    @configclass
    class GeneratedTaskEnvCfg(RobolabDefaultEnvCfg):
        observations = observations_cfg
        actions = actions_cfg
        subtasks = task_class.subtasks
        task_attributes = attributes


        def __post_init__(self):
            super().__post_init__()  # Set all defaults first

            self.episode_length_s: int = task_class.episode_length_s
            self.decimation: int = decimation
            self.sim.dt: int = dt
            self.sim.render_interval: int = render_interval

            # Can be overwritten during parse_env_cfg
            self.seed: int = seed
            self.num_envs: int = num_envs
            self.env_spacing: float = env_spacing
            self.viewer.eye: tuple[float, float, float] = eye
            self.viewer.lookat: tuple[float, float, float] = lookat

            # Set task-specific configs
            self.scene = scene_env_cfg(num_envs=num_envs, env_spacing=env_spacing)
            self.contact_gripper = contact_gripper
            self.instruction = task_class.instruction
            self.terminations = task_class.terminations()
            self.contact_object_list = task_class.contact_object_list

            # Set optional rewards if provided by the task
            if getattr(task_class, 'rewards', None) is not None:
                self.rewards = task_class.rewards()

            # Set optional events if provided by the task
            if getattr(task_class, 'events', None) is not None:
                self.events = task_class.events()

            # Must specify this after the scene is set.
            create_contact_sensors(self)

    # Set a meaningful name for the generated class
    GeneratedTaskEnvCfg.__name__ = f"{task_class.__name__}EnvCfg"

    # Store task_attributes at class level (since @configclass removes it as a direct class attribute)
    GeneratedTaskEnvCfg._task_attributes = attributes
    GeneratedTaskEnvCfg._task_name = getattr(task_class, "task_name", None) or task_class.__name__  # If task_name is not provided, use the task class name.

    return GeneratedTaskEnvCfg


# ============================================================================
# Auto Generation from Task Files
# ============================================================================

def auto_generate_task_env(task_file_path: str,
                          robot_cfg,
                          camera_cfg = None,
                          lighting_cfg=None,
                          background_cfg=None,
                          observations_cfg = None,
                          actions_cfg = None,
                          **env_kwargs) -> Type[RobolabDefaultEnvCfg]:
    """
    Automatically generate a complete task environment configuration from a task file.

    Args:
        task_file_path: Path to the task file (e.g., 'sauce_bottles_crate.py')
        robot_cfg: Robot configuration class to include
        camera_cfg: Camera configuration class to include, or multiple
        lighting_cfg: Lighting configuration class to include
        observations_cfg: Observations configuration
        actions_cfg: Actions configuration
        **env_kwargs: Additional environment configuration parameters

    Returns:
        A complete environment configuration class
    """
    # Load the task class from the file
    task_class = load_task_from_file(task_file_path)

    # Generate the scene environment configuration
    scene_env_cfg = generate_scene_env_cfg(
        task_class, robot_cfg, camera_cfg, lighting_cfg, background_cfg
    )

    # Generate the complete task environment configuration
    task_env_cfg = generate_task_env_cfg(
        task_class, scene_env_cfg, observations_cfg, actions_cfg, **env_kwargs
    )

    return task_env_cfg


# ============================================================================
# Environment Registration
# ============================================================================

def register_generated_env(task_env_cfg: RobolabDefaultEnvCfg, env_name: str = None):
    """
    Register a generated environment configuration with gymnasium.

    Args:
        task_env_cfg: The generated environment configuration class
        env_name: Name for the environment (defaults to class name)
    """
    if env_name is None:
        env_name = task_env_cfg.__name__.replace('EnvCfg', '')

    gym.register(
        id=env_name,
        entry_point="robolab.core.environments.env:RobolabEnv",
        kwargs={
            "env_cfg_entry_point": task_env_cfg,
        },
        disable_env_checker=True,
    )

    return env_name


# ============================================================================
# High-Level Environment Configuration Creation
# ============================================================================

def generate_env_cfg_from_task(task_file_path: str,
                    env_name: str = None,
                    env_prefix: str="",
                    env_postfix: str="",
                    register: bool = True,
                    **kwargs) -> tuple[Type[RobolabDefaultEnvCfg], str]:
    """
    Create and optionally register a task environment from a task file.
    Basic function for creating an Environment from Task.


    Args:
        task_file_path: Path to the task file
        env_name: Name for the environment (if registering)
        register: Whether to register the environment with gymnasium
        **kwargs: Additional arguments for auto_generate_task_env

    Returns:
        The generated environment configuration class
    """
    task_env_cfg = auto_generate_task_env(task_file_path, **kwargs)

    if env_name is None:
        env_name = task_env_cfg.__name__.replace('EnvCfg', '')
    env_name = env_prefix+env_name+env_postfix

    # Update the class name to reflect the final environment name
    new_class_name = f"{env_name}EnvCfg"
    task_env_cfg.__name__ = new_class_name

    if register:
        env_name = register_generated_env(task_env_cfg, env_name)

    return task_env_cfg, env_name


# ============================================================================
# Environment Configuration Parsing
# ============================================================================

def parse_env_cfg(
    task_name: str,
    device: str = "cuda:0",
    seed: int = None,
    num_envs: int | None = None,
    env_spacing: float = 5.0,
    eye: tuple[float, float, float] = (1.5, 0.0, 1.0),
    lookat: tuple[float, float, float] = (0.2, 0.0, 0.0),
    use_fabric: bool | None = None,
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    """Parse configuration for an environment and override based on inputs.
    Adapted from isaaclab_tasks.utils.parse_env_cfg to allow overriding num_envs, seed, eye, lookat, and env_spacing.

    Args:
        task_name: The name of the environment.
        device: The device to run the simulation on. Defaults to "cuda:0".
        num_envs: Number of environments to create. Defaults to None, in which case it is left unchanged.
        env_spacing: Spacing between environments. Defaults to 7.0.
        eye: Eye position for the viewer. Defaults to (1.5, 0.0, 1.0).
        lookat: Lookat position for the viewer. Defaults to (0.2, 0.0, 0.0).
        seed: Seed for the random number generator. Defaults to None, in which case it is left unchanged.
        use_fabric: Whether to enable/disable fabric interface. If false, all read/write operations go through USD.
            This slows down the simulation but allows seeing the changes in the USD through the USD stage.
            Defaults to None, in which case it is left unchanged.

    Returns:
        The parsed configuration object.

    Raises:
        RuntimeError: If the configuration for the task is not a class. We assume users always use a class for the
            environment configuration.
    """
    # load the default configuration
    cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

    # check that it is not a dict
    # we assume users always use a class for the configuration
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for the task: '{task_name}' is not a class. Please provide a class.")

    # simulation device
    cfg.sim.device = device

    if seed is not None:
        cfg.seed = seed
    if eye is not None:
        cfg.viewer.eye = eye
    if lookat is not None:
        cfg.viewer.lookat = lookat
    if env_spacing is not None:
        cfg.env_spacing = env_spacing

    # disable fabric to read/write through USD
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    # number of environments
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
        cfg.num_envs = num_envs

    return cfg


# ============================================================================
# Utility Functions
# ============================================================================

def print_env_cfg(env_cfg: RobolabDefaultEnvCfg, prefix=""):
    """Print environment configuration details."""
    print(f"{prefix}instruction: {env_cfg.instruction}")
    print(f"{prefix}scene: {env_cfg.scene.__class__.__name__}")
    print(f"{prefix}observations: {env_cfg.observations.__class__.__name__}")
    print(f"{prefix}actions: {env_cfg.actions.__class__.__name__}")
    print(f"{prefix}terminations: {env_cfg.terminations.__class__.__name__}")
    print(f"{prefix}contact enabled for:")
    print(f"{prefix}  gripper: {env_cfg.contact_gripper}")
    print(f"{prefix}  objects: {env_cfg.contact_object_list}")
    print(f"{prefix}sim:")
    print(f"{prefix}  episode_length_s: {env_cfg.episode_length_s}")
    print(f"{prefix}  decimation: {env_cfg.decimation}")
    print(f"{prefix}  dt: {env_cfg.sim.dt}")
    print(f"{prefix}  render_interval: {env_cfg.sim.render_interval}")
    print(f"{prefix}  seed: {env_cfg.seed}")
    print(f"{prefix}  num_envs: {env_cfg.num_envs}")
    print(f"{prefix}  env_spacing: {env_cfg.env_spacing}")
