# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""RoboLab environment module.

This module provides the core environment infrastructure for RoboLab:

- base: Base configuration classes (RobolabDefaultEnvCfg, etc.)
- config: Environment configuration generation and parsing
- env: RobolabEnv runtime class
- factory: EnvFactory for automatic environment creation
- runtime: Environment creation and management utilities
"""

# Base configuration classes
from robolab.core.environments.base import (
    ActionCfg,
    BaseEventCfg,
    BaseRecorderManagerCfg,
    CommandsCfg,
    CurriculumCfg,
    ObservationCfg,
    RewardsCfg,
    RobolabDefaultEnvCfg,
    TerminationsCfg,
    create_recorder_config,
)

# Configuration generation and parsing
from robolab.core.environments.config import (
    auto_generate_task_env,
    generate_env_cfg_from_task,
    generate_scene_env_cfg,
    generate_task_env_cfg,
    parse_env_cfg,
    print_env_cfg,
    register_generated_env,
)

# Runtime environment class
from robolab.core.environments.env import RobolabEnv

# Factory and convenience functions
from robolab.core.environments.factory import (
    EnvFactory,
    auto_discover_and_create_cfgs,
    batch_create_env_cfgs,
    get_all_envs,
    get_envs,
    get_envs_by_tag,
    get_envs_by_task,
    get_global_env_factory,
    print_env_table,
)

# Runtime utilities
from robolab.core.environments.runtime import (
    check_scene_valid,
    check_terminated,
    create_env,
    end_episode,
)

__all__ = [
    # Base configs
    "RobolabDefaultEnvCfg",
    "ObservationCfg",
    "ActionCfg",
    "BaseEventCfg",
    "BaseRecorderManagerCfg",
    "CommandsCfg",
    "RewardsCfg",
    "TerminationsCfg",
    "CurriculumCfg",
    "create_recorder_config",
    # Runtime class
    "RobolabEnv",
    # Config generation
    "generate_scene_env_cfg",
    "generate_task_env_cfg",
    "auto_generate_task_env",
    "register_generated_env",
    "generate_env_cfg_from_task",
    "parse_env_cfg",
    "print_env_cfg",
    # Runtime utilities
    "create_env",
    "end_episode",
    "check_terminated",
    "check_scene_valid",
    # Factory
    "EnvFactory",
    "get_global_env_factory",
    "batch_create_env_cfgs",
    "auto_discover_and_create_cfgs",
    "get_all_envs",
    "get_envs_by_task",
    "get_envs_by_tag",
    "get_envs",
    "print_env_table",
]
