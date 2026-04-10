# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Environment runtime utilities.

This module provides functions for creating and managing environment instances
at runtime, including environment creation, episode management, and
termination checking.
"""

import json
import os

import carb
import gymnasium as gym
import numpy as np
import omni.usd
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv

import robolab.constants
from robolab.constants import get_output_dir
from robolab.core.environments.config import parse_env_cfg
from robolab.core.events.utils import merge_events_cfg
from robolab.core.task.task import resolve_instruction


def check_scene_valid(env: ManagerBasedEnv) -> bool:
    """
    Checks the scene has all the required fields for RoboLab.
    """
    if 'robot' not in env.scene.articulations.keys():
        raise ValueError("Scene entity 'robot' not found; Available articulations: " + str(list(env.scene.articulations.keys())))

    return True


def create_env(scene: str | ManagerBasedEnvCfg,
               device="cuda:0",
               seed=0,
               num_envs=1,
               env_spacing=None,
               eye=None,
               lookat=None,
               use_fabric=True,
               events=None,
               instruction_type="default",
               policy=None,
    ):
    """
    Creates and initializes a gym environment for the specified scene. Supported types: str, ManagerBasedEnvCfg.
    Example use:
        env_cfg = BananaEnvCfg()
        env = create_env(scene=env_cfg, device="cuda:0", num_envs=1, use_fabric=True)

        env=create_env(scene="BananaEnv", device="cuda:0", num_envs=1, use_fabric=True)

        # With a variation event (example: camera pose variation)
        from robolab.core.events.reset_camera import reset_camera_pose_uniform
        from isaaclab.managers import EventTermCfg as EventTerm
        env, env_cfg = create_env(
            scene="BananaEnv",
            events={
                "reset_camera": EventTerm(
                    func=reset_camera_pose_uniform,
                    mode="reset",
                    params={
                        "camera_names": ["external_cam"],
                        "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    }
                )
            }
        )

        # Or using the configclass helper
        from robolab.core.events.reset_camera import RandomizeCameraPoseUniform
        events = RandomizeCameraPoseUniform.from_params(
            cameras=["external_cam"],
            pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
        )
        env, env_cfg = create_env(scene="BananaEnv", events=events)

    Args:
        scene (str, ManagerBasedEnvCfg): The scene name, or environment configuration
        device: The device to run the environment on (e.g., 'cuda', 'cpu')
        num_envs (int, optional): Number of environments to spawn. Defaults to 1.
        use_fabric (bool, optional): Whether to use fabric for physics simulation. Defaults to True.
        events: Event configuration to add to the environment. Can be:
            - A dict of {event_name: EventTerm} (automatically converted to configclass)
            - A configclass instance with EventTerm attributes
        instruction_type: Which instruction variant to use when the task defines
            instruction as a dict (e.g., "default", "vague", "specific"). Ignored
            when instruction is a plain string. Defaults to "default".
        policy: Policy backend name (e.g., "pi0", "gr00t"). Stored on env_cfg
            so downstream code (e.g., run_episode) can read it.

    Raises:
        ValueError: If the scene type is not supported or environment creation fails

    Returns:
        tuple: (env, env_cfg) - The created environment instance and its configuration
    """
    env = None

    if isinstance(scene, str):
        # create a new stage
        omni.usd.get_context().new_stage()
        # reset the rtx sensors carb setting to False
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)

        try:
            # Initialize the env for current scene
            env_cfg = parse_env_cfg(
                scene,
                device=device,
                seed=seed,
                num_envs=num_envs,
                env_spacing=env_spacing,
                use_fabric=use_fabric,
                eye=eye,
                lookat=lookat,
            )

            env_cfg._instruction_variants = env_cfg.instruction
            env_cfg.instruction = resolve_instruction(env_cfg.instruction, instruction_type)

            # Merge events into the environment configuration if provided
            # This preserves existing events (like reset_scene_to_default) while adding new ones
            if events is not None:
                env_cfg.events = merge_events_cfg(env_cfg.events, events)
                if robolab.constants.VERBOSE:
                    print(f"Merged events into environment configuration: {env_cfg.events}")

            # Create new environment
            env = gym.make(scene, cfg=env_cfg).unwrapped
        except Exception as e:
            if "env" in locals() and hasattr(env, "_is_closed"):
                env.close()
            else:
                if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                    e.obj.close()
                else:
                    raise ValueError(f"Failed to create environment for scene {scene}; exception: {e}")

    elif isinstance(scene, ManagerBasedEnvCfg):
        # create a new stage
        omni.usd.get_context().new_stage()
        # reset the rtx sensors carb setting to False
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)
        env_cfg = scene

        env_cfg._instruction_variants = env_cfg.instruction
        env_cfg.instruction = resolve_instruction(env_cfg.instruction, instruction_type)

        # Merge events into the environment configuration if provided
        # This preserves existing events (like reset_scene_to_default) while adding new ones
        if events is not None:
            env_cfg.events = merge_events_cfg(env_cfg.events, events)
            if robolab.constants.VERBOSE:
                print(f"Merged events into environment configuration: {env_cfg.events}")

        env = ManagerBasedRLEnv(env_cfg)
    else:
        raise ValueError(f"Unsupported scene type: {type(scene)}")

    if env is None:
        raise ValueError(f"Failed to create environment for scene {scene}")

    check_scene_valid(env)

    # disable control on stop
    env.sim._app_control_on_stop_handle = None  # type: ignore

    env.output_dir = get_output_dir()
    os.makedirs(env.output_dir, exist_ok=True)

    if policy is not None:
        env_cfg.policy = policy

    from robolab.core.utils.print_utils import print_env_info
    env_name = scene if isinstance(scene, str) else env_cfg.__class__.__name__
    print_env_info(
        env_name=env_name,
        instruction=env_cfg.instruction,
        instruction_type=instruction_type,
        seed=env_cfg.seed,
        policy=policy or "",
        scene_name=env_cfg.scene.__class__.__name__,
        attributes=getattr(env_cfg, '_task_attributes', None),
    )

    # Save env_cfg as json for metadata
    with open(os.path.join(env.output_dir, "env_cfg.json"), "w") as f:
        json.dump(env_cfg.to_dict(), f, default=str)
        if robolab.constants.VERBOSE:
            print(f"Saved env_cfg to {os.path.join(env.output_dir, 'env_cfg.json')}")

    return env, env_cfg


def end_episode(env: ManagerBasedRLEnv):
    from robolab.core.logging.recorder_manager import RobolabRecorderManager
    # Clean up env for the next episode
    if env.recorder_manager is not None:
        if isinstance(env.recorder_manager, RobolabRecorderManager)and env.recorder_manager.initialized:
            if robolab.constants.VERBOSE:
                print("Exporting data....")
            env.recorder_manager.export_episodes()
            if robolab.constants.VERBOSE:
                print("Episodes exported. ")
            env.recorder_manager.clear()


def check_terminated(env: ManagerBasedRLEnv, term, trunc) -> np.ndarray:
    """
    Check termination status for each environment.

    Usage:
        succ_vec = check_terminated(env, term, trunc)
        if any(succ_vec, None):
            # still running
            continue
        else:
            return succ_vec

    Args:
        env: The environment instance
        term: Termination tensor of shape [N]
        trunc: Truncation tensor of shape [N]

    Returns:
        Numpy array of shape [N] stored on CPU where:
        - True: episode terminated (term=True)
        - False: episode truncated (trunc=True and term=False)
        - None: episode still running (both term=False and trunc=False)
    """
    import torch

    # Convert to boolean tensors and move to CPU
    term_bool = term.bool().cpu()
    trunc_bool = trunc.bool().cpu()

    # Create numpy array with object dtype to support None values
    result = torch.zeros_like(term, dtype=torch.bool).cpu().numpy().astype(object)

    # Set True where term is True
    result[term_bool.numpy()] = True

    # Set False where trunc is True and term is False
    result[(trunc_bool & ~term_bool).numpy()] = False

    # Set None where neither term nor trunc is True (still running)
    still_running = ~(term_bool | trunc_bool)
    result[still_running.numpy()] = None

    return result
