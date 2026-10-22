# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment runtime utilities.

This module provides functions for creating and managing environment instances
at runtime, including environment creation, episode management, and
termination checking.
"""

import json
import logging
import os

import carb
import gymnasium as gym
import numpy as np
import omni.usd
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv

import robolab.constants
from robolab.constants import get_output_dir
from robolab.core.environments.config import parse_env_cfg
from robolab.core.environments.env import RobolabEnv
from robolab.core.events.utils import merge_events_cfg
from robolab.core.task.instruction import primary_instruction, select_instruction_fields
from robolab.core.utils.isaaclab_compat import env_cfg_to_recording_dict, prepare_env_cfg

logger = logging.getLogger(__name__)

# Maps the user-facing --renderer choice to the Omniverse "/rtx/rendermode"
# carb value. "realtime" is IsaacLab's default; we leave it unset so behavior
# is unchanged and only override the carb setting for the path tracer.
_RENDERER_TO_RTX_MODE = {"realtime": "RaytracedLighting", "pathtracing": "PathTracing"}


def _apply_render_settings(env_cfg: ManagerBasedEnvCfg, renderer: str, rendering_mode: str | None) -> None:
    """Apply renderer mode and quality preset onto ``env_cfg.sim.render``.

    Must be called before the simulation context is created (i.e. before
    ``gym.make``/``RobolabEnv``), since :class:`isaaclab.sim.RenderCfg` is read
    when the RTX renderer is configured.

    Args:
        env_cfg: The environment config whose ``sim.render`` (a ``RenderCfg``) is mutated.
        renderer: ``"realtime"`` (RaytracedLighting, the default — applied as a no-op
            to preserve IsaacLab behavior) or ``"pathtracing"`` (PathTracing, which
            overrides the ``/rtx/rendermode`` carb setting).
        rendering_mode: Realtime quality preset (``"performance"``/``"balanced"``/
            ``"quality"``) or ``None`` to leave IsaacLab's default (``"balanced"``).
    """
    if renderer not in _RENDERER_TO_RTX_MODE:
        raise ValueError(f"Unknown renderer '{renderer}'; expected one of {list(_RENDERER_TO_RTX_MODE)}.")

    if rendering_mode is not None:
        env_cfg.sim.render.rendering_mode = rendering_mode

    # Record the renderer carb setting for both modes so env_cfg.json positively
    # logs which RTX renderer the run used. For "realtime" this matches IsaacLab's
    # default (its WAR already enforces "RaytracedLighting"), so it is behavior-
    # preserving; "pathtracing" is a genuine override.
    carb_settings = dict(env_cfg.sim.render.carb_settings or {})
    carb_settings["/rtx/rendermode"] = _RENDERER_TO_RTX_MODE[renderer]
    env_cfg.sim.render.carb_settings = carb_settings


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
               instruction_type=None,
               policy=None,
               renderer="realtime",
               rendering_mode=None,
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
                        "camera_names": ["over_shoulder_left_camera"],
                        "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    }
                )
            }
        )

        # Or using the configclass helper
        from robolab.core.events.reset_camera import RandomizeCameraPoseUniform
        events = RandomizeCameraPoseUniform.from_params(
            cameras=["over_shoulder_left_camera"],
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
        instruction_type: Optional instruction field name or names to expose to
            the policy (e.g., ``"vague"`` or ``["vague", "video"]``). When
            omitted, all fields are exposed. Plain-string tasks are normalized
            to a sole ``default`` field.
        policy: Policy backend name (e.g., "pi0", "gr00t"). Stored on env_cfg
            so downstream code (e.g., run_episode) can read it.
        renderer: RTX renderer mode, "realtime" (RaytracedLighting, default) or
            "pathtracing" (PathTracing). Default preserves IsaacLab behavior.
        rendering_mode: Realtime quality preset ("performance"/"balanced"/"quality")
            or None to leave IsaacLab's default ("balanced").

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
            env_cfg.instruction = select_instruction_fields(env_cfg.instruction, instruction_type)

            # Configure the RTX renderer before the sim context is created.
            _apply_render_settings(env_cfg, renderer, rendering_mode)

            # Merge events into the environment configuration if provided
            # This preserves existing events (like reset_scene_to_default) while adding new ones
            if events is not None:
                env_cfg.events = merge_events_cfg(env_cfg.events, events)
                if robolab.constants.VERBOSE:
                    print(f"Merged events into environment configuration: {env_cfg.events}")

            # RoboLab authors and records WXYZ quaternions. Isaac Lab 3 uses
            # XYZW internally, so translate config rotations at the boundary.
            prepare_env_cfg(env_cfg)

            # Create new environment
            env = gym.make(scene, cfg=env_cfg).unwrapped
        except Exception:
            # Best-effort cleanup of partially-constructed env; always re-raise
            # so the caller sees the original traceback (don't wrap in
            # ValueError — that hides the root cause).
            if env is not None and hasattr(env, "_is_closed") and not env._is_closed:
                try:
                    env.close()
                except Exception:
                    logger.exception("env.close() failed during error cleanup")
            raise

    elif isinstance(scene, ManagerBasedEnvCfg):
        # create a new stage
        omni.usd.get_context().new_stage()
        # reset the rtx sensors carb setting to False
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)
        env_cfg = scene

        env_cfg._instruction_variants = env_cfg.instruction
        env_cfg.instruction = select_instruction_fields(env_cfg.instruction, instruction_type)

        # Configure the RTX renderer before the sim context is created.
        _apply_render_settings(env_cfg, renderer, rendering_mode)

        # Merge events into the environment configuration if provided
        # This preserves existing events (like reset_scene_to_default) while adding new ones
        if events is not None:
            env_cfg.events = merge_events_cfg(env_cfg.events, events)
            if robolab.constants.VERBOSE:
                print(f"Merged events into environment configuration: {env_cfg.events}")

        prepare_env_cfg(env_cfg)
        env = RobolabEnv(env_cfg)
    else:
        raise ValueError(f"Unsupported scene type: {type(scene)}")

    if env is None:
        raise ValueError(f"Failed to create environment for scene {scene}")

    check_scene_valid(env)

    # disable control on stop
    if hasattr(env.sim, "_app_control_on_stop_handle"):
        env.sim._app_control_on_stop_handle = None  # type: ignore

    env.output_dir = get_output_dir()
    os.makedirs(env.output_dir, exist_ok=True)

    if policy is not None:
        env_cfg.policy = policy

    # Stamp the resolved renderer for run provenance (serialized into env_cfg.json).
    env_cfg.renderer = renderer

    from robolab.core.utils.print_utils import print_env_info
    env_name = scene if isinstance(scene, str) else env_cfg.__class__.__name__
    _, display_instruction = primary_instruction(env_cfg.instruction)
    instruction_type_display = (
        "all" if instruction_type is None
        else instruction_type if isinstance(instruction_type, str)
        else ", ".join(instruction_type)
    )
    print_env_info(
        env_name=env_name,
        instruction=display_instruction,
        instruction_type=instruction_type_display,
        seed=env_cfg.seed,
        policy=policy or "",
        scene_name=env_cfg.scene.__class__.__name__,
        attributes=getattr(env_cfg, '_task_attributes', None),
    )

    # Save env_cfg as json for metadata
    with open(os.path.join(env.output_dir, "env_cfg.json"), "w") as f:
        json.dump(env_cfg_to_recording_dict(env_cfg), f, default=str)
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
