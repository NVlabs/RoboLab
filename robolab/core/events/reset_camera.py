# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import Camera
from isaaclab.utils import configclass

import robolab.constants


########################################################
#  Config class for randomizing camera pose
########################################################
@configclass
class RandomizeCameraPoseUniform:
    """Configuration for randomizing camera pose uniformly."""

    @classmethod
    def from_params(cls,
        cameras: list[str] | str,
        pose_range: dict):
        """Create a RandomizeCameraPoseUniform instance with custom parameters.

        Args:
            cameras: The camera name(s) to randomize. Must match names in env.scene.sensors.
            pose_range: Dictionary of pose ranges for each axis (x, y, z, roll, pitch, yaw).
                Each key maps to a tuple of (min, max) values.
                Example: {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
                         "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)}
        """
        # Create a new class with the custom parameters
        class CustomRandomizeCameraPoseUniform(cls):
            randomize_camera_pose = EventTerm(
                func=reset_camera_pose_uniform,
                mode="reset",
                params={
                    "pose_range": pose_range,
                    "camera_names": cameras,
                }
            )

        return CustomRandomizeCameraPoseUniform()


########################################################
#  Camera pose parsing utilities
########################################################

def _parse_camera_names(camera_cfg: list[str] | str) -> list[str]:
    """Parse camera_cfg into a list of camera names.

    Args:
        camera_cfg: Camera configuration - can be a string or list of strings.

    Returns:
        List of camera name strings.
    """
    if isinstance(camera_cfg, str):
        return [camera_cfg]
    elif isinstance(camera_cfg, list):
        return [name for name in camera_cfg if isinstance(name, str)]
    else:
        return list(camera_cfg)


def _get_all_camera_names(env: ManagerBasedEnv) -> set[str]:
    """Get all camera names in the scene."""
    camera_names = set()
    for name, sensor in env.scene.sensors.items():
        if isinstance(sensor, Camera):
            camera_names.add(name)
    return camera_names


########################################################
#  Camera pose sampling and reset functions
########################################################

def sample_camera_pose_uniform(
    env: ManagerBasedEnv,
    camera: Camera,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random camera pose uniformly within the given ranges.

    The pose is sampled as a delta from the camera's default/initial pose.

    Args:
        env: The environment instance.
        camera: The camera sensor to sample pose for.
        env_ids: The environment indices to sample for.
        pose_range: Dictionary of pose ranges for each axis.
            Keys are "x", "y", "z", "roll", "pitch", "yaw".
            Values are tuples of (min, max).

    Returns:
        Tuple of (positions, orientations) tensors.
            - positions: Shape (num_envs, 3)
            - orientations: Shape (num_envs, 4) in quaternion format (w, x, y, z)
    """
    num_envs = len(env_ids)
    device = camera.device

    # Get current camera poses as the base
    # Camera data provides pos_w and quat_w_ros (or quat_w depending on convention)
    current_positions = camera.data.pos_w[env_ids].clone()
    current_orientations = camera.data.quat_w_ros[env_ids].clone()

    # Create pose ranges tensor
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)

    # Sample random deltas
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=device)

    # Apply position deltas
    positions = current_positions + rand_samples[:, 0:3]

    # Apply orientation deltas as euler angles
    orientation_deltas = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(current_orientations, orientation_deltas)

    if robolab.constants.VERBOSE:
        print(f"Camera positions before: {current_positions}")
        print(f"Camera positions after: {positions}")
        print(f"Camera orientation deltas (euler): {rand_samples[:, 3:6]}")

    return positions, orientations


def reset_camera_pose_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    camera_names: list[str] | str,
):
    """Reset camera pose to a random position and orientation uniformly within the given ranges.

    This function randomizes the camera pose by applying a delta to the current camera pose.
    The delta is sampled uniformly from the specified ranges.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples
    of the form ``(min, max)``. If the dictionary does not contain a key, the position or
    orientation delta is set to zero for that axis.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        pose_range: Dictionary of pose ranges for each axis.
            Example: {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
                     "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)}
        camera_names: The camera name(s) to randomize. Must match names in env.scene.sensors.
    """
    names = _parse_camera_names(camera_names)

    for camera_name in names:
        # Get camera from scene sensors
        if camera_name not in env.scene.sensors:
            if robolab.constants.VERBOSE:
                print(f"Warning: Camera '{camera_name}' not found in scene sensors. Skipping.")
            continue

        camera = env.scene.sensors[camera_name]
        if not isinstance(camera, Camera):
            if robolab.constants.VERBOSE:
                print(f"Warning: Sensor '{camera_name}' is not a Camera. Skipping.")
            continue

        if robolab.constants.VERBOSE:
            print(f"Randomizing camera pose for '{camera_name}' with ranges: {pose_range}")

        # Sample new poses
        positions, orientations = sample_camera_pose_uniform(
            env, camera, env_ids, pose_range
        )

        # Set the camera poses
        # Note: set_world_poses expects positions (N, 3) and orientations (N, 4)
        camera.set_world_poses(positions=positions, orientations=orientations, env_ids=env_ids)

        if robolab.constants.VERBOSE:
            print(f"Camera '{camera_name}' pose updated successfully")


def reset_camera_pose_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    camera_names: list[str] | str,
):
    """Reset the specified cameras' pose to the default state specified in the scene configuration.

    This function resets cameras to their initial/default poses as defined in the CameraCfg.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        camera_names: The camera name(s) to reset. Must match names in env.scene.sensors.
    """
    names = _parse_camera_names(camera_names)

    for camera_name in names:
        if camera_name not in env.scene.sensors:
            if robolab.constants.VERBOSE:
                print(f"Warning: Camera '{camera_name}' not found in scene sensors. Skipping.")
            continue

        camera = env.scene.sensors[camera_name]
        if not isinstance(camera, Camera):
            if robolab.constants.VERBOSE:
                print(f"Warning: Sensor '{camera_name}' is not a Camera. Skipping.")
            continue

        # Reset to the camera's default/initial pose
        # The camera's cfg.offset contains the default offset from parent frame
        # We need to reset using the camera's reset method or set to original values
        camera.reset(env_ids)

        if robolab.constants.VERBOSE:
            print(f"Camera '{camera_name}' reset to default pose")


def reset_camera_pose_absolute(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    camera_names: list[str] | str,
    position: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Reset camera pose to a specific absolute position and orientation.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        camera_names: The camera name(s) to reset. Must match names in env.scene.sensors.
        position: The target position as (x, y, z). If None, position is unchanged.
        orientation: The target orientation as quaternion (w, x, y, z). If None, orientation is unchanged.
    """
    names = _parse_camera_names(camera_names)

    for camera_name in names:
        if camera_name not in env.scene.sensors:
            if robolab.constants.VERBOSE:
                print(f"Warning: Camera '{camera_name}' not found in scene sensors. Skipping.")
            continue

        camera = env.scene.sensors[camera_name]
        if not isinstance(camera, Camera):
            if robolab.constants.VERBOSE:
                print(f"Warning: Sensor '{camera_name}' is not a Camera. Skipping.")
            continue

        num_envs = len(env_ids)
        device = camera.device

        # Get current poses
        current_positions = camera.data.pos_w[env_ids].clone()
        current_orientations = camera.data.quat_w_ros[env_ids].clone()

        # Override with specified values
        if position is not None:
            positions = torch.tensor([position], device=device).expand(num_envs, -1)
        else:
            positions = current_positions

        if orientation is not None:
            orientations = torch.tensor([orientation], device=device).expand(num_envs, -1)
        else:
            orientations = current_orientations

        # Set the camera poses
        camera.set_world_poses(positions=positions, orientations=orientations, env_ids=env_ids)

        if robolab.constants.VERBOSE:
            print(f"Camera '{camera_name}' pose set to position={position}, orientation={orientation}")
