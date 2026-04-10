# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Any, List

import isaaclab.envs.mdp as mdp
import numpy as np
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


def generate_image_obs_from_cameras(camera_cfgs: List[Any] | Any):
    """
    Dynamically create an image observation group configuration from one or more camera configs.

    Example usage:
        from robolab.variations.camera import EgocentricWideAngleCameraCfg, EgocentricMirroredWideAngleCameraCfg, EgocentricMirroredCameraCfg

        # Create a list of camera configuration classes
        camera_cfgs = [EgocentricWideAngleCameraCfg, EgocentricMirroredWideAngleCameraCfg, EgocentricMirroredCameraCfg]

        # Generate dynamic observation group configuration
        DynamicImageObsCfg = generate_image_obs_from_cameras(camera_cfgs)

        # Instantiate the configuration
        image_obs_cfg = DynamicImageObsCfg()

        # The image_obs_cfg will now contain observation terms for all cameras found in the camera configs:
        # - egocentric_wide_angle_camera
        # - egocentric_mirrored_wide_angle_camera
        # - egocentric_mirrored_camera

    Args:
        camera_cfgs: List of camera configuration classes (e.g., [EgocentricWideAngleCameraCfg, ...])

    Returns:
        A dynamically generated observation group configuration class (ObsGroup)
    """
    # Create a dictionary to store observation terms
    obs_terms = {}

    if not isinstance(camera_cfgs, list):
        camera_cfgs = [camera_cfgs]

    # Iterate through camera configs and extract camera names
    for camera_cfg in camera_cfgs:
        # Create an instance of the camera config to access its attributes
        camera_cfg_instance = camera_cfg()

        # Get all camera attributes from the config class instance
        for attr_name in dir(camera_cfg_instance):
            if not attr_name.startswith('_'):
                attr_value = getattr(camera_cfg_instance, attr_name)
                # Check if this attribute is a CameraCfg instance (has prim_path)
                if isinstance(attr_value, CameraCfg):
                # if hasattr(attr_value, 'prim_path'):
                    camera_name = attr_name
                    obs_terms[camera_name] = ObsTerm(
                        func=mdp.observations.image,
                        params={
                            "sensor_cfg": SceneEntityCfg(camera_name),
                            "data_type": "rgb",
                            "normalize": False,
                        }
                    )

    # Create the dynamic image observation group class
    @configclass
    class DynamicImageObsCfg(ObsGroup):
        """Dynamically generated image observations for policy."""

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # Add observation terms to the class
    for camera_name, obs_term in obs_terms.items():
        setattr(DynamicImageObsCfg, camera_name, obs_term)

    return DynamicImageObsCfg

def generate_obs_cfg(obs_groups: dict[str, ObsGroup]):
    """
    Dynamically create an observation configuration from multiple observation groups.

    Example usage:
        from robolab.variations.camera import EgocentricWideAngleCameraCfg, EgocentricMirroredWideAngleCameraCfg
        from robolab.core.observations.image_obs import generate_image_obs_from_cameras, generate_obs_cfg

        # Generate image observation group
        ImageObsCfg = generate_image_obs_from_cameras([EgocentricWideAngleCameraCfg, EgocentricMirroredWideAngleCameraCfg])

        # Create main observation configuration with multiple groups
        ObservationCfg = generate_obs_cfg({
            "image_obs": ImageObsCfg(),
            # "policy": other_obs_group,  # Could add other observation groups
        })

        # Instantiate the configuration
        obs_cfg = ObservationCfg()

    Args:
        obs_groups: Dictionary mapping group names to ObsGroup instances
                   e.g., {"image_obs": ImageObsCfg(), "policy": PolicyObsCfg()}

    Returns:
        A dynamically generated observation configuration class
    """
    # Create the main configuration class
    @configclass
    class DynamicObservationCfg:
        """Dynamically generated observation configuration."""
        pass

    # Add observation groups to the class
    for group_name, obs_group in obs_groups.items():
        setattr(DynamicObservationCfg, group_name, obs_group)

    return DynamicObservationCfg

def unpack_image_obs(obs, obs_group_name="image_obs", camera_suffix=["_camera", "_cam", "_img", "_image"], scale: float = 1.0, env_id: int = 0):
    """
    Unpack image observations from an observation dictionary.

    Args:
        obs: Observation dictionary
        obs_group_name: Name of the observation group to unpack, default is "image_obs"
        camera_suffix: Suffix of the camera to unpack, default is ["_camera", "_cam", "_img", "_image"]
        scale: Scale factor for resizing on GPU before CPU transfer (0.5 = half size).
               Resizing on GPU is faster than transferring full resolution then resizing on CPU.

    Returns:
        Dictionary containing the unpacked image observations
    """
    import torch.nn.functional as F

    images = []
    image_dict = {}
    for key, value in obs[obs_group_name].items():
        if any(key.endswith(suffix) for suffix in camera_suffix):
            tensor = value[env_id].detach()

            # Resize on GPU before CPU transfer (much faster for small scale values)
            if scale != 1.0:
                # tensor shape is (H, W, C), need (1, C, H, W) for interpolate
                # Convert to float for interpolate, then back to uint8
                original_dtype = tensor.dtype
                tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
                tensor = F.interpolate(tensor, scale_factor=scale, mode='bilinear', align_corners=False)
                tensor = tensor.squeeze(0).permute(1, 2, 0).to(original_dtype)

            image = tensor.cpu().numpy()
            image_dict[key] = image
            images.append(image)
    combined_image = np.concatenate(images, axis=1)
    image_dict["combined_image"] = combined_image
    return image_dict

def unpack_proprio_obs(obs, obs_group_name="proprio_obs", env_id: int = 0):
    """
    Unpack proprioceptive observations from an observation dictionary.

    Args:
        obs: Observation dictionary
        obs_group_name: Name of the observation group to unpack, default is "proprio_obs"
        env_id: Environment index to unpack, default is 0

    Returns:
        Dictionary containing the unpacked proprioceptive observations
    """
    proprio_dict = {}
    for key, value in obs[obs_group_name].items():
        proprio_dict[key] = value[env_id].clone().detach().cpu().numpy()
    return proprio_dict

def unpack_viewport_cams(obs, obs_group_name="viewport_cam", camera_suffix=["_camera", "_cam", "_img", "_image"], scale: float = 1.0, env_id: int = 0):
    """
    Unpack viewport camera observations from an observation dictionary.

    Args:
        obs: Observation dictionary
        obs_group_name: Name of the observation group to unpack, default is "viewport_cam"
        camera_suffix: Suffix of the camera to unpack, default is ["_camera", "_cam", "_img", "_image"]
        scale: Scale factor for resizing on GPU before CPU transfer (0.5 = half size).
               Resizing on GPU is faster than transferring full resolution then resizing on CPU.
        env_id: Environment index to unpack, default is 0

    Returns:
        Dictionary containing the unpacked viewport camera observations
    """
    import torch.nn.functional as F

    images = []
    viewport_dict = {}
    for key, value in obs[obs_group_name].items():
        if any(key.endswith(suffix) for suffix in camera_suffix):
            tensor = value[env_id].detach()

            # Resize on GPU before CPU transfer (much faster for small scale values)
            if scale != 1.0:
                # tensor shape is (H, W, C), need (1, C, H, W) for interpolate
                # Convert to float for interpolate, then back to uint8
                original_dtype = tensor.dtype
                tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
                tensor = F.interpolate(tensor, scale_factor=scale, mode='bilinear', align_corners=False)
                tensor = tensor.squeeze(0).permute(1, 2, 0).to(original_dtype)

            image = tensor.cpu().numpy()
            viewport_dict[key] = image
            images.append(image)
    combined_image = np.concatenate(images, axis=1)
    viewport_dict["combined_image"] = combined_image
    return viewport_dict