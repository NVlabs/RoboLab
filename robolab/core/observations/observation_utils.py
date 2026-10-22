# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List

import isaaclab.envs.mdp as mdp
import numpy as np
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from robolab.core.utils.isaaclab_compat import as_torch, quat_isaaclab_to_wxyz


def _image_observation_func():
    """Resolve the IsaacLab image observation function across versions.

    IsaacLab 2.2 (IsaacSim 5.0) exposes it as ``mdp.observations.image``;
    IsaacLab 2.3 (IsaacSim 5.1) also exposes it directly as ``mdp.image``.
    Prefer the direct attribute, then fall back to the nested module.
    """
    image_func = getattr(mdp, "image", None)
    if image_func is not None:
        return image_func
    observations = getattr(mdp, "observations", None)
    image_func = getattr(observations, "image", None) if observations is not None else None
    if image_func is None:
        raise AttributeError("IsaacLab image observation function not found")
    return image_func


def image_safe(
    env: Any,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("over_shoulder_left_camera"),
    data_type: str = "rgb",
    normalize: bool = True,
) -> torch.Tensor:
    """Read a camera image, tolerating the first pre-render manager query.

    With lazy sensor updates disabled, annotators such as ``depth`` are only
    populated after the first render. Return a correctly shaped-and-typed zero
    image until then so the observation manager can size its buffers.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    if data_type not in sensor.data.output:
        channels = 3 if data_type == "rgb" else 1
        dtype = torch.float32 if (normalize or data_type != "rgb") else torch.uint8
        return torch.zeros(
            (env.num_envs, sensor.cfg.height, sensor.cfg.width, channels),
            device=env.device,
            dtype=dtype,
        )
    return _image_observation_func()(
        env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize
    )


def camera_pos(env: Any, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera position, env-local frame (world minus env origin), meters. Shape (num_envs, 3)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return as_torch(sensor.data.pos_w) - env.scene.env_origins


def camera_quat(env: Any, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera orientation, world frame, OpenGL convention, (w, x, y, z). Shape (num_envs, 4)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return quat_isaaclab_to_wxyz(as_torch(sensor.data.quat_w_opengl))


def camera_intrinsics(env: Any, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera intrinsic matrix K in pixels. Shape (num_envs, 3, 3)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    matrices = getattr(sensor.data, "intrinsic_matrices", None)
    if matrices is None:
        return torch.zeros((env.num_envs, 3, 3), device=env.device, dtype=torch.float32)
    return as_torch(matrices)


def object_pos(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object root position, env-local frame (world minus env origin), meters. Shape (num_envs, 3)."""
    asset = env.scene[asset_cfg.name]
    return as_torch(asset.data.root_pos_w) - env.scene.env_origins


def object_quat(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object root orientation, world frame, (w, x, y, z). Shape (num_envs, 4)."""
    return quat_isaaclab_to_wxyz(as_torch(env.scene[asset_cfg.name].data.root_quat_w))


def object_vel(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object root velocity, world frame, linear (m/s) + angular (rad/s). Shape (num_envs, 6)."""
    return as_torch(env.scene[asset_cfg.name].data.root_vel_w)


def generate_object_state_obs(object_names: List[str]):
    """Create an observation group with ground-truth state terms per object.

    Each object gets ``<name>_pos`` (env-local meters), ``<name>_quat``
    (world-frame w, x, y, z), and ``<name>_vel`` (world-frame linear+angular)
    terms, evaluated once per environment step like any other observation.
    Opt in at registration time via ``object_state_obs=True`` on
    ``auto_register_droid_envs`` / ``generate_task_env_cfg`` — the object
    list then comes from the task's ``contact_object_list``.

    Args:
        object_names: Scene entity names to observe (rigid objects or
            articulations with a root state).

    Returns:
        A dynamically generated observation group class (ObsGroup subclass).
    """
    obs_terms = {}
    for name in object_names:
        for suffix, func in (("pos", object_pos), ("quat", object_quat), ("vel", object_vel)):
            obs_terms[f"{name}_{suffix}"] = ObsTerm(
                func=func,
                params={"asset_cfg": SceneEntityCfg(name)},
            )

    @configclass
    class DynamicObjectStateObsCfg(ObsGroup):
        """Dynamically generated ground-truth object-state observations."""

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    for term_name, obs_term in obs_terms.items():
        setattr(DynamicObjectStateObsCfg, term_name, obs_term)

    return DynamicObjectStateObsCfg


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
                        func=_image_observation_func(),
                        params={
                            "sensor_cfg": SceneEntityCfg(camera_name),
                            "data_type": "rgb",
                            "normalize": False,
                        }
                    )
                    # Cameras that render depth (see robolab.variations.camera.with_depth)
                    # also get <camera>_depth plus pose/intrinsics metadata terms, so
                    # calibration flows through the standard observation pipeline
                    # (per-env, recorded like any other term).
                    if "depth" in attr_value.data_types:
                        obs_terms[f"{camera_name}_depth"] = ObsTerm(
                            func=image_safe,
                            params={
                                "sensor_cfg": SceneEntityCfg(camera_name),
                                "data_type": "depth",
                                "normalize": False,
                            }
                        )
                        for term_suffix, term_func in (
                            ("pos", camera_pos),
                            ("quat", camera_quat),
                            ("K", camera_intrinsics),
                        ):
                            obs_terms[f"{camera_name}_{term_suffix}"] = ObsTerm(
                                func=term_func,
                                params={"sensor_cfg": SceneEntityCfg(camera_name)},
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
