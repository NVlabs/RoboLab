# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from collections.abc import Sequence

import torch
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.sensors import Camera
from isaaclab.utils import configclass

########################################################################################
# Recorder terms. Adapted from isaaclab.envs.mdp.recorders.recorders.
########################################################################################

class InitialStateRecorder(RecorderTerm):
    """Recorder term that records the initial state of the environment after reset.

    This includes scene state (articulations, rigid objects) and optionally camera
    extrinsics (position and orientation) under the 'cameras' key.
    """
    initial_state = None

    def __init__(self, cfg: "InitialStateRecorderCfg", env):
        super().__init__(cfg, env)
        self._camera_names = cfg.camera_names
        self._cameras: dict[str, Camera] = {}
        self._cameras_initialized = False

    def _init_cameras(self):
        """Lazy initialization to find cameras in the scene."""
        if self._cameras_initialized:
            return

        self._cameras_initialized = True
        # Get all cameras from scene sensors
        for name, sensor in self._env.scene.sensors.items():
            if isinstance(sensor, Camera):
                # If camera_names is None or empty, record all cameras
                # Otherwise, only record specified cameras
                if not self._camera_names or name in self._camera_names:
                    self._cameras[name] = sensor

    def _get_camera_poses(self, env_ids: Sequence[int] | None = None):
        """Get camera poses for specified environment IDs."""
        camera_poses = {}
        for name, camera in self._cameras.items():
            if env_ids is not None:
                camera_poses[name] = {
                    "position": camera.data.pos_w[env_ids].clone(),  # (len(env_ids), 3)
                    "orientation": camera.data.quat_w_ros[env_ids].clone(),  # (len(env_ids), 4)
                }
            else:
                camera_poses[name] = {
                    "position": camera.data.pos_w.clone(),  # (num_envs, 3)
                    "orientation": camera.data.quat_w_ros.clone(),  # (num_envs, 4)
                }
        return camera_poses

    def extract_env_ids_values(self, value, env_ids):
        if isinstance(value, dict):
            return {k: self.extract_env_ids_values(v, env_ids) for k, v in value.items()}
        return value[env_ids]

    def record_post_reset(self, env_ids: Sequence[int] | None):
        return "initial_state", self.initial_state

    def reset(self, env_ids: Sequence[int] | None = None):
        # Captures this value and returns it so that it is accessible during episodes.
        # Note, reset here is called *after* randomization and object position, so it doesn't necessarily have to be in post_reset.
        self.initial_state = self.extract_env_ids_values(self._env.scene.get_state(is_relative=True), env_ids)

        # Add camera extrinsics under 'cameras' key
        self._init_cameras()
        if self._cameras:
            self.initial_state["cameras"] = self._get_camera_poses(env_ids)

        return self.initial_state


class PostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepFlatPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def record_pre_step(self):
        return "obs", self._env.obs_buf


class PostStepEndEffectorPoseRecorder(RecorderTerm):
    """Recorder term that records the end effector pose and velocity at the end of each step.

    Uses the articulation's body state directly (no FrameTransformer needed).
    Records position, orientation (quaternion), linear velocity, and angular velocity
    in world frame for the specified end effector body.
    """

    def __init__(self, cfg: "PostStepEndEffectorPoseRecorderCfg", env):
        super().__init__(cfg, env)
        self._robot_cfg_name = cfg.robot_cfg_name
        self._ee_body_name = cfg.ee_body_name
        self._robot = None
        self._ee_body_idx = None
        self._initialized = False

    def record_post_step(self):
        # Lazy initialization to find robot and EE body index
        if not self._initialized:
            self._initialized = True
            if self._robot_cfg_name in self._env.scene.articulations:
                self._robot = self._env.scene[self._robot_cfg_name]
                body_names = self._robot.data.body_names
                # Try exact match first, then partial match
                if self._ee_body_name in body_names:
                    self._ee_body_idx = body_names.index(self._ee_body_name)
                else:
                    # Try to find a body containing the ee_body_name
                    for i, name in enumerate(body_names):
                        if self._ee_body_name in name:
                            self._ee_body_idx = i
                            break
                    if self._ee_body_idx is None:
                        print(f"[PostStepEndEffectorPoseRecorder] Body '{self._ee_body_name}' not found. Available: {body_names}")
            else:
                print(f"[PostStepEndEffectorPoseRecorder] Robot '{self._robot_cfg_name}' not found in scene.")

        if self._robot is None or self._ee_body_idx is None:
            return None, None

        # Get body pose from articulation (already computed by physics, no extra cost)
        ee_pos = self._robot.data.body_pos_w[:, self._ee_body_idx, :]  # (num_envs, 3)
        ee_quat = self._robot.data.body_quat_w[:, self._ee_body_idx, :]  # (num_envs, 4)

        # Get body velocity from articulation
        ee_lin_vel = self._robot.data.body_lin_vel_w[:, self._ee_body_idx, :]  # (num_envs, 3)
        ee_ang_vel = self._robot.data.body_ang_vel_w[:, self._ee_body_idx, :]  # (num_envs, 3)

        return "ee_pose", {
            "position": ee_pos,
            "orientation": ee_quat,
            "linear_velocity": ee_lin_vel,
            "angular_velocity": ee_ang_vel,
        }


class InitialCameraExtrinsicsRecorder(RecorderTerm):
    """Recorder term that records camera extrinsics (position and orientation) after reset.

    Records the world pose of all cameras in the scene after initialization/reset,
    similar to how InitialStateRecorder captures initial object poses.

    The camera pose consists of:
    - Position (pos_w): World position of the camera (x, y, z)
    - Orientation (quat_w_ros): World orientation as quaternion in ROS convention (x, y, z, w)

    This is useful for recording camera viewpoint at the start of each episode,
    especially when camera pose randomization is enabled.
    """
    _initial_camera_extrinsics = None

    def __init__(self, cfg: "InitialCameraExtrinsicsRecorderCfg", env):
        super().__init__(cfg, env)
        self._camera_names = cfg.camera_names
        self._cameras: dict[str, Camera] = {}
        self._initialized = False

    def _init_cameras(self):
        """Lazy initialization to find cameras in the scene."""
        if self._initialized:
            return

        self._initialized = True
        # Get all cameras from scene sensors
        for name, sensor in self._env.scene.sensors.items():
            if isinstance(sensor, Camera):
                # If camera_names is None or empty, record all cameras
                # Otherwise, only record specified cameras
                if not self._camera_names or name in self._camera_names:
                    self._cameras[name] = sensor

        if not self._cameras:
            print(f"[InitialCameraExtrinsicsRecorder] No cameras found in scene.")

    def _get_camera_poses(self, env_ids: Sequence[int] | None = None):
        """Get camera poses for specified environment IDs."""
        camera_poses = {}
        for name, camera in self._cameras.items():
            if env_ids is not None:
                camera_poses[name] = {
                    "position": camera.data.pos_w[env_ids].clone(),  # (len(env_ids), 3)
                    "orientation": camera.data.quat_w_ros[env_ids].clone(),  # (len(env_ids), 4)
                }
            else:
                camera_poses[name] = {
                    "position": camera.data.pos_w.clone(),  # (num_envs, 3)
                    "orientation": camera.data.quat_w_ros.clone(),  # (num_envs, 4)
                }
        return camera_poses

    def record_post_reset(self, env_ids: Sequence[int] | None):
        """Record camera extrinsics after reset."""
        return "initial_camera_extrinsics", self._initial_camera_extrinsics

    def reset(self, env_ids: Sequence[int] | None = None):
        """Capture camera extrinsics after reset (called after randomization events)."""
        self._init_cameras()

        if not self._cameras:
            return None

        # Capture current camera poses (after any randomization has been applied)
        self._initial_camera_extrinsics = self._get_camera_poses(env_ids)
        return self._initial_camera_extrinsics


########################################################################################
# Recorder manager configurations.
########################################################################################

@configclass
class InitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term.

    Records the initial scene state (articulations, rigid objects) and optionally
    camera extrinsics (position and orientation) under the 'cameras' key.

    Attributes:
        camera_names: List of camera names to record extrinsics for. If None or empty,
            all cameras in the scene will be recorded. Default: None (record all)
    """

    class_type: type[RecorderTerm] = InitialStateRecorder
    camera_names: list[str] | None = None


@configclass
class PostStepStatesRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = PostStepStatesRecorder


@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action recorder term."""

    class_type: type[RecorderTerm] = PreStepActionsRecorder


@configclass
class PreStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatPolicyObservationsRecorder


@configclass
class PostStepEndEffectorPoseRecorderCfg(RecorderTermCfg):
    """Configuration for the end effector pose recorder term.

    Uses the articulation's body state directly (no FrameTransformer needed).

    Attributes:
        robot_cfg_name: Name of the robot articulation in the scene. Default: "robot"
        ee_body_name: Name of the end effector body to record. Default: "base_link"
            (for DROID, this matches Gripper/Robotiq_2F_85/base_link)
    """

    class_type: type[RecorderTerm] = PostStepEndEffectorPoseRecorder
    robot_cfg_name: str = "robot"
    ee_body_name: str = "base_link"


@configclass
class InitialCameraExtrinsicsRecorderCfg(RecorderTermCfg):
    """Configuration for the initial camera extrinsics recorder term.

    Records the world pose of cameras after reset, similar to InitialStateRecorder.
    Useful for recording camera viewpoint at the start of each episode,
    especially when camera pose randomization is enabled.

    Attributes:
        camera_names: List of camera names to record. If None or empty,
            all cameras in the scene will be recorded. Default: None (record all)
    """

    class_type: type[RecorderTerm] = InitialCameraExtrinsicsRecorder
    camera_names: list[str] | None = None


class PostStepBBoxRecorder(RecorderTerm):
    """Recorder term that records bounding box corners and centroids per rigid object.

    Saves per-step data for all rigid objects in the scene:
    - ``bbox_mm/{object_name}``: OBB corners as int16 in millimeters, shape ``(num_envs, 8, 3)``
      Corner order: [0-3] bottom face, [4-7] top face (same as ``usd_utils.get_bbox``).
    - ``centroid/{object_name}``: centroid as float16 in meters, shape ``(num_envs, 3)``
    """

    def __init__(self, cfg: "PostStepBBoxRecorderCfg", env):
        super().__init__(cfg, env)
        self._world = None
        self._object_names: list[str] = []
        self._initialized = False

    def _init(self):
        if self._initialized:
            return
        self._initialized = True
        from robolab.core.world.world_state import get_world
        self._world = get_world(self._env)
        self._object_names = list(self._env.scene.rigid_objects.keys())

    def record_post_step(self):
        self._init()
        if not self._object_names:
            return None, None

        result = {}
        for name in self._object_names:
            corners, centroid = self._world.get_bbox(name, env_id=None)
            # corners: (N, 8, 3) float32 → int16 mm
            result[f"bbox_mm/{name}"] = (corners * 1000).to(torch.int16)
            # centroid: (N, 3) float32 → float16
            result[f"centroid/{name}"] = centroid.to(torch.float16)

        return "bbox", result


@configclass
class PostStepBBoxRecorderCfg(RecorderTermCfg):
    """Configuration for the bounding box recorder term.

    Records OBB corners (int16, millimeters) and centroids (float16, meters)
    for all rigid objects at each step.
    """

    class_type: type[RecorderTerm] = PostStepBBoxRecorder
