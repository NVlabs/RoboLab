# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.sensors import Camera
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from robolab.core.sensors.camera_utils import is_camera
from robolab.core.utils.isaaclab_compat import (
    as_torch,
    quat_isaaclab_to_wxyz,
    scene_state_from_isaaclab,
)

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
            if is_camera(sensor):
                # If camera_names is None or empty, record all cameras
                # Otherwise, only record specified cameras
                if not self._camera_names or name in self._camera_names:
                    self._cameras[name] = sensor

    def _get_camera_poses(self, env_ids: Sequence[int] | None = None):
        """Get camera poses for specified environment IDs.

        Position is returned in the env-local frame (relative to each env's scene
        origin), consistent with the env-local object poses recorded alongside it.
        Orientation is unaffected by the per-env translation and stays in world frame.
        """
        camera_poses = {}
        origins = self._env.scene.env_origins[:, 0:3]
        for name, camera in self._cameras.items():
            if env_ids is not None:
                camera_poses[name] = {
                    "position": (as_torch(camera.data.pos_w)[env_ids] - origins[env_ids]).clone(),
                    # ``ros`` describes the camera axes, not a stable tensor
                    # ordering. Isaac Lab 2 returns WXYZ and 3 returns XYZW.
                    "orientation": quat_isaaclab_to_wxyz(
                        as_torch(camera.data.quat_w_ros)[env_ids]
                    ).clone(),
                }
            else:
                camera_poses[name] = {
                    "position": (as_torch(camera.data.pos_w) - origins).clone(),
                    "orientation": quat_isaaclab_to_wxyz(
                        as_torch(camera.data.quat_w_ros)
                    ).clone(),
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
        state = scene_state_from_isaaclab(self._env.scene.get_state(is_relative=True))
        self.initial_state = self.extract_env_ids_values(state, env_ids)

        # Add camera extrinsics under 'cameras' key
        self._init_cameras()
        if self._cameras:
            self.initial_state["cameras"] = self._get_camera_poses(env_ids)

        return self.initial_state


class PostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        return "states", scene_state_from_isaaclab(self._env.scene.get_state(is_relative=True))


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
    for the specified end effector body.

    Position and orientation are recorded in the robot-root frame (relative to the
    articulation's root link), matching the ``ee_pos``/``ee_quat`` observation terms
    (see docs/frames.md). Velocities remain on world axes.
    """

    def __init__(self, cfg: "PostStepEndEffectorPoseRecorderCfg", env):
        super().__init__(cfg, env)
        self._robot_cfg_name = cfg.robot_cfg_name
        self._ee_body_name = cfg.ee_body_name
        self._record_key = cfg.record_key
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
                        # A misconfigured label must fail loudly: silently
                        # returning None here would drop the channel and every
                        # downstream metric with it.
                        raise ValueError(
                            f"[{self._record_key}] EE body '{self._ee_body_name}' not found on "
                            f"articulation '{self._robot_cfg_name}'. Available bodies: {body_names}. "
                            "Check the robot cfg's ee_recorder_bodies label (docs/robots.md)."
                        )
            else:
                raise ValueError(
                    f"[{self._record_key}] Articulation '{self._robot_cfg_name}' not found in scene "
                    f"(articulations: {sorted(self._env.scene.articulations)}). "
                    "Check the robot cfg's ee_recorder_bodies label (docs/robots.md)."
                )

        if self._robot is None or self._ee_body_idx is None:
            return None, None

        # Get body pose from articulation (already computed by physics, no extra cost)
        # and express it in the robot-root frame (docs/frames.md). For robots whose
        # root sits at the env origin with identity rotation (Franka family) this is
        # numerically identical to the old env-local recording.
        ee_pos, ee_quat = subtract_frame_transforms(
            as_torch(self._robot.data.root_pos_w),
            as_torch(self._robot.data.root_quat_w),
            as_torch(self._robot.data.body_pos_w)[:, self._ee_body_idx, :],
            as_torch(self._robot.data.body_quat_w)[:, self._ee_body_idx, :],
        )  # (num_envs, 3), (num_envs, 4)
        ee_quat = quat_isaaclab_to_wxyz(ee_quat)

        # Get body velocity from articulation
        ee_lin_vel = as_torch(self._robot.data.body_lin_vel_w)[:, self._ee_body_idx, :]
        ee_ang_vel = as_torch(self._robot.data.body_ang_vel_w)[:, self._ee_body_idx, :]

        return self._record_key, {
            "position": ee_pos,
            "orientation": ee_quat,
            "linear_velocity": ee_lin_vel,
            "angular_velocity": ee_ang_vel,
        }


class PostStepRobotRootPoseRecorder(RecorderTerm):
    """Recorder term that records the robot root pose at the end of each step.

    Position is recorded in the env-local frame (root world position minus each
    env's scene origin); orientation is the root quaternion (w, x, y, z), which
    the per-env origin does not affect. This is the bridge channel between
    env-local and robot-root quantities (see docs/frames.md): consumers must
    read it instead of assuming the root sits at the env origin, which holds
    for Franka-family robots but not for floor-standing embodiments.
    """

    def __init__(self, cfg: "PostStepRobotRootPoseRecorderCfg", env):
        super().__init__(cfg, env)
        self._robot_cfg_name = cfg.robot_cfg_name
        self._robot = None
        self._initialized = False

    def record_post_step(self):
        if not self._initialized:
            self._initialized = True
            if self._robot_cfg_name in self._env.scene.articulations:
                self._robot = self._env.scene[self._robot_cfg_name]
            else:
                print(f"[PostStepRobotRootPoseRecorder] Robot '{self._robot_cfg_name}' not found in scene.")

        if self._robot is None:
            return None, None

        root_pos = as_torch(self._robot.data.root_pos_w) - self._env.scene.env_origins[:, 0:3]
        root_quat = quat_isaaclab_to_wxyz(as_torch(self._robot.data.root_quat_w))
        return "robot_root_pose", {
            "position": root_pos,
            "orientation": root_quat,
        }


class InitialCameraExtrinsicsRecorder(RecorderTerm):
    """Recorder term that records camera extrinsics (position and orientation) after reset.

    Records the pose of all cameras in the scene after initialization/reset,
    similar to how InitialStateRecorder captures initial object poses.

    The camera pose consists of:
    - Position: env-local position of the camera (x, y, z), relative to each env's
      scene origin (consistent with the env-local object poses)
    - Orientation (quat_w_ros): World orientation using ROS camera axes, stored as (w, x, y, z)

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
            if is_camera(sensor):
                # If camera_names is None or empty, record all cameras
                # Otherwise, only record specified cameras
                if not self._camera_names or name in self._camera_names:
                    self._cameras[name] = sensor

        if not self._cameras:
            print(f"[InitialCameraExtrinsicsRecorder] No cameras found in scene.")

    def _get_camera_poses(self, env_ids: Sequence[int] | None = None):
        """Get camera poses for specified environment IDs.

        Position is returned in the env-local frame (relative to each env's scene
        origin), consistent with the env-local object poses recorded alongside it.
        Orientation is unaffected by the per-env translation and stays in world frame.
        """
        camera_poses = {}
        origins = self._env.scene.env_origins[:, 0:3]
        for name, camera in self._cameras.items():
            if env_ids is not None:
                camera_poses[name] = {
                    "position": (as_torch(camera.data.pos_w)[env_ids] - origins[env_ids]).clone(),
                    "orientation": quat_isaaclab_to_wxyz(
                        as_torch(camera.data.quat_w_ros)[env_ids]
                    ).clone(),
                }
            else:
                camera_poses[name] = {
                    "position": (as_torch(camera.data.pos_w) - origins).clone(),
                    "orientation": quat_isaaclab_to_wxyz(
                        as_torch(camera.data.quat_w_ros)
                    ).clone(),
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
        robot_cfg_name: Name of the robot articulation in the scene. Default: "robot";
            multi-articulation embodiments set it per channel via the dict form of
            the ``ee_recorder_bodies`` label (``{"asset": ..., "body": ...}``).
        ee_body_name: Name of the end effector body to record. Required; sourced
            from the robot cfg's ``ee_recorder_bodies`` label (see docs/robots.md).
        record_key: HDF5 channel name to record under (e.g. "ee_pose"). Required.
    """

    class_type: type[RecorderTerm] = PostStepEndEffectorPoseRecorder
    robot_cfg_name: str = "robot"
    ee_body_name: str = MISSING
    record_key: str = MISSING


@configclass
class PostStepRobotRootPoseRecorderCfg(RecorderTermCfg):
    """Configuration for the robot root pose recorder term.

    Attributes:
        robot_cfg_name: Name of the robot articulation in the scene. Default: "robot"
    """

    class_type: type[RecorderTerm] = PostStepRobotRootPoseRecorder
    robot_cfg_name: str = "robot"


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
