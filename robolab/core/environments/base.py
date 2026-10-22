# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base configuration classes for RoboLab environments.

This module contains all the base configuration classes used to define
environment configurations, including observations, actions, events,
rewards, terminations, and the main RobolabDefaultEnvCfg.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robolab.constants
from robolab.constants import get_output_dir
from robolab.core.events.basic_recorders import (
    InitialStateRecorderCfg,
    PostStepBBoxRecorderCfg,
    PostStepEndEffectorPoseRecorderCfg,
    PostStepRobotRootPoseRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
    PreStepFlatPolicyObservationsRecorderCfg,
)
from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderCfg


@configclass
class ObservationCfg:
    """Observation terms for the MDP."""

@configclass
class ActionCfg:
    """Observation terms for the MDP."""

@configclass
class BaseEventCfg:
    """Configuration for events."""
    reset = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class BaseRecorderManagerCfg(RecorderManagerBaseCfg):
    """Base recorder configuration with common settings. By default, proprio data is recorded.

    Note: Camera extrinsics are recorded automatically as part of initial_state under
    the 'cameras' key. Use InitialStateRecorderCfg.camera_names to filter which cameras
    are recorded (default: all cameras).
    """
    record_initial_state: InitialStateRecorderCfg = InitialStateRecorderCfg()
    record_states: PostStepStatesRecorderCfg = PostStepStatesRecorderCfg()
    record_actions: PreStepActionsRecorderCfg = PreStepActionsRecorderCfg()
    # EE-pose channels are attached dynamically by create_recorder_config from
    # the robot cfg's mandatory ee_recorder_bodies label (one term per channel).
    record_robot_root_pose: PostStepRobotRootPoseRecorderCfg = PostStepRobotRootPoseRecorderCfg()
    record_bbox: PostStepBBoxRecorderCfg = PostStepBBoxRecorderCfg()
    record_policy_observations: PreStepFlatPolicyObservationsRecorderCfg | None = None
    record_subtask_completion: SubtaskCompletionRecorderCfg | None = None
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL

def resolve_ee_body_label(value: str | dict) -> tuple[str, str]:
    """Resolve one ``ee_recorder_bodies`` value to ``(articulation name, body name)``.

    A bare string names a body on the primary ``robot`` articulation. The
    explicit dict form ``{"asset": <articulation>, "body": <body>}`` selects a
    body on another articulation — required for multi-articulation embodiments
    (see docs/robots.md). ``"asset"`` is optional and defaults to ``"robot"``.
    """
    if isinstance(value, str):
        return "robot", value
    if not isinstance(value, dict):
        raise TypeError(f"ee_recorder_bodies values must be str or dict, got {type(value).__name__}: {value!r}")
    unknown_keys = set(value) - {"asset", "body"}
    if unknown_keys:
        raise ValueError(f"Unknown ee_recorder_bodies keys {sorted(unknown_keys)}; expected 'asset' and 'body'")
    if "body" not in value:
        raise ValueError(f"ee_recorder_bodies dict entry is missing 'body': {value!r}")
    return value.get("asset", "robot"), value["body"]


def create_recorder_config(
    include_policy_observations: bool = False,
    include_subtask_tracking: bool = False,
    export_dir: str | None = None,
    filename: str = "data.hdf5",
    ee_recorder_bodies: dict[str, str | dict] | None = None,
) -> RecorderManagerBaseCfg:
    """Factory function to create appropriate recorder configuration.

    Args:
        include_policy_observations: Whether to record policy observations (images, etc.)
        include_subtask_tracking: Whether to track subtask completion
        export_dir: Directory to export data to
        filename: Name of the output file
        ee_recorder_bodies: Mapping of HDF5 channel name -> EE body, one
            recorder term per entry. Values are a body name on the primary
            ``robot`` articulation, or ``{"asset": ..., "body": ...}`` to record
            from another articulation (see ``resolve_ee_body_label``). Required;
            sourced from the robot cfg's ``ee_recorder_bodies`` label. Pass
            ``{}`` to disable EE-pose recording.

    Returns:
        Configured RecorderManagerBaseCfg instance
    """
    # No fallback: every robot cfg must declare its EE bodies explicitly.
    if ee_recorder_bodies is None:
        raise ValueError(
            "ee_recorder_bodies is required. Declare the label on the robot cfg, "
            "e.g. DroidCfg.ee_recorder_bodies = {'ee_pose': 'base_link'}; use {} "
            "to disable EE-pose recording. See docs/robots.md."
        )

    # Create base config
    config = BaseRecorderManagerCfg(
        export_in_record_pre_reset=False,
        dataset_export_dir_path=export_dir,
        dataset_filename=filename
    )

    # One EE-pose recorder per channel. RecorderManager._prepare_terms iterates
    # the cfg instance __dict__, so setattr-attached terms are instantiated.
    for record_key, body_label in ee_recorder_bodies.items():
        robot_cfg_name, ee_body_name = resolve_ee_body_label(body_label)
        setattr(
            config,
            f"record_{record_key}",
            PostStepEndEffectorPoseRecorderCfg(
                record_key=record_key,
                ee_body_name=ee_body_name,
                robot_cfg_name=robot_cfg_name,
            ),
        )

    # Conditionally add policy observations
    if include_policy_observations:
        config.record_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()

    # Conditionally add subtask tracking
    if include_subtask_tracking:
        config.record_subtask_completion = SubtaskCompletionRecorderCfg()

    return config

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum configuration."""

@configclass
class RobolabDefaultEnvCfg(ManagerBasedRLEnvCfg):
    observations = None
    actions = None
    rewards = None
    commands = None
    events = None
    curriculum = None
    recorders = None
    rerender_on_reset: bool = True
    seed: int | None = 0
    subtasks: list[dict[str, dict]] | None = None
    # Set from the robot cfg's mandatory ee_recorder_bodies label (HDF5 channel
    # name -> EE body name). No default; env generation fails if the robot cfg
    # omits the label. See docs/robots.md.
    ee_recorder_bodies: dict[str, str] | None = None

    def __post_init__(self):
        if self.observations is None:
            self.observations = ObservationCfg()
        if self.actions is None:
            self.actions = ActionCfg()
        if self.rewards is None:
            self.rewards = RewardsCfg()
        if self.commands is None:
            self.commands = CommandsCfg()
        if self.events is None:
            self.events = BaseEventCfg()
        if self.curriculum is None:
            self.curriculum = CurriculumCfg()
        if self.recorders is None:
            # Determine recorder configuration based on flags
            enable_subtask_tracking = robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING and self.subtasks is not None
            enable_policy_observations = robolab.constants.RECORD_IMAGE_DATA

            # Log configuration
            print(f"[INFO] Subtask progress checking {'ON' if enable_subtask_tracking else 'OFF'}")
            print(f"[INFO] Image observations recording {'ON' if enable_policy_observations else 'OFF'}")

            # Create recorder configuration
            self.recorders = create_recorder_config(
                include_policy_observations=enable_policy_observations,
                include_subtask_tracking=enable_subtask_tracking,
                export_dir=get_output_dir(),
                filename="data.hdf5",
                ee_recorder_bodies=self.ee_recorder_bodies,
            )

        self.viewer.cam_prim_path = "/OmniverseKit_Persp"
        self.viewer.eye = (1.5, 0.0, 1.0)
        self.viewer.lookat = (0.2, 0.0, 0.0)
        self.viewer.resolution = (1280, 720)
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 8
        self.scene.env_spacing = 2.0
        self.sim.use_fabric = True

        # Isaac Lab 3 moved the backend config from sim.physx to sim.physics.
        physx = getattr(self.sim, "physx", None)
        if physx is None and hasattr(self.sim, "physics"):
            if self.sim.physics is None:
                from isaaclab_physx.physics import PhysxCfg

                self.sim.physics = PhysxCfg()
            physx = self.sim.physics
        physx_settings = {
            "gpu_temp_buffer_capacity": 2**30,
            "gpu_heap_capacity": 2**30,
            "gpu_collision_stack_size": 2**30,
            "enable_ccd": True,
            "contact_offset": 0.02,
            "rest_offset": 0.01,
            "num_position_iterations": 32,        # IsaacLab 2.2
            "num_velocity_iterations": 1,         # IsaacLab 2.2
            "max_position_iteration_count": 32,   # IsaacLab 2.3
            "max_velocity_iteration_count": 1,    # IsaacLab 2.3
            "bounce_threshold_velocity": 0.2,
            "max_depenetration_velocity": 100.0,
            "solver_type": 1,
            "num_threads": 4,
            "relaxation": 0.75,
            "warm_start": 0.4,
            "shape_collision_distance": 0.0,
            "shape_collision_margin": 0.0,
        }
        if physx is not None:
            for attr_name, value in physx_settings.items():
                if hasattr(physx, attr_name):
                    setattr(physx, attr_name, value)
