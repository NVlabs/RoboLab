# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

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
    record_ee_pose: PostStepEndEffectorPoseRecorderCfg = PostStepEndEffectorPoseRecorderCfg()
    record_bbox: PostStepBBoxRecorderCfg = PostStepBBoxRecorderCfg()
    record_policy_observations: PreStepFlatPolicyObservationsRecorderCfg | None = None
    record_subtask_completion: SubtaskCompletionRecorderCfg | None = None
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL

def create_recorder_config(
    include_policy_observations: bool = False,
    include_subtask_tracking: bool = False,
    export_dir: str | None = None,
    filename: str = "data.hdf5"
) -> RecorderManagerBaseCfg:
    """Factory function to create appropriate recorder configuration.

    Args:
        include_policy_observations: Whether to record policy observations (images, etc.)
        include_subtask_tracking: Whether to track subtask completion
        export_dir: Directory to export data to
        filename: Name of the output file

    Returns:
        Configured RecorderManagerBaseCfg instance
    """
    # Create base config
    config = BaseRecorderManagerCfg(
        export_in_record_pre_reset=False,
        dataset_export_dir_path=export_dir,
        dataset_filename=filename
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
                filename="data.hdf5"
            )

        self.viewer.cam_prim_path = "/OmniverseKit_Persp"
        self.viewer.eye = (1.5, 0.0, 1.0)
        self.viewer.lookat = (0.2, 0.0, 0.0)
        self.viewer.resolution = (1280, 720)
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 8
        self.scene.env_spacing = 2.0
        self.sim.use_fabric = True

        # PhysX settings
        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.sim.physx.enable_ccd = True
        self.sim.physx.contact_offset = 0.02
        self.sim.physx.rest_offset = 0.01
        self.sim.physx.num_position_iterations = 32
        self.sim.physx.num_velocity_iterations = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.max_depenetration_velocity = 100.0
        self.sim.physx.solver_type = 1
        self.sim.physx.num_threads = 4
        self.sim.physx.relaxation = 0.75
        self.sim.physx.warm_start = 0.4
        self.sim.physx.shape_collision_distance = 0.0
        self.sim.physx.shape_collision_margin = 0.0
