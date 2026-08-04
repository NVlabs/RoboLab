# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy episode runner for RoboLab.

This module contains the run_episode function that executes a single
policy-controlled episode given any :class:`InferenceClient` subclass.
The function stays policy-agnostic — concrete clients live under
``policies/<policy>/client.py``.

Supports multi-env: one PolicyClient per env, per-env video writers,
actions inferred per active env and stacked for env.step().
"""

import logging
import os
import re
import time
from collections import defaultdict

import cv2
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TimingStats:
    """Simple timing utility for profiling code sections."""

    def __init__(self):
        self.times = defaultdict(list)
        self._start_times = {}

    def start(self, name: str):
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str):
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.times[name].append(elapsed)
            del self._start_times[name]

    def to_dict(self, num_steps: int) -> dict:
        """Return timing summary as a dict for results logging."""
        d = {}
        for name, times in self.times.items():
            d[f"{name}_s"] = round(sum(times), 3)
            d[f"{name}_avg_ms"] = round(sum(times) / len(times) * 1000, 1) if times else 0
        d["wall_total_s"] = round(sum(sum(t) for t in self.times.values()), 3)
        d["it_per_sec"] = round(num_steps / d["wall_total_s"], 2) if d["wall_total_s"] > 0 else 0
        return d

from robolab.constants import VISUALIZE, get_output_dir
from robolab.core.logging.results import get_all_env_subtask_infos
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.utils.video_utils import VideoWriter
from robolab.core.world.world_state import get_world
from robolab.eval.base_client import InferenceClient


def run_episode(env, env_cfg, episode, client: InferenceClient, *, headless=False, save_videos=True, video_mode="all"):
    """Run a policy-controlled episode across all parallel envs.

    The policy client is constructed by the caller (typically a per-policy
    runner under ``policies/<policy>/run.py``). This function stays
    policy-agnostic.

    Args:
        env: The environment instance (RobolabEnv with num_envs >= 1)
        env_cfg: Environment configuration
        episode: Run index (each run produces num_envs episodes)
        client: Constructed inference client. One connection shared across envs
            with per-env chunk state keyed by ``env_id``.
        headless: If True, don't display video
        save_videos: If True, save per-env episode videos
        video_mode: Which videos to save: 'all', 'viewport', 'sensor', or 'none'

    Returns:
        tuple: (env_results, subtask_status, timing)
            env_results: per-env dicts with {env_id, success, step}
            subtask_status: list of per-step subtask info dicts
            timing: dict with wall-clock timing breakdown
    """
    timer = TimingStats()

    # Keep the historical warm-up reset, but suppress a generated task's
    # expensive reset pre-roll here.  The second reset below is the actual
    # episode reset and is the only one allowed to perform/record setup.
    env._dynamic_setup_enabled = False
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt) # Hz
    # The policy loop can run faster than the configured render cadence.  Write
    # only at that cadence so one second of policy time remains one second of
    # MP4 time (and matches reset pre-roll frames, which are rendered at the
    # same cadence).
    video_write_stride = max(1, round(1 / (env.step_dt * video_fps)))
    instruction = env_cfg.instruction
    # Pull action dim from the env's action manager (IsaacLab canonical),
    # falling back to the gym action space if the manager isn't available.
    action_dim = getattr(
        getattr(env, "action_manager", None),
        "total_action_dim",
        None,
    ) or env.action_space.shape[-1]

    subtask_status = []

    # Setup per-env streaming video writers
    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")
    cleaned_instruction = re.sub(r'[^\w\s]', '', instruction).replace(' ', '_')
    # Define unconditionally so the finally clause below can iterate them either way.
    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_{episode}_env{env_id}" if env.num_envs > 1 else f"_{episode}"
            if save_sensor:
                video_path = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}.mp4")
                video_writers_obs.append(VideoWriter(video_path, video_fps))
            if save_viewport:
                video_path_viewport = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(video_path_viewport, video_fps))

    def write_video_frames(frame_obs, *, skip_frozen: bool = False) -> None:
        """Append a synchronized frame to each enabled per-env video stream."""
        if not save_videos:
            return
        for env_id in range(env.num_envs):
            if skip_frozen and env._frozen_envs[env_id]:
                continue
            if save_sensor:
                sensor_frame = unpack_image_obs(frame_obs, scale=0.5, env_id=env_id).get("combined_image")
                video_writers_obs[env_id].write(sensor_frame)
            if save_viewport:
                viewport_frame = unpack_viewport_cams(frame_obs, env_id=env_id).get("combined_image")
                video_writers_viewport[env_id].write(viewport_frame)

    # The reset event owns setup physics.  It calls this lightweight callback
    # only after it rendered a setup frame; the same writers are then retained
    # for policy frames, yielding one continuous dashboard MP4.
    def capture_setup_frame(reset_env) -> None:
        setup_obs = reset_env.observation_manager.compute(update_history=False)
        write_video_frames(setup_obs)

    record_setup_video = bool(
        save_videos
        and getattr(env_cfg, "record_setup_video", False)
        and (save_sensor or save_viewport)
    )
    env._dynamic_setup_capture = capture_setup_frame if record_setup_video else None
    env._dynamic_setup_enabled = True
    obs, _ = env.reset()
    # Do not retain a callback into closed video writers on later incidental
    # resets (e.g. teardown or a caller's next episode warm-up).
    env._dynamic_setup_capture = None

    # Set up per-run HDF5 file and per-env demo indices only after reset.  The
    # reset pre-roll is intentionally not a policy trajectory/action history.
    if env.recorder_manager is not None and hasattr(env.recorder_manager, 'set_hdf5_file'):
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        for env_id in range(env.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    actual_steps = 0
    try:
        for step in tqdm(range(max_steps)):

            while not timeline.is_playing():
                kit_app.update()

            timer.start("policy_inference")
            # Infer actions for all active (non-frozen) envs in ONE call.
            # Batching-capable clients send a single request for every env
            # needing a replan; the InferenceClient default is a serial
            # loop, so other policies behave exactly as before.
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            last_viz = None
            rets = client.infer_batch(
                obs, instruction, env_ids=list(env.active_env_ids)
            )
            for env_id in sorted(rets):
                ret = rets[env_id]
                actions[env_id] = torch.tensor(ret["action"], device=env.device)
                if env_id == 0 or last_viz is None:
                    last_viz = ret.get("viz")
            timer.stop("policy_inference")

            if not headless and last_viz is not None:
                cv2.imshow(f"{instruction}", cv2.cvtColor(last_viz, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if VISUALIZE:
                get_world(env).visualize()

            timer.start("env_step")
            obs, reward, term, trunc, info = env.step(actions)
            timer.stop("env_step")

            # Collect per-env subtask info (list of dicts, one per env)
            per_env_infos = get_all_env_subtask_infos(env)
            subtask_status.append(per_env_infos)

            # Write per-env video frames (skip frozen envs)
            if save_videos and actual_steps % video_write_stride == 0:
                timer.start("video_write")
                # Keep the old frozen-env behavior for policy frames. Setup
                # frames are written by the reset callback before freezing can
                # occur, so both streams remain continuous and synchronized.
                write_video_frames(obs, skip_frozen=True)
                timer.stop("video_write")

            actual_steps += 1

            # RobolabEnv freezes terminated envs and exports recordings automatically
            if env.all_terminated:
                break
    finally:
        for vw in video_writers_obs + video_writers_viewport:
            try:
                vw.release()
            except Exception:
                logger.exception("Failed to release video writer")
        try:
            client.reset()
        except Exception:
            logger.exception("Failed to reset client after episode")

    timing = timer.to_dict(actual_steps)
    setup_sim = getattr(env, "_dynamic_setup_sim_duration_s", {})
    setup_video = getattr(env, "_dynamic_setup_video_duration_s", {})
    if setup_sim:
        timing["setup_sim_s"] = round(max(setup_sim.values()), 3)
    if setup_video:
        timing["setup_video_s"] = round(max(setup_video.values()), 3)
    return env.get_env_results(), subtask_status, timing
