# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Policy episode runner for RoboLab.

This module contains the run_episode function that executes policy-controlled
episodes using various policy backends (pi0, gr00t, dreamzero, molmo, openvla, etc.).

Supports multi-env: one PolicyClient per env, per-env video writers,
actions inferred per active env and stacked for env.step().
"""

import os
import re
import time
from collections import defaultdict

import cv2
import torch
from tqdm import tqdm

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


def run_episode(env, env_cfg, episode, headless=False, save_videos=True, video_mode="all", remote_host="localhost", remote_port="8000"):
    """Run a policy-controlled episode across all parallel envs.

    Args:
        env: The environment instance (RobolabEnv with num_envs >= 1)
        env_cfg: Environment configuration with policy attribute
        episode: Run index (each run produces num_envs episodes)
        headless: If True, don't display video
        save_videos: If True, save per-env episode videos
        video_mode: Which videos to save: 'all', 'viewport', 'sensor', or 'none'
        remote_host: Host for policy server
        remote_port: Port for policy server

    Returns:
        tuple: (env_results, subtask_status, timing)
            env_results: per-env dicts with {env_id, success, step}
            subtask_status: list of per-step subtask info dicts
            timing: dict with wall-clock timing breakdown
    """
    timer = TimingStats()
    # Dynamically choose policy backend
    backend = getattr(env_cfg, "policy", "pi0").lower()

    if backend in ("pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05"):
        from robolab.inference.pi0_family import Pi0DroidJointposClient as PolicyClient
    elif "gr00t" in backend:
        from robolab.inference.gr00t import GR00TDroidJointposClient as PolicyClient
    elif backend == "dreamzero":
        from robolab.inference.dreamzero import DreamZeroClient as PolicyClient
    elif backend == "molmo":
        from robolab.inference.droid_molmo import MolmoActClient as PolicyClient
    elif backend == "openvla":
        from robolab.inference.openvla import OpenVLAClient as PolicyClient
    elif backend == "openvla_oft":
        from robolab.inference.openvla_oft import OpenVLAOFTClient as PolicyClient
    else:
        raise ValueError(
            f"Unsupported policy '{backend}'. Choose 'pi0', 'pi0_fast', 'pi05', 'paligemma', 'paligemma_fast', 'gr00t', 'dreamzero', 'molmo', 'openvla', 'openvla_oft'"
        )

    obs, _ = env.reset()
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt) # Hz
    instruction = env_cfg.instruction
    action_dim = 8  # 7 joints + 1 gripper

    subtask_status = []

    # Single client (one WebSocket connection) with per-env chunk state
    client = PolicyClient(remote_host=remote_host, remote_port=remote_port)
    clients = [client] * env.num_envs

    # Set up per-run HDF5 file and per-env demo indices
    if env.recorder_manager is not None and hasattr(env.recorder_manager, 'set_hdf5_file'):
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        for env_id in range(env.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])

    # Setup per-env streaming video writers
    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")
    cleaned_instruction = re.sub(r'[^\w\s]', '', instruction).replace(' ', '_')
    if save_videos:
        video_writers_obs = []
        video_writers_viewport = []
        for env_id in range(env.num_envs):
            suffix = f"_{episode}_env{env_id}" if env.num_envs > 1 else f"_{episode}"
            if save_sensor:
                video_path = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}.mp4")
                video_writers_obs.append(VideoWriter(video_path, video_fps))
            if save_viewport:
                video_path_viewport = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(video_path_viewport, video_fps))

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    actual_steps = 0
    for step in tqdm(range(max_steps)):

        while not timeline.is_playing():
            kit_app.update()

        timer.start("policy_inference")
        # Infer actions for all active (non-frozen) envs
        actions = torch.zeros(env.num_envs, action_dim, device=env.device)
        last_viz = None
        for env_id in env.active_env_ids:
            ret = clients[env_id].infer(obs, instruction, env_id=env_id)
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
        if save_videos:
            timer.start("video_write")
            for env_id in range(env.num_envs):
                if env._frozen_envs[env_id]:
                    continue
                if save_sensor:
                    frame_obs = unpack_image_obs(obs, scale=0.5, env_id=env_id).get("combined_image")
                    video_writers_obs[env_id].write(frame_obs)
                if save_viewport:
                    frame_vp = unpack_viewport_cams(obs, env_id=env_id).get("combined_image")
                    video_writers_viewport[env_id].write(frame_vp)
            timer.stop("video_write")

        actual_steps += 1

        # RobolabEnv freezes terminated envs and exports recordings automatically
        if env.all_terminated:
            break

    if save_videos:
        for vw in video_writers_obs + video_writers_viewport:
            vw.release()

    client.reset()

    timing = timer.to_dict(actual_steps)
    return env.get_env_results(), subtask_status, timing
