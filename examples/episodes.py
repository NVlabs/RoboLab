# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode running utilities for RoboLab examples.

This module contains utility functions for running different types of episodes:
- run_gripper_toggle_episode: Test gripper toggling
- run_prerecorded_episode: Replay from numpy file
- run_prerecorded_episode_hdf5: Replay from HDF5 file
- run_empty_episode: Run with random actions (for testing)

Note: For policy-controlled episodes, see policy/episode.py
"""

import os
import re

import cv2
import numpy as np
import torch
from isaaclab.envs.utils.spaces import sample_space
from tqdm import tqdm

from robolab.constants import PACKAGE_DIR, get_output_dir
from robolab.core.logging.results import extract_initial_state_info, extract_subtask_info
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.utils.video_utils import VideoWriter


def run_gripper_toggle_episode(env, env_cfg=None, *, save_videos=True, video_mode="all",
                               headless=False, num_steps=100, toggle_every=5, settle_steps=0):
    """Toggle the gripper open/closed every `toggle_every` steps while holding the
    arm joints fixed.

    Video saving mirrors ``robolab.eval.episode.run_episode``: per-env writers,
    fps derived from ``env_cfg.sim``, sensor + viewport streams selectable via
    ``video_mode`` ("all" / "sensor" / "viewport" / "none"), files named after
    the task instruction and dropped under ``get_output_dir()``.
    """
    robot = env.scene["robot"]
    obs, _ = env.reset()

    if settle_steps > 0:
        current_joint_pos = robot.data.joint_pos[0, :7]
        gripper_action = torch.tensor([0.785398163], device=env.device)
        actions = torch.cat([current_joint_pos, gripper_action]).unsqueeze(0)
        for _ in tqdm(range(settle_steps), desc="Settling scene"):
            obs, _, _, _, _ = env.step(actions)

    instruction = getattr(env_cfg, "instruction", None) or "gripper_toggle"
    if isinstance(instruction, dict):
        instruction = instruction.get("default", "gripper_toggle")
    cleaned_instruction = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")

    if env_cfg is not None:
        video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    else:
        video_fps = 15

    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")

    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_env{env_id}" if env.num_envs > 1 else ""
            if save_sensor:
                p = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}.mp4")
                video_writers_obs.append(VideoWriter(p, video_fps))
            if save_viewport:
                p = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(p, video_fps))

    toggle_gripper = False
    subtask_status = []
    try:
        for count in tqdm(range(num_steps)):
            if count % toggle_every == 0:
                toggle_gripper = not toggle_gripper
                print(f"[Step {count:04d}] Gripper state: {'open' if toggle_gripper else 'closed'}")

            current_joint_pos = robot.data.joint_pos[0, :7]
            gripper_width = 0.0 if toggle_gripper else 0.785398163
            gripper_action = torch.tensor([gripper_width], device=env.device)
            actions = torch.cat([current_joint_pos, gripper_action]).unsqueeze(0)

            obs, _, term, trunc, info = env.step(actions)

            if save_videos:
                for env_id in range(env.num_envs):
                    if save_sensor:
                        frame = unpack_image_obs(obs, scale=0.5, env_id=env_id).get("combined_image")
                        if frame is not None:
                            video_writers_obs[env_id].write(frame)
                    if save_viewport:
                        frame_vp = unpack_viewport_cams(obs, env_id=env_id).get("combined_image")
                        if frame_vp is not None:
                            video_writers_viewport[env_id].write(frame_vp)

            if not headless:
                viz = unpack_image_obs(obs).get("combined_image")
                if viz is not None:
                    cv2.imshow("camera", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
    finally:
        for vw in video_writers_obs + video_writers_viewport:
            try:
                vw.release()
            except Exception:
                pass

    return True, subtask_status


def run_prerecorded_episode(env, episode, save_videos=True, headless=False):

    obs, _ = env.reset()
    data = np.load(os.path.join(PACKAGE_DIR, 'fake_data', 'actions.npz'))
    actions = data.get('arr_0')
    max_steps = len(actions)

    if save_videos:
        video_path = os.path.join(get_output_dir(), f"video_{episode}.mp4")
        video_writer = VideoWriter(video_path, fps=15)

    for i in tqdm(range(max_steps)):
        action = actions[i]
        print(f"gripper: {action[-1]}")
        action = torch.tensor(action)[None]

        obs, _, term, trunc, _ = env.step(action)

        # Generate video
        combined_image = unpack_image_obs(obs).get("combined_image")
        if save_videos:
            video_writer.write(combined_image)
        if not headless:
            cv2.imshow("camera", cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if term or trunc:
            break

    if save_videos:
        video_writer.release()


def run_prerecorded_episode_hdf5(env, hdf5_path: str, episode=0, save_videos=True, headless=False):

    obs, _ = env.reset()

    # Set up per-run HDF5 file and per-env demo indices
    if env.recorder_manager is not None and hasattr(env.recorder_manager, 'set_hdf5_file'):
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        for env_id in range(env.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])

    from robolab.core.utils.file_utils import load_hdf5_episode_data
    print(f"Loading actions from {hdf5_path} for episode {episode}")
    actions = load_hdf5_episode_data(hdf5_path, episode, 'actions')

    max_steps = len(actions)

    if save_videos:
        video_writers = []
        for env_id in range(env.num_envs):
            if env.num_envs == 1:
                video_path = os.path.join(get_output_dir(), f"video_{episode}.mp4")
            else:
                video_path = os.path.join(get_output_dir(), f"video_{episode}_env{env_id}.mp4")
            video_writers.append(VideoWriter(video_path, fps=15))

    subtask_status = []

    for i in tqdm(range(max_steps+10)):
        action = actions[min(i, len(actions)-1)]
        # Repeat action for multiple environments
        action = torch.tensor(action).unsqueeze(0).repeat(env.num_envs, 1)

        obs, _, term, trunc, info = env.step(action)

        status = extract_subtask_info(info)
        if status.get('status') != 0:
            print(f"status: {status}")
        subtask_status.append(status)

        if save_videos:
            for env_id in range(env.num_envs):
                combined_image = unpack_image_obs(obs, env_id=env_id).get("combined_image")
                video_writers[env_id].write(combined_image)
        if not headless:
            combined_image = unpack_image_obs(obs, env_id=0).get("combined_image")
            cv2.imshow("camera", cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # RobolabEnv freezes terminated envs and exports recordings automatically
        if env.all_terminated:
            break

    if save_videos:
        for vw in video_writers:
            vw.release()

    # Get per-env results from the env (success/truncated tracking is in RobolabEnv)
    return env.get_env_results(), subtask_status


def run_empty_episode(env, env_cfg, num_envs, num_steps=50, episode=0, save_videos=False, save_image=False):
    obs, _ = env.reset()
    success = False
    subtask_status = []
    init_state_poses = {}
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt) # Hz

    if save_videos:
        video_path = os.path.join(get_output_dir(), f"empty_{episode}_numsteps{num_steps}.mp4")
        video_writer = VideoWriter(video_path, fps=video_fps)

    last_frame = None
    init_state_data = None
    for i in tqdm(range(num_steps)):

        actions = sample_space(env.single_action_space, device=env.device, batch_size=num_envs)

        obs, _, term, trunc, info = env.step(actions)
        frame = unpack_image_obs(obs, obs_group_name="image_obs", camera_suffix="_camera").get("over_shoulder_left_camera")
        if save_videos:
            video_writer.write(frame)
        if save_image:
            last_frame = frame

        init_state_data = extract_initial_state_info(info)
        status = extract_subtask_info(info)
        subtask_status.append(status)

    for object, values in init_state_data.items():
        init_state_poses[object] = values["root_pose"].squeeze(0).cpu().numpy()

    if save_image and last_frame is not None:
        image_path = os.path.join(get_output_dir(), f"empty_{episode}.png")
        cv2.imwrite(image_path, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))

    if save_videos:
        video_writer.release()

    return success, subtask_status
