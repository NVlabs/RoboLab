# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import os
import time
from typing import Any, Dict, List

import torch
from isaaclab.managers import EventTermCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

"""
This function is  not used; refer to subtask_recorder.py for subtask_state logging via env.step()
"""

def get_state(env, env_id=0, is_relative=True):
    state = dict()
    # articulations
    for asset_name, articulation in env._articulations.items():
        asset_state = dict()
        pos = articulation.data.root_pos_w[env_id].clone()
        quat = articulation.data.root_quat_w[env_id].clone()
        asset_state["pose"] = torch.cat([pos, quat], dim=0)
        if is_relative:
            asset_state["pose"][:3] -= env.env_origins[env_id]
        asset_state["velocity"] = articulation.data.root_vel_w[env_id].clone()
        asset_state["joint_names"] = articulation.data.joint_names
        asset_state["joint_position"] = articulation.data.joint_pos[env_id].clone()
        asset_state["joint_velocity"] = articulation.data.joint_vel[env_id].clone()
        asset_state["type"] = "articulation"
        state[asset_name] = asset_state
    # deformable objects
    for asset_name, deformable_object in env._deformable_objects.items():
        asset_state = dict()
        asset_state["nodal_position"] = deformable_object.data.nodal_pos_w[env_id].clone()
        if is_relative:
            asset_state["nodal_position"][:, :3] -= env.env_origins[env_id]
        asset_state["nodal_velocity"] = deformable_object.data.nodal_vel_w[env_id].clone()
        asset_state["type"] = "deformable"
        state[asset_name] = asset_state
    # rigid objects
    state["rigid_object"] = dict()
    for asset_name, rigid_object in env._rigid_objects.items():
        asset_state = dict()
        pos = rigid_object.data.root_pos_w[env_id].clone()
        quat = rigid_object.data.root_quat_w[env_id].clone()
        asset_state["pose"] = torch.cat([pos, quat], dim=0)
        if is_relative:
            asset_state["pose"][:3] -= env.env_origins[env_id]
        asset_state["velocity"] = rigid_object.data.root_vel_w[env_id].clone()
        asset_state["type"] = "rigid_object"
        state[asset_name] = asset_state
    return state

def convert_tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy().tolist()
    elif isinstance(tensor, dict):
        return {k: convert_tensor_to_list(v) for k, v in tensor.items()}
    elif isinstance(tensor, list):
        return [convert_tensor_to_list(v) for v in tensor]
    else:
        return tensor

def state_logging(
    env,
    env_ids: List[int] = None,
    file_name: str = "states.json",
    **kwargs
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Logs state of all objects in environment(s) to JSON file
    Format: {env0: [{time: value, state: value}, ...], env1: [...]}

    Args:
        env: RL environment instance
        env_ids: List of environment IDs to log (None = all)
        file_path: Output JSON file path
    """
    try:
        file_path = os.path.join(env.output_dir, file_name)
        # Load existing log data
        with open(file_path, 'r') as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log_data = {}

    # Get current timestamp
    current_time = env.elapsed_time if hasattr(env, 'elapsed_time') else time.time()

    # Process all environments if none specified
    if env_ids is None:
        env_ids = list(range(env.num_envs))

    for env_id in env_ids:
        env_key = f"env_{env_id}"

        # Get world state
        state = convert_tensor_to_list(get_state(env.scene, env_id))
        # # Get complete environment state
        env_state = {
            "state": state,
            # "observations": convert_tensor_to_list(env.obs_buf),
            # "actions": convert_tensor_to_list(env.actions_buf),
        #     "rewards": float(env.reward_buf[env_id]),
        #     "dones": bool(env.reset_buf[env_id]),
        #     "episode_data": {
        #         "success": bool(env.episode_success_buf[env_id]),
        #         "length": int(env.episode_length_buf[env_id])
        #     }
        }

        # Initialize environment log if needed
        if env_key not in log_data:
            log_data[env_key] = []

        # Append new entry
        log_data[env_key].append({
            "time": current_time,
            "state": env_state
        })

    # Save updated log
    with open(file_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    return log_data

@configclass
class StateLoggingEventCfg:
    state_logging_event = EventTermCfg(
        func=state_logging,
        mode="interval",
        interval_range_s=(0.1, 0.1),  # Log every 0.1 seconds
        is_global_time=True,           # Use real-world time
        params={
            "file_name": "states.json",
        }
    )