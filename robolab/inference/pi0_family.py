# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from PIL import Image

from .base_client import InferenceClient


class Pi0DroidJointposClient(InferenceClient):
    def __init__(self,
                remote_host:str = "localhost",
                remote_port:int = 8000,
                open_loop_horizon:int = 8,
                 ) -> None:
        # Open a single websocket connection shared across all env_ids
        print(f"[{self.__class__.__name__}] Awaiting for server on {remote_host}:{remote_port} to be ready...")
        self.client = websocket_client_policy.WebsocketClientPolicy(
            remote_host, remote_port
        )
        print(f"[{self.__class__.__name__}] Server on {remote_host}:{remote_port} is ready.")

        self.open_loop_horizon = open_loop_horizon
        # Per-env action chunk state so a single client can serve multiple envs
        self._env_chunk: dict[int, object] = {}
        self._env_counter: dict[int, int] = {}

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def reset(self):
        self._env_chunk.clear()
        self._env_counter.clear()

    def infer(self, obs: dict, instruction: str, *, env_id: int = 0) -> dict:
        """
        Infer the next action from the policy in a server-client setup.
        Per-env action chunk state is tracked by env_id so a single client
        instance can correctly serve multiple parallel environments.
        """
        curr_obs = self._extract_observation(obs, env_id=env_id)

        counter = self._env_counter.get(env_id, 0)
        chunk = self._env_chunk.get(env_id, None)

        if counter == 0 or counter >= self.open_loop_horizon or chunk is None:
            counter = 0
            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    curr_obs["right_image"], 224, 224
                ),
                "observation/wrist_image_left": image_tools.resize_with_pad(
                    curr_obs["wrist_image"], 224, 224
                ),
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            chunk = self.client.infer(request_data)["actions"]
            self._env_chunk[env_id] = chunk

        action = chunk[counter]
        self._env_counter[env_id] = counter + 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        # print(f"joint action: {action[:7]} gripper action: {action[-1]:.2f}")

        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        both = np.concatenate([img1, img2], axis=1)

        return {"action": action, "viz": both}

    def _extract_observation(self, obs_dict, *, env_id=0, save_to_disk=False):
        # Assign images
        right_image = obs_dict["image_obs"]["external_cam"][env_id].clone().detach().cpu().numpy()
        wrist_image = obs_dict["image_obs"]["wrist_cam"][env_id].clone().detach().cpu().numpy()

        # Capture proprioceptive state
        robot_state = obs_dict["proprio_obs"]
        joint_position = robot_state["arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

if __name__ == "__main__":
    import torch
    import tyro
    args = tyro.cli(Pi0DroidJointposClient.Args)
    client = Pi0DroidJointposClient(args)
    fake_obs = {
        "splat": {
            "right_cam": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_cam": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "policy": {
            "arm_joint_pos": torch.zeros((7,), dtype=torch.float32),
            "gripper_pos": torch.zeros((1,), dtype=torch.float32),

        },
    }
    fake_instruction = "pick up the object"

    import time

    start = time.time()
    client.infer(fake_obs, fake_instruction) # warm up
    num = 20
    for i in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
