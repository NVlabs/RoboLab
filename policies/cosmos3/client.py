# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
import torch
import torch.nn.functional as F

from openpi_client import image_tools, websocket_client_policy

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)


class Cosmos3Client(InferenceClient):
    """ """

    IMAGE_W = 640
    IMAGE_H = 360
    OPEN_LOOP_HORIZON = 32

    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000):
        """ """
        super().__init__()
        self._remote_host = remote_host
        self._remote_port = remote_port

        self._image_w = self.IMAGE_W
        self._image_h = self.IMAGE_H

        self.open_loop_horizon = self.OPEN_LOOP_HORIZON

        display = f"{self._remote_host}:{self._remote_port}"
        print(f"[{self.__class__.__name__}] Awaiting for server on {display} to be ready...")
        self.client = self._connect()
        print(f"[{self.__class__.__name__}] Connected to {display}.")

    def _connect(self) -> websocket_client_policy.WebsocketClientPolicy:
        """ """
        return websocket_client_policy.WebsocketClientPolicy(self._remote_host, self._remote_port)

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        """ """
        import websockets.exceptions

        for attempt in range(max_retries):
            try:
                return self.client.infer(request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as e:
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "[%s] Connection lost (%s), reconnecting (attempt %d/%d)...",
                    self.__class__.__name__,
                    e,
                    attempt + 1,
                    max_retries,
                )
                self.client = self._connect()
                # Flush chunk cache so all envs re-request on next step.
                self._chunks.clear()
                self._counters.clear()

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        """ """
        left_image = raw_obs["image_obs"]["over_shoulder_left_camera"][env_id].cpu().numpy()
        left_image = image_tools.resize_with_pad(left_image, self._image_h, self._image_w)
        right_image = raw_obs["image_obs"]["over_shoulder_right_camera"][env_id].cpu().numpy()
        right_image = image_tools.resize_with_pad(right_image, self._image_h, self._image_w)
        wrist_image = raw_obs["image_obs"]["wrist_cam"][env_id].cpu().numpy()
        wrist_image = image_tools.resize_with_pad(wrist_image, self._image_h, self._image_w)

        joint_position = raw_obs["proprio_obs"]["arm_joint_pos"][env_id].cpu().numpy()
        gripper_position = raw_obs["proprio_obs"]["gripper_pos"][env_id].cpu().numpy()

        return {
            "left_image": left_image,
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        """ """
        # Compose wrist, left, and right views into a single frame.
        wrist = extracted_obs["wrist_image"]
        size = (self._image_h // 2, self._image_w // 2)
        left = torch.from_numpy(extracted_obs["left_image"]).permute(2, 0, 1).unsqueeze(0).float()
        left = F.interpolate(left, size=size, mode="bilinear")
        left = left.squeeze(0).permute(1, 2, 0).numpy().astype(wrist.dtype)
        right = torch.from_numpy(extracted_obs["right_image"]).permute(2, 0, 1).unsqueeze(0).float()
        right = F.interpolate(right, size=size, mode="bilinear")
        right = right.squeeze(0).permute(1, 2, 0).numpy().astype(wrist.dtype)
        image = np.concatenate((wrist, np.concatenate((left, right), axis=1)))

        return {
            "observation/image": image,
            "observation/joint_position": extracted_obs["joint_position"],
            "observation/gripper_position": extracted_obs["gripper_position"],
            "prompt": instruction,
        }

    def _query_server(self, request: dict) -> dict:
        """ """
        return self._infer_with_retry(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        """ """
        return np.asarray(response["action"])

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """ """
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        """ """
        left = extracted_obs["left_image"]
        wrist = extracted_obs["wrist_image"]
        right = extracted_obs["right_image"]
        return np.concatenate((left, wrist, right), axis=1)
