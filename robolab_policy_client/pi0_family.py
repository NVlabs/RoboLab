# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging

import numpy as np
from openpi_client import image_tools, websocket_client_policy

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)


class Pi0DroidJointposClient(InferenceClient):
    # Per-variant action horizons. One Pi0 server class serves multiple trained
    # variants; each has its own training-time action_horizon. Callers pass
    # ``policy_variant`` to select the right default, or override directly via
    # ``open_loop_horizon``.
    DEFAULT_HORIZONS: dict[str, int] = {
        "pi0": 10,
        "pi0_fast": 10,
        "paligemma": 10,
        "paligemma_fast": 10,
        "pi05": 15,
    }
    FALLBACK_HORIZON: int = 15

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int | None = None,
        remote_uri: str | None = None,
        policy_variant: str = "pi05",
    ) -> None:
        super().__init__()
        if open_loop_horizon is None:
            open_loop_horizon = self.DEFAULT_HORIZONS.get(policy_variant, self.FALLBACK_HORIZON)
        self.open_loop_horizon = int(open_loop_horizon)
        self.policy_variant = policy_variant
        self._remote_uri = remote_uri
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._display = remote_uri if remote_uri is not None else f"{remote_host}:{remote_port}"

        print(f"[{self.__class__.__name__}] Awaiting for server on {self._display} to be ready...")
        self.client = self._connect()
        print(f"[{self.__class__.__name__}] Connected to {self._display}.")

    def _connect(self):
        if self._remote_uri is not None:
            return websocket_client_policy.WebsocketClientPolicy(self._remote_uri)
        return websocket_client_policy.WebsocketClientPolicy(self._remote_host, self._remote_port)

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        """Call server, reconnecting up to ``max_retries`` times on connection drop."""
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
                    self.__class__.__name__, e, attempt + 1, max_retries,
                )
                self.client = self._connect()
                # Flush chunk cache so all envs re-request on next step
                self._chunks.clear()
                self._counters.clear()

    # ---- required hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        right_image = raw_obs["image_obs"]["external_cam"][env_id].clone().detach().cpu().numpy()
        wrist_image = raw_obs["image_obs"]["wrist_cam"][env_id].clone().detach().cpu().numpy()

        robot_state = raw_obs["proprio_obs"]
        joint_position = robot_state["arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy()

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                extracted_obs["right_image"], 224, 224
            ),
            "observation/wrist_image_left": image_tools.resize_with_pad(
                extracted_obs["wrist_image"], 224, 224
            ),
            "observation/joint_position": extracted_obs["joint_position"],
            "observation/gripper_position": extracted_obs["gripper_position"],
            "prompt": instruction,
        }

    def _query_server(self, request: dict) -> dict:
        return self._infer_with_retry(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        return np.asarray(response["actions"])

    # ---- optional hooks -----------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        img1 = image_tools.resize_with_pad(extracted_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(extracted_obs["wrist_image"], 224, 224)
        return np.concatenate([img1, img2], axis=1)


if __name__ == "__main__":
    import time

    import torch

    client = Pi0DroidJointposClient()
    fake_obs = {
        "image_obs": {
            "external_cam": [torch.zeros((224, 224, 3), dtype=torch.uint8)],
            "wrist_cam": [torch.zeros((224, 224, 3), dtype=torch.uint8)],
        },
        "proprio_obs": {
            "arm_joint_pos": torch.zeros((1, 7), dtype=torch.float32),
            "gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
        },
    }
    fake_instruction = "pick up the object"

    start = time.time()
    client.infer(fake_obs, fake_instruction)  # warm up
    num = 20
    for _ in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
