# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
from openpi_client import image_tools, websocket_client_policy

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)


class Pi0PiperDualArmClient(InferenceClient):
    """Dual-arm client for the "Double Piper" robot.

    The AgileX/OpenPI server expects ordinary-policy observations in this shape:
    ``state`` = [left_arm(6), right_arm(6)], ``gripper_position`` = [left, right],
    and three RGB images under ``images`` in CHW uint8 layout.

    Server actions arrive as [left_arm(6), left_gripper, right_arm(6), right_gripper].
    RoboLab's Piper env expects [left_arm(6), right_arm(6), left_gripper, right_gripper],
    with gripper commands in meters rather than normalized [0, 1].
    """

    DEFAULT_HORIZONS: dict[str, int] = {
        "pi0": 10,
        "pi0_fast": 10,
        "paligemma": 10,
        "paligemma_fast": 10,
        "pi05": 15,
    }
    FALLBACK_HORIZON: int = 15
    MAX_GRIPPER_OPENING: float = 0.035

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
                self._chunks.clear()
                self._counters.clear()

    # ---- required hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs["image_obs"]
        left_hand_image = image_obs["left_hand_camera"][env_id].clone().detach().cpu().numpy()
        right_hand_image = image_obs["right_hand_camera"][env_id].clone().detach().cpu().numpy()
        first_person_image = image_obs["first_person_camera"][env_id].clone().detach().cpu().numpy()

        robot_state = raw_obs["proprio_obs"]
        left_arm_joint_pos = robot_state["left_arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        right_arm_joint_pos = robot_state["right_arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        left_gripper_pos = robot_state["left_gripper_pos"][env_id].clone().detach().cpu().numpy()
        right_gripper_pos = robot_state["right_gripper_pos"][env_id].clone().detach().cpu().numpy()

        return {
            "left_hand_image": left_hand_image,
            "right_hand_image": right_hand_image,
            "first_person_image": first_person_image,
            "left_arm_joint_pos": left_arm_joint_pos,
            "right_arm_joint_pos": right_arm_joint_pos,
            "left_gripper_pos": left_gripper_pos,
            "right_gripper_pos": right_gripper_pos,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        left_gripper = self._gripper_scalar(extracted_obs["left_gripper_pos"])
        right_gripper = self._gripper_scalar(extracted_obs["right_gripper_pos"])

        return {
            "state": np.concatenate(
                [
                    extracted_obs["left_arm_joint_pos"],
                    extracted_obs["right_arm_joint_pos"],
                ]
            ).astype(np.float32),
            "gripper_position": np.asarray([left_gripper, right_gripper], dtype=np.float32),
            "images": {
                "cam_top": self._to_chw_uint8(extracted_obs["first_person_image"]),
                "cam_left_wrist": self._to_chw_uint8(extracted_obs["left_hand_image"]),
                "cam_right_wrist": self._to_chw_uint8(extracted_obs["right_hand_image"]),
            },
            "prompt": instruction,
        }

    def _query_server(self, request: dict) -> dict:
        return self._infer_with_retry(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        return np.asarray(response["actions"])

    # ---- optional hooks -----------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        # Server: [left_arm(6), left_gripper, right_arm(6), right_gripper]
        # Env:    [left_arm(6), right_arm(6), left_gripper, right_gripper]
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape[-1] != 14:
            raise ValueError(f"Expected Piper action chunks with 14 dims, got shape {chunk.shape}")

        env_chunk = np.empty_like(chunk)
        env_chunk[..., 0:6] = chunk[..., 0:6]
        env_chunk[..., 6:12] = chunk[..., 7:13]
        env_chunk[..., 12] = np.clip(chunk[..., 6], 0.0, 1.0) * self.MAX_GRIPPER_OPENING
        env_chunk[..., 13] = np.clip(chunk[..., 13], 0.0, 1.0) * self.MAX_GRIPPER_OPENING
        return env_chunk

    @staticmethod
    def _to_chw_uint8(image: np.ndarray) -> np.ndarray:
        """Convert IsaacLab HWC RGB/RGBA images to the AgileX server's CHW uint8 RGB layout."""
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")
        if image.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HWC RGB/RGBA image, got shape {image.shape}")
        image = image[..., :3]
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size and np.nanmax(image) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image.transpose(2, 0, 1))

    @staticmethod
    def _gripper_scalar(gripper_obs: np.ndarray) -> float:
        """Collapse one side's finger observations to a single normalized opening ratio."""
        value = np.max(np.abs(np.asarray(gripper_obs, dtype=np.float32)))
        return float(np.clip(value, 0.0, 1.0))

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        img1 = image_tools.resize_with_pad(extracted_obs["first_person_image"], 224, 224)
        img2 = image_tools.resize_with_pad(extracted_obs["left_hand_image"], 224, 224)
        img3 = image_tools.resize_with_pad(extracted_obs["right_hand_image"], 224, 224)
        return np.concatenate([img1, img2, img3], axis=1)


if __name__ == "__main__":
    import time

    import torch

    client = Pi0PiperDualArmClient()
    fake_obs = {
        "image_obs": {
            "left_hand_camera": [torch.zeros((480, 640, 3), dtype=torch.uint8)],
            "right_hand_camera": [torch.zeros((480, 640, 3), dtype=torch.uint8)],
            "first_person_camera": [torch.zeros((480, 640, 3), dtype=torch.uint8)],
        },
        "proprio_obs": {
            "left_arm_joint_pos": torch.zeros((1, 6), dtype=torch.float32),
            "right_arm_joint_pos": torch.zeros((1, 6), dtype=torch.float32),
            "left_gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
            "right_gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
        },
    }
    fake_instruction = "pick up the rubiks cube and place it in the box"

    start = time.time()
    client.infer(fake_obs, fake_instruction)  # warm up
    num = 20
    for _ in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
