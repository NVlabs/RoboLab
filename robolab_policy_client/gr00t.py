# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import io
from typing import Any

import msgpack
import numpy as np
import zmq
from PIL import Image

from robolab.eval.base_client import InferenceClient

# GR00T policy resolution
RESOLUTION = (180, 320)


def quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in XYZ convention."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)


# ==============================================================================
# Minimal GR00T Policy Client (embedded from server_client.py)
# ==============================================================================


class _MsgSerializer:
    """Msgpack serializer with numpy array support."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=_MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=_MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class GR00TPolicyClient:
    """Minimal ZMQ client for GR00T policy server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        api_token: str = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.api_token = api_token
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get action from the policy server."""
        request = {
            "endpoint": "get_action",
            "data": {"observation": observation, "options": None},
        }
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(_MsgSerializer.to_bytes(request))
        message = self.socket.recv()

        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure the GR00T policy server is running.")

        response = _MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")

        return tuple(response)  # (action_dict, info_dict)

    def ping(self) -> bool:
        """Check if the server is reachable."""
        try:
            request = {"endpoint": "ping"}
            if self.api_token:
                request["api_token"] = self.api_token
            self.socket.send(_MsgSerializer.to_bytes(request))
            self.socket.recv()
            return True
        except zmq.error.ZMQError:
            return False

    def close(self) -> None:
        try:
            self.socket.close()
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass

    def __del__(self):
        self.close()


# ==============================================================================
# Image utilities
# ==============================================================================


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Resizes images to target size with padding to preserve aspect ratio."""
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    """Resize single image with padding."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.array(zero_image)


# ==============================================================================
# GR00T Inference Client
# ==============================================================================


class GR00TDroidJointposClient(InferenceClient):
    """Inference client for GR00T policy on DROID with joint position action space."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 5555,
        open_loop_horizon: int = 10,
        api_token: str = None,
    ) -> None:
        super().__init__()
        self.open_loop_horizon = int(open_loop_horizon)
        print(f"[{self.__class__.__name__}] Connecting to GR00T policy server at {remote_host}:{remote_port}...")
        self.client = GR00TPolicyClient(host=remote_host, port=remote_port, api_token=api_token)
        print(f"[{self.__class__.__name__}] Connected to GR00T policy server.")

    # ---- required hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        external_image = raw_obs["image_obs"]["external_cam"][env_id].clone().detach().cpu().numpy()
        wrist_image = raw_obs["image_obs"]["wrist_cam"][env_id].clone().detach().cpu().numpy()

        robot_state = raw_obs["proprio_obs"]
        joint_position = robot_state["arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy()

        eef_position = robot_state["ee_pos"][env_id].clone().detach().cpu().numpy()
        eef_quat = robot_state["ee_quat"][env_id].clone().detach().cpu().numpy()
        eef_euler = quat_to_euler_xyz(eef_quat)

        return {
            "external_image": external_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
            "eef_position": eef_position,
            "eef_euler": eef_euler,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        ext_image = resize_with_pad(extracted_obs["external_image"], RESOLUTION[0], RESOLUTION[1])
        wrist_image = resize_with_pad(extracted_obs["wrist_image"], RESOLUTION[0], RESOLUTION[1])
        return {
            "video.exterior_image_1_left": ext_image[None, None, ...],  # [1, 1, H, W, C]
            "video.wrist_image_left": wrist_image[None, None, ...],  # [1, 1, H, W, C]
            "state.eef_position": extracted_obs["eef_position"][None, None, ...],  # [1, 1, 3]
            "state.eef_rotation": extracted_obs["eef_euler"][None, None, ...],  # [1, 1, 3]
            "state.joint_position": extracted_obs["joint_position"][None, None, ...].astype(np.float32),
            "state.gripper_position": extracted_obs["gripper_position"][None, None, ...].astype(np.float32),
            "annotation.language.language_instruction": [instruction],
            "annotation.language.language_instruction_2": [instruction],
            "annotation.language.language_instruction_3": [instruction],
        }

    def _query_server(self, request: dict) -> tuple:
        return self.client.get_action(request)

    def _unpack_response(self, response: tuple) -> np.ndarray:
        action_dict = response[0]
        joint_action = action_dict["action.joint_position"][0]  # [N, 7]
        gripper_action = action_dict["action.gripper_position"][0]  # [N, 1]
        return np.concatenate([joint_action, gripper_action], axis=1)  # [N, 8]

    # ---- optional hooks -----------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        ext_img = resize_with_pad(extracted_obs["external_image"], RESOLUTION[0], RESOLUTION[1])
        wrist_img = resize_with_pad(extracted_obs["wrist_image"], RESOLUTION[0], RESOLUTION[1])
        return np.concatenate([ext_img, wrist_img], axis=1)

    def close(self) -> None:
        self.client.close()


if __name__ == "__main__":
    import time

    import torch

    client = GR00TDroidJointposClient(
        remote_host="localhost",
        remote_port=5555,
        open_loop_horizon=10,
    )

    fake_obs = {
        "image_obs": {
            "external_cam": torch.zeros((1, 480, 640, 3), dtype=torch.uint8),
            "wrist_cam": torch.zeros((1, 480, 640, 3), dtype=torch.uint8),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.zeros((1, 7), dtype=torch.float32),
            "gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
            "ee_pos": torch.zeros((1, 3), dtype=torch.float32),
            "ee_quat": torch.zeros((1, 4), dtype=torch.float32),
        },
    }
    fake_instruction = "pick up the object"

    start = time.time()
    client.infer(fake_obs, fake_instruction)  # warm up
    num = 20
    for _ in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(f"Action shape: {ret['action'].shape}")
    end = time.time()

    print(f"Average inference time: {(end - start) / num:.4f}s")
