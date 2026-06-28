# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any

import cv2
import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

from robolab.eval.base_client import InferenceClient

# GR00T DROID images are sent as HWC uint8 at 180x320, then the N1.7 processor
# applies its own SmallestMaxSize/center-crop transforms. Do not letterbox here.
RESOLUTION = (180, 320)

DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("EEF quaternion norm must be non-zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compute_eef_9d(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert RoboLab base-link EEF pose to the GR00T DROID 9D state."""
    rot_robot = quat_wxyz_to_matrix(quat_wxyz)
    rot_mat = rot_robot @ DROID_EEF_ROTATION_CORRECT
    rot6d = rot_mat[:2, :].reshape(6)
    return np.concatenate([np.asarray(position, dtype=np.float64).reshape(3), rot6d]).astype(
        np.float32
    )


# ==============================================================================
# Minimal GR00T Policy Client (compatible with Isaac-GR00T server_client.py)
# ==============================================================================


class _MsgSerializer:
    """msgpack_numpy serializer with an object-dtype ndarray safety boundary."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        default = functools.partial(_MsgSerializer._safe_encode, chain=lambda obj: obj)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        object_hook = functools.partial(_MsgSerializer._safe_decode, chain=lambda obj: obj)
        return msgpack.unpackb(data, object_hook=object_hook, raw=False)

    @staticmethod
    def _safe_encode(obj: Any, chain=None) -> Any:
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError(
                f"Refusing to encode object-dtype ndarray (shape={obj.shape}); "
                "convert to a concrete numeric dtype before sending."
            )
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj: Any, chain=None) -> Any:
        if isinstance(obj, dict):
            nd_val = obj.get(b"nd", obj.get("nd"))
            kind_val = obj.get(b"kind", obj.get("kind"))
            if nd_val and kind_val in (b"O", "O"):
                raise ValueError("Refusing to decode object-dtype ndarray payload.")
        return mnp.decode(obj, chain=chain)


class GR00TPolicyClient:
    """Minimal ZMQ client for the GR00T policy server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        api_token: str | None = None,
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


def resize_no_pad(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize HWC images to target shape by stretching, without padding or letterbox bars."""
    if images.shape[-3:-1] == (height, width):
        return np.asarray(images, dtype=np.uint8)

    original_shape = images.shape
    flat_images = np.asarray(images, dtype=np.uint8).reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8)
            for image in flat_images
        ]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _get_action_value(action_dict: dict[str, Any], name: str) -> Any:
    prefixed = f"action.{name}"
    if prefixed in action_dict:
        return action_dict[prefixed]
    if name in action_dict:
        return action_dict[name]
    raise KeyError(f"Missing action key {prefixed!r} or {name!r}; keys={sorted(action_dict)}")


def _as_action_chunk(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 2:
        return arr
    if name == "gripper_position" and arr.ndim == 1:
        return arr.reshape(-1, 1)
    raise ValueError(f"Unexpected action shape for {name}: {arr.shape}")


# ==============================================================================
# GR00T Inference Client
# ==============================================================================


class GR00TDroidJointposClient(InferenceClient):
    """Inference client for GR00T N1.7 DROID with joint position actions."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 5555,
        open_loop_horizon: int = 10,
        api_token: str | None = None,
    ) -> None:
        super().__init__()
        self.open_loop_horizon = int(open_loop_horizon)
        print(f"[{self.__class__.__name__}] Connecting to GR00T policy server at {remote_host}:{remote_port}...")
        self.client = GR00TPolicyClient(host=remote_host, port=remote_port, api_token=api_token)
        print(
            f"[{self.__class__.__name__}] Connected; "
            f"open_loop_horizon={self.open_loop_horizon}, resize=no_pad_stretch_area."
        )

    # ---- required hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        external_image = raw_obs["image_obs"]["over_shoulder_left_camera"][env_id].clone().detach().cpu().numpy()
        wrist_image = raw_obs["image_obs"]["wrist_cam"][env_id].clone().detach().cpu().numpy()

        robot_state = raw_obs["proprio_obs"]
        joint_position = robot_state["arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy()

        eef_position = robot_state["ee_pos"][env_id].clone().detach().cpu().numpy()
        eef_quat = robot_state["ee_quat"][env_id].clone().detach().cpu().numpy()

        return {
            "external_image": external_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position.astype(np.float32),
            "gripper_position": gripper_position.astype(np.float32),
            "eef_9d": compute_eef_9d(eef_position, eef_quat),
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        ext_image = resize_no_pad(extracted_obs["external_image"], RESOLUTION[0], RESOLUTION[1])
        wrist_image = resize_no_pad(extracted_obs["wrist_image"], RESOLUTION[0], RESOLUTION[1])
        return {
            "video.exterior_image_1_left": ext_image[None, None, ...].astype(np.uint8),
            "video.wrist_image_left": wrist_image[None, None, ...].astype(np.uint8),
            "state.eef_9d": extracted_obs["eef_9d"][None, None, ...].astype(np.float32),
            "state.joint_position": extracted_obs["joint_position"][None, None, ...].astype(np.float32),
            "state.gripper_position": extracted_obs["gripper_position"][None, None, ...].astype(np.float32),
            "annotation.language.language_instruction": [instruction],
        }

    def _query_server(self, request: dict) -> tuple:
        return self.client.get_action(request)

    def _unpack_response(self, response: tuple) -> np.ndarray:
        action_dict = response[0]
        joint_action = _as_action_chunk(
            _get_action_value(action_dict, "joint_position"),
            name="joint_position",
        )
        gripper_action = _as_action_chunk(
            _get_action_value(action_dict, "gripper_position"),
            name="gripper_position",
        )
        return np.concatenate([joint_action, gripper_action], axis=1)

    # ---- optional hooks -----------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        ext_img = resize_no_pad(extracted_obs["external_image"], RESOLUTION[0], RESOLUTION[1])
        wrist_img = resize_no_pad(extracted_obs["wrist_image"], RESOLUTION[0], RESOLUTION[1])
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
            "over_shoulder_left_camera": torch.zeros((1, 180, 320, 3), dtype=torch.uint8),
            "wrist_cam": torch.zeros((1, 180, 320, 3), dtype=torch.uint8),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.zeros((1, 7)),
            "gripper_pos": torch.zeros((1, 1)),
            "ee_pos": torch.zeros((1, 3)),
            "ee_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        },
    }

    for i in range(3):
        try:
            start = time.time()
            result = client.infer(fake_obs, "pick up the object", env_id=0)
            print(f"Step {i}: action shape={result['action'].shape}, time={time.time() - start:.3f}s")
        except Exception as exc:
            print(f"Error: {exc}")
            break

    client.close()
