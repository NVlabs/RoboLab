"""DreamZero VLA client for robolab.

DreamZero is a world model that predicts future observations while predicting actions.
It uses the roboarena WebSocket protocol, similar to OpenPI but with:
  - Support for multiple external cameras (2 for DROID)
  - Temporal frame input (can send multiple frames for temporal context)
  - Session ID tracking for episode-level history
  - Action chunks output (N, 8)

Server launch example:
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
        socket_test_optimized_AR.py --port 5000 --enable-dit-cache --model-path <path/to/checkpoint>

Usage:
    python examples/policy/run_eval.py --policy dreamzero --remote-port 5000 --task BananaInBowlTableTask
"""

import time
import uuid
import logging
import numpy as np
from PIL import Image
import websockets.sync.client

from .base_client import InferenceClient

logger = logging.getLogger(__name__)

# Increase timeouts for DreamZero's longer inference times (world model prediction)
PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600

# Connection safeguards
CONNECT_TIMEOUT_SECS = 300
RECV_TIMEOUT_SECS = 300
MAX_CONNECT_RETRIES = 5
MAX_INFER_RETRIES = 3
RETRY_BACKOFF_BASE_SECS = 2


class MsgPackNumpy:
    """Simple msgpack wrapper with numpy support.

    Mirrors the client_lib.msgpack_numpy from dreamzero.
    """

    def __init__(self):
        import msgpack
        self._msgpack = msgpack

    def pack(self, obj):
        # Don't use strict_types=True - it breaks tuple serialization
        return self._msgpack.packb(obj, default=self._encode_numpy)

    def unpack(self, data):
        return self._msgpack.unpackb(data, object_hook=self._decode_numpy, strict_map_key=False)

    def _encode_numpy(self, obj):
        """Encode numpy arrays and generics for msgpack serialization."""
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind in ("V", "O", "c"):
                raise ValueError(f"Unsupported dtype: {obj.dtype}")
            return {
                b"__ndarray__": True,
                b"data": obj.tobytes(),
                b"dtype": obj.dtype.str,
                b"shape": obj.shape,
            }

        # Handle numpy scalar types
        if isinstance(obj, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": obj.item(),
                b"dtype": obj.dtype.str,
            }

        # Let msgpack handle other types (including tuples, lists, etc.)
        return obj

    def _decode_numpy(self, obj):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj


class DreamZeroClient(InferenceClient):
    """Inference client for DreamZero VLA model.

    DreamZero uses the roboarena WebSocket protocol with:
      - 2 external cameras + 1 wrist camera
      - Image resolution: 180x320 (H x W) for DROID config
      - Joint position action space (7 DoF + gripper)
      - Session ID for episode tracking
      - Action chunks output

    Input observation keys:
      - observation/exterior_image_0_left: (H, W, 3) uint8 - External camera 1
      - observation/exterior_image_1_left: (H, W, 3) uint8 - External camera 2
      - observation/wrist_image_left: (H, W, 3) uint8 - Wrist camera
      - observation/joint_position: (7,) float32 - Joint positions
      - observation/cartesian_position: (6,) float32 - Cartesian pose
      - observation/gripper_position: (1,) float32 - Gripper position
      - prompt: str - Language instruction
      - session_id: str - Unique session/episode ID

    Output:
      - actions: (N, 8) float32 - N actions with 7 joints + 1 gripper
    """

    def __init__(self,
                 remote_host: str = "localhost",
                 remote_port: int = 5000,  # Default DreamZero port
                 open_loop_horizon: int = 24,  # DreamZero uses 24-step action chunks
                 image_height: int = 180,  # DreamZero DROID default
                 image_width: int = 320,   # DreamZero DROID default
                 ) -> None:
        """Initialize DreamZero client.

        Args:
            remote_host: Server hostname
            remote_port: Server port (default 5000 for DreamZero)
            open_loop_horizon: Number of actions to execute before re-querying
            image_height: Target image height for resizing
            image_width: Target image width for resizing
        """
        self.host = remote_host
        self.port = remote_port
        self.open_loop_horizon = open_loop_horizon
        self.image_height = image_height
        self.image_width = image_width

        # Action chunking state
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

        # Session tracking (unique per episode)
        self.session_id = str(uuid.uuid4())

        # MsgPack for numpy serialization
        self._packer = MsgPackNumpy()

        # Connect to server
        self._uri = f"ws://{remote_host}:{remote_port}"
        self._ws = None
        self._connect_with_retries()

    @staticmethod
    def _wait_with_progress(tag: str, label: str, duration: float, interval: float = 5.0):
        """Sleep for *duration* seconds, printing a progress bar to stdout."""
        elapsed = 0.0
        width = 30
        while elapsed < duration:
            step = min(interval, duration - elapsed)
            time.sleep(step)
            elapsed += step
            frac = elapsed / duration
            filled = int(width * frac)
            bar = "=" * filled + "-" * (width - filled)
            mins_left = (duration - elapsed) / 60
            print(f"\r{tag} {label} [{bar}] {frac*100:5.1f}%  {mins_left:.1f}m left", end="", flush=True)
        print()

    def _connect_with_retries(self):
        """Establish WebSocket connection with retries and exponential backoff."""
        tag = f"[{self.__class__.__name__}]"
        print(f"{tag} Connecting to DreamZero server at {self._uri}...")

        for attempt in range(1, MAX_CONNECT_RETRIES + 1):
            if attempt > 1:
                print(f"{tag} Connection attempt {attempt}/{MAX_CONNECT_RETRIES}...")
            try:
                try:
                    self._ws = websockets.sync.client.connect(
                        self._uri,
                        compression=None,
                        max_size=None,
                        open_timeout=CONNECT_TIMEOUT_SECS,
                        ping_interval=PING_INTERVAL_SECS,
                        ping_timeout=PING_TIMEOUT_SECS,
                    )
                except TypeError:
                    # Older websockets (e.g. 11.x bundled with Isaac Sim) lacks ping_interval/ping_timeout
                    self._ws = websockets.sync.client.connect(
                        self._uri,
                        compression=None,
                        max_size=None,
                        open_timeout=CONNECT_TIMEOUT_SECS,
                    )

                self._server_metadata = self._packer.unpack(
                    self._ws.recv(timeout=RECV_TIMEOUT_SECS)
                )
                print(f"{tag} Server metadata: {self._server_metadata}")
                print(f"{tag} Connected successfully.")
                return

            except Exception as e:
                if self._ws is not None:
                    try:
                        self._ws.close()
                    except Exception:
                        pass
                    self._ws = None

                if attempt == MAX_CONNECT_RETRIES:
                    raise ConnectionError(
                        f"{tag} Failed to connect after {MAX_CONNECT_RETRIES} attempts. "
                        f"Last error: {e}"
                    ) from e

                wait = RETRY_BACKOFF_BASE_SECS ** attempt
                logger.warning(
                    "%s Connection attempt %d/%d failed (%s). Retrying in %.1fs...",
                    tag, attempt, MAX_CONNECT_RETRIES, e, wait,
                )
                print(f"{tag} Connection attempt {attempt}/{MAX_CONNECT_RETRIES} failed ({e}).")
                self._wait_with_progress(tag, f"Waiting to retry", wait)

    def _ensure_connected(self):
        """Reconnect if the WebSocket has been closed or lost."""
        try:
            if self._ws is not None and self._ws.socket is not None:
                return
        except Exception:
            pass
        tag = f"[{self.__class__.__name__}]"
        print(f"{tag} Connection lost. Reconnecting...")
        self._connect_with_retries()

    def _send_recv(self, data: bytes, *, timeout: float = RECV_TIMEOUT_SECS) -> bytes:
        """Send packed data and receive response with timeout and auto-reconnect.

        Retries up to MAX_INFER_RETRIES times on transient failures.
        """
        tag = f"[{self.__class__.__name__}]"
        last_exc = None

        for attempt in range(1, MAX_INFER_RETRIES + 1):
            if attempt > 1:
                print(f"{tag} send/recv attempt {attempt}/{MAX_INFER_RETRIES}...")
            try:
                self._ensure_connected()
                assert self._ws is not None
                self._ws.send(data)
                return self._ws.recv(timeout=timeout)
            except Exception as e:
                last_exc = e
                if self._ws is not None:
                    try:
                        self._ws.close()
                    except Exception:
                        pass
                    self._ws = None

                if attempt == MAX_INFER_RETRIES:
                    break

                wait = RETRY_BACKOFF_BASE_SECS * attempt
                logger.warning(
                    "%s send/recv attempt %d/%d failed (%s). Reconnecting in %.1fs...",
                    tag, attempt, MAX_INFER_RETRIES, e, wait,
                )
                print(f"{tag} send/recv attempt {attempt}/{MAX_INFER_RETRIES} failed ({e}).")
                self._wait_with_progress(tag, "Waiting to reconnect", wait)

        raise ConnectionError(
            f"{tag} send/recv failed after {MAX_INFER_RETRIES} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc

    def reset(self):
        """Reset for new episode - generates new session ID and clears action buffer."""
        reset_data = {"endpoint": "reset"}
        self._send_recv(self._packer.pack(reset_data))

        # Reset local state
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())
        print(f"[{self.__class__.__name__}] Reset complete. New session_id: {self.session_id}")

    def infer(self, obs: dict, instruction: str, *, env_id: int = 0) -> dict:
        """Query DreamZero server and return the next action.

        Args:
            obs: Observation dict from robolab environment
            instruction: Natural language instruction
            env_id: Environment index to extract observations from.

        Returns:
            dict with:
                - "action": np.ndarray (8,) - 7 joint positions + 1 gripper
                - "viz": np.ndarray - Combined camera visualization
        """
        curr_obs = self._extract_observation(obs, env_id=env_id)

        # Only query server when action buffer is exhausted
        if (self.actions_from_chunk_completed == 0 or
            self.actions_from_chunk_completed >= self.open_loop_horizon):

            self.actions_from_chunk_completed = 0

            # Build request in DreamZero/roboarena format
            request_data = {
                # Camera images - resize to DreamZero expected resolution
                "observation/exterior_image_0_left": self._resize_image(
                    curr_obs["right_image"], self.image_height, self.image_width
                ),
                "observation/exterior_image_1_left": self._resize_image(
                    curr_obs["right_image"], self.image_height, self.image_width
                ),  # Using same image for both if only one external camera available
                "observation/wrist_image_left": self._resize_image(
                    curr_obs["wrist_image"], self.image_height, self.image_width
                ),
                # Proprioception
                "observation/joint_position": curr_obs["joint_position"],
                "observation/cartesian_position": np.zeros(6, dtype=np.float32),  # Placeholder if not available
                "observation/gripper_position": curr_obs["gripper_position"],
                # Language and session
                "prompt": instruction,
                "session_id": self.session_id,
                # Endpoint marker for server
                "endpoint": "infer",
            }

            # Send request and receive action chunk (with timeout + auto-reconnect)
            response = self._send_recv(self._packer.pack(request_data))

            if isinstance(response, str):
                raise RuntimeError(f"DreamZero server error:\n{response}")

            self.pred_action_chunk = self._packer.unpack(response)

            # Handle different response formats
            if isinstance(self.pred_action_chunk, dict):
                self.pred_action_chunk = self.pred_action_chunk.get("actions", self.pred_action_chunk)

            self.pred_action_chunk = np.asarray(self.pred_action_chunk)

            # Ensure 2D shape
            if self.pred_action_chunk.ndim == 1:
                self.pred_action_chunk = self.pred_action_chunk.reshape(1, -1)

        # Get current action from chunk (copy to make writable)
        assert self.pred_action_chunk is not None
        action = self.pred_action_chunk[self.actions_from_chunk_completed].copy()
        self.actions_from_chunk_completed += 1

        # Ensure 8-dim action (7 joints + gripper)
        if action.size == 7:
            action = np.concatenate([action, np.zeros((1,))])

        # Binarize gripper to {0, 1}
        if action.size >= 8:
            action[-1] = 1.0 if action[-1] > 0.5 else 0.0

        # Build visualization
        viz = self._build_visualization(curr_obs)

        return {"action": action, "viz": viz}

    def visualize(self, request: dict):
        """Return camera views visualization."""
        curr_obs = self._extract_observation(request)
        return self._build_visualization(curr_obs)

    def _extract_observation(self, obs_dict, *, env_id=0, save_to_disk=False):
        """Extract observations from robolab environment format.

        Converts robolab obs structure to intermediate format for DreamZero.
        """
        # Extract images
        right_image = obs_dict["image_obs"]["external_cam"][env_id].clone().detach().cpu().numpy()
        wrist_image = obs_dict["image_obs"]["wrist_cam"][env_id].clone().detach().cpu().numpy()

        # Extract proprioception
        robot_state = obs_dict["proprio_obs"]
        joint_position = robot_state["arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            Image.fromarray(combined_image).save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position.astype(np.float32),
            "gripper_position": gripper_position.astype(np.float32),
        }

    def _resize_image(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize image to target resolution.

        Args:
            image: (H, W, 3) uint8 RGB image
            height: Target height
            width: Target width

        Returns:
            Resized (height, width, 3) uint8 image
        """
        import cv2
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.uint8)

    def _build_visualization(self, curr_obs: dict) -> np.ndarray:
        """Build combined camera view for visualization.

        Uses the same dimensions as what's sent to the model to avoid distortion.
        """
        # Use same dimensions as model input (preserves aspect ratio of what model sees)
        img1 = self._resize_image(curr_obs["right_image"], self.image_height, self.image_width)
        img2 = self._resize_image(curr_obs["wrist_image"], self.image_height, self.image_width)
        return np.concatenate([img1, img2], axis=1)


if __name__ == "__main__":
    import torch

    # Test with fake observations
    client = DreamZeroClient(remote_host="localhost", remote_port=5000)

    fake_obs = {
        "image_obs": {
            "external_cam": [torch.zeros((180, 320, 3), dtype=torch.uint8)],
            "wrist_cam": [torch.zeros((180, 320, 3), dtype=torch.uint8)],
        },
        "proprio_obs": {
            "arm_joint_pos": [torch.zeros((7,), dtype=torch.float32)],
            "gripper_pos": [torch.zeros((1,), dtype=torch.float32)],
        },
    }
    fake_instruction = "pick up the object"

    print("Testing inference...")
    ret = client.infer(fake_obs, fake_instruction)
    print(f"Action shape: {ret['action'].shape}")
    print(f"Action: {ret['action']}")

    print("\nTesting reset...")
    client.reset()
    print("Done.")
