# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote Cosmos3 student policy for the public RoboLab System-2 gate."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np

from ..io import require

MODEL = "nvidia/Cosmos3-Nano-Policy-DROID"


class Cosmos3Student:
    """One endpoint connection and policy session owned by one episode."""

    def __init__(self, uri, token, *, robolab_revision, action_horizon=15,
                 transport_cls=None):
        if type(action_horizon) is not int or not 1 <= action_horizon <= 32:
            raise ValueError("action_horizon must be an integer in 1..32")
        self.action_horizon = action_horizon
        if transport_cls is None:
            from .transport import MsgPackWebSocketTransport
            transport_cls = MsgPackWebSocketTransport
        self.uri = uri
        self._token = token
        self._transport_cls = transport_cls
        self._connect_kwargs = dict(open_timeout=30, close_timeout=5,
                                    ping_interval=20, ping_timeout=20)
        self.session_id = "robolab-system2-" + uuid.uuid4().hex
        self.client, server_metadata = self._connect()
        require(isinstance(server_metadata, dict), "Cosmos metadata must be a mapping")
        self._server_metadata = server_metadata
        identity_source = ("owner_declared_model; endpoint_metadata_reported"
                           if server_metadata else
                           "owner_declared_model; endpoint_metadata_empty_unverified")
        identity = dict(declared_model=MODEL, endpoint=uri,
            robolab_revision=robolab_revision, server_metadata=server_metadata,
            identity_source=identity_source)
        identity_sha256 = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.metadata = dict(identity, identity_sha256=identity_sha256,
            action_semantics="absolute_joint_position_7_plus_binary_gripper",
            raw_action_shape=[32, 8], gated_action_shape=[action_horizon, 8])
        self.index = 0

    def _connect(self):
        client = self._transport_cls(self.uri, api_token=self._token,
            connect_kwargs=self._connect_kwargs, metadata_timeout=30)
        return client, client.connect()

    def _request_with_retry(self, request, max_retries=3):
        """Mirror the official Cosmos client's bounded reconnect behavior."""
        import websockets.exceptions

        connection_errors = (websockets.exceptions.ConnectionClosedError,
                             websockets.exceptions.ConnectionClosedOK,
                             OSError, TimeoutError)
        for attempt in range(max_retries):
            try:
                return self.client.request(request)
            except connection_errors:
                if attempt + 1 >= max_retries:
                    raise
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client, metadata = self._connect()
                require(metadata == self._server_metadata,
                        "Cosmos metadata changed after reconnect")

    def infer(self, observation, output):
        before = time.monotonic()
        image = np.asarray(observation["cosmos_image"])
        joints = np.asarray(observation["states"][:7], dtype=np.float32)
        gripper = np.asarray(observation["states"][7:8], dtype=np.float32)
        require(image.dtype == np.uint8 and image.shape == (540, 640, 3),
                "Expected official Cosmos uint8 [540,640,3] image")
        require(joints.shape == (7,) and gripper.shape == (1,)
                and np.isfinite(joints).all() and np.isfinite(gripper).all(),
                "Invalid Cosmos proprioception")
        request = {
            "observation/image": image,
            "observation/joint_position": joints,
            "observation/gripper_position": gripper,
            "prompt": observation["instruction"],
            "session_id": self.session_id,
        }
        response = self._request_with_retry(request)
        server_timing = None
        if isinstance(response, dict):
            if response.get("type") == "error":
                raise RuntimeError(str(response.get("message") or "Cosmos inference failed"))
            server_timing = response.get("server_timing")
            response = response.get("action", response.get("actions"))
        raw = np.asarray(response)
        require(raw.dtype == np.float32 and raw.shape == (32, 8) and np.isfinite(raw).all(),
                "Expected finite Cosmos float32 [32,8] chunk")
        actions = raw[:self.action_horizon].copy()
        actions[:, 7] = (actions[:, 7] > .5).astype(np.float32)
        output = Path(output)
        np.savez_compressed(output, raw_actions=raw, actions=actions,
            cosmos_image=image, states=np.asarray(observation["states"], np.float32))
        prediction = dict(prediction_id=uuid.uuid4().hex, inference_index=self.index,
            inference_seconds=time.monotonic() - before, raw_action_shape=[32, 8],
            gated_action_shape=[self.action_horizon, 8], policy_session_id=self.session_id,
            server_timing=server_timing)
        self.index += 1
        return actions, prediction

    def close(self):
        self.client.close()
