# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from policies.cosmos3.client import Cosmos3Client
from robolab.eval.base_client import InferenceClient


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return _Tensor(self.value[index])

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Transport:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.requests = []
        self.closed = False

    def request(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


def _client_without_connection() -> Cosmos3Client:
    client = Cosmos3Client.__new__(Cosmos3Client)
    InferenceClient.__init__(client)
    client._image_h = 2
    client._image_w = 2
    return client


def test_constructor_remote_uri_and_explicit_token_take_precedence(monkeypatch):
    transport = _Transport()
    monkeypatch.setenv("COSMOS3_API_TOKEN", "environment-token")
    monkeypatch.setattr(Cosmos3Client, "_connect", lambda _self: transport)

    client = Cosmos3Client(
        remote_host="ignored-host",
        remote_port=9999,
        remote_uri="wss://policy.example/v1/realtime/robot/openpi",
        api_token="explicit-token",
    )

    assert client._uri == "wss://policy.example/v1/realtime/robot/openpi"
    assert client._api_token == "explicit-token"
    assert client.client is transport


def test_constructor_uses_host_port_and_environment_token(monkeypatch):
    monkeypatch.setenv("COSMOS3_API_TOKEN", "environment-token")
    monkeypatch.setattr(Cosmos3Client, "_connect", lambda _self: _Transport())

    client = Cosmos3Client(remote_host="policy-host", remote_port=8123)

    assert client._uri == "ws://policy-host:8123"
    assert client._api_token == "environment-token"


def test_parallel_environments_receive_distinct_stable_session_ids():
    client = _client_without_connection()
    client.begin_episode(4)
    images = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    raw_obs = {
        "image_obs": {
            "over_shoulder_left_camera": _Tensor(images),
            "over_shoulder_right_camera": _Tensor(images),
            "wrist_cam": _Tensor(images),
        },
        "proprio_obs": {
            "arm_joint_pos": _Tensor(np.zeros((2, 7), dtype=np.float32)),
            "gripper_pos": _Tensor(np.zeros((2, 1), dtype=np.float32)),
        },
    }

    env0 = client._extract_observation(raw_obs, env_id=0)
    env1 = client._extract_observation(raw_obs, env_id=1)

    assert env0["session_id"] == "robolab-episode-4-env-0"
    assert env1["session_id"] == "robolab-episode-4-env-1"
    assert client._pack_request(env0, "pick")["session_id"] == env0["session_id"]

    client.begin_episode(5)
    assert client._extract_observation(raw_obs, env_id=0)["session_id"] == "robolab-episode-5-env-0"


@pytest.mark.parametrize(
    "response",
    [
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        {"action": np.asarray([[1.0, 2.0]], dtype=np.float32)},
        {"actions": np.asarray([[1.0, 2.0]], dtype=np.float32)},
    ],
)
def test_unpack_response_accepts_supported_shapes(response):
    client = _client_without_connection()

    np.testing.assert_array_equal(
        client._unpack_response(response),
        np.asarray([[1.0, 2.0]], dtype=np.float32),
    )


def test_unpack_response_reports_server_error():
    client = _client_without_connection()

    with pytest.raises(RuntimeError, match="model failed"):
        client._unpack_response({"type": "error", "message": "model failed"})


def test_unpack_response_rejects_unknown_mapping():
    client = _client_without_connection()

    with pytest.raises(RuntimeError, match="did not contain"):
        client._unpack_response({"video": b"unused"})


def test_infer_retry_reconnects_with_existing_configuration_and_clears_chunks(monkeypatch):
    client = _client_without_connection()
    first = _Transport([OSError("connection lost")])
    second = _Transport([np.asarray([[0.0]], dtype=np.float32)])
    client.client = first
    client._chunks[0] = np.asarray([[1.0]])
    client._counters[0] = 1
    monkeypatch.setattr(client, "_connect", lambda: second)

    response = client._infer_with_retry({"prompt": "pick"})

    np.testing.assert_array_equal(response, np.asarray([[0.0]], dtype=np.float32))
    assert first.closed is True
    assert client.client is second
    assert client._chunks == {}
    assert client._counters == {}
