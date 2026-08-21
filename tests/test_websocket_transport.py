# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from robolab.eval import websocket_transport
from robolab.eval.websocket_transport import MsgPackNumpy, MsgPackWebSocketTransport


class _FakeWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.socket = object()
        self.closed = False
        self.recv_timeouts = []

    def recv(self, timeout=None):
        self.recv_timeouts.append(timeout)
        return self.messages.pop(0)

    def send(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True
        self.socket = None


def test_msgpack_numpy_round_trip_and_writable_arrays():
    codec = MsgPackNumpy()
    payload = {
        "array": np.arange(6, dtype=np.float32).reshape(2, 3)[:, ::-1],
        "scalar": np.int64(7),
    }

    decoded = codec.unpack(codec.pack(payload))

    np.testing.assert_array_equal(decoded["array"], payload["array"])
    assert decoded["array"].flags.writeable is True
    assert decoded["scalar"] == np.int64(7)


def test_msgpack_numpy_rejects_unsupported_arrays():
    codec = MsgPackNumpy()

    with pytest.raises(ValueError, match="Unsupported dtype"):
        codec.pack(np.asarray([object()], dtype=object))


def test_transport_sends_bearer_header_consumes_handshake_and_decodes_response(monkeypatch):
    codec = MsgPackNumpy()
    action = np.asarray([[1.0, 2.0]], dtype=np.float32)
    ws = _FakeWebSocket([codec.pack({"action_space": "joint_position"}), codec.pack(action)])
    calls = []

    def connect(uri, **kwargs):
        calls.append((uri, kwargs))
        return ws

    monkeypatch.setattr(websocket_transport.websockets.sync.client, "connect", connect)
    transport = MsgPackWebSocketTransport(
        "wss://policy.example/v1/realtime/robot/openpi",
        api_token="secret-token",
        connect_kwargs={"open_timeout": 10, "ping_interval": 60},
        metadata_timeout=20,
    )

    metadata = transport.connect()
    response = transport.request({"prompt": "pick"}, timeout=30)

    assert metadata == {"action_space": "joint_position"}
    np.testing.assert_array_equal(response, action)
    assert calls[0][0] == "wss://policy.example/v1/realtime/robot/openpi"
    assert calls[0][1]["additional_headers"] == {"Authorization": "Bearer secret-token"}
    assert calls[0][1]["compression"] is None
    assert calls[0][1]["max_size"] is None
    assert ws.recv_timeouts == [20, 30]
    assert codec.unpack(ws.sent[0]) == {"prompt": "pick"}

    transport.close()
    assert ws.closed is True
    assert transport.connected is False


def test_transport_retries_without_ping_tuning_for_older_websockets(monkeypatch):
    codec = MsgPackNumpy()
    ws = _FakeWebSocket([codec.pack({})])
    calls = []

    def connect(_uri, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TypeError("unsupported ping option")
        return ws

    monkeypatch.setattr(websocket_transport.websockets.sync.client, "connect", connect)
    transport = MsgPackWebSocketTransport(
        "ws://localhost:8000",
        connect_kwargs={"ping_interval": 60, "ping_timeout": 600, "open_timeout": 300},
    )

    transport.connect()

    assert calls[0]["ping_interval"] == 60
    assert "ping_interval" not in calls[1]
    assert "ping_timeout" not in calls[1]
    assert calls[1]["open_timeout"] == 300


def test_transport_rejects_text_inference_reply(monkeypatch):
    codec = MsgPackNumpy()
    ws = _FakeWebSocket([codec.pack({}), "server traceback"])
    monkeypatch.setattr(websocket_transport.websockets.sync.client, "connect", lambda *_args, **_kwargs: ws)
    transport = MsgPackWebSocketTransport("ws://localhost:8000")
    transport.connect()

    with pytest.raises(RuntimeError, match="server traceback"):
        transport.request({"prompt": "pick"})


def test_transport_closes_connection_after_invalid_handshake(monkeypatch):
    ws = _FakeWebSocket(["authentication failed"])
    monkeypatch.setattr(websocket_transport.websockets.sync.client, "connect", lambda *_args, **_kwargs: ws)
    transport = MsgPackWebSocketTransport("ws://localhost:8000")

    with pytest.raises(RuntimeError, match="authentication failed"):
        transport.connect()

    assert ws.closed is True
    assert transport.connected is False


def test_transport_rejects_structured_handshake_error(monkeypatch):
    codec = MsgPackNumpy()
    ws = _FakeWebSocket([codec.pack({"type": "error", "message": "policy disabled"})])
    monkeypatch.setattr(websocket_transport.websockets.sync.client, "connect", lambda *_args, **_kwargs: ws)
    transport = MsgPackWebSocketTransport("ws://localhost:8000")

    with pytest.raises(RuntimeError, match="policy disabled"):
        transport.connect()

    assert ws.closed is True


def test_transport_explains_close_before_handshake(monkeypatch):
    class ClosedBeforeMetadata(Exception):
        pass

    class ClosingWebSocket(_FakeWebSocket):
        def recv(self, timeout=None):
            raise ClosedBeforeMetadata("closed")

    ws = ClosingWebSocket([])
    monkeypatch.setattr(websocket_transport.websockets.sync.client, "connect", lambda *_args, **_kwargs: ws)
    monkeypatch.setattr("websockets.exceptions.ConnectionClosed", ClosedBeforeMetadata)
    transport = MsgPackWebSocketTransport("ws://localhost:8000")

    with pytest.raises(ConnectionError, match="closed before sending its metadata handshake"):
        transport.connect()

    assert ws.closed is True
