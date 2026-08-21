# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from policies.dreamzero import client as dreamzero_client


class _FakeTransport:
    def __init__(self, uri, **kwargs):
        self.uri = uri
        self.kwargs = kwargs
        self.packer = object()


def test_dreamzero_reuses_shared_transport_and_environment_token(monkeypatch):
    created = []

    def make_transport(uri, **kwargs):
        transport = _FakeTransport(uri, **kwargs)
        created.append(transport)
        return transport

    monkeypatch.setenv("DREAMZERO_API_TOKEN", "environment-token")
    monkeypatch.setattr(dreamzero_client, "MsgPackWebSocketTransport", make_transport)
    monkeypatch.setattr(dreamzero_client.DreamZeroClient, "_connect_with_retries", lambda _self: None)

    client = dreamzero_client.DreamZeroClient(
        remote_uri="wss://dreamzero.example/policy",
        api_token=None,
    )

    assert client._transport is created[0]
    assert created[0].uri == "wss://dreamzero.example/policy"
    assert created[0].kwargs == {
        "api_token": "environment-token",
        "connect_kwargs": {
            "open_timeout": dreamzero_client.CONNECT_TIMEOUT_SECS,
            "ping_interval": dreamzero_client.PING_INTERVAL_SECS,
            "ping_timeout": dreamzero_client.PING_TIMEOUT_SECS,
        },
        "metadata_timeout": dreamzero_client.RECV_TIMEOUT_SECS,
    }
