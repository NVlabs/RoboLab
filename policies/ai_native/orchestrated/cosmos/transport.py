# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded synchronous transport for the official Cosmos/OpenPI wire format."""
from __future__ import annotations

import threading

import numpy as np


def _pack_array(value):
    """Encode NumPy values with OpenPI's MsgPack extension schema."""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(),
                b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(),
                b"dtype": value.dtype.str}
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _unpack_array(value):
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]),
                          shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def _pack(value):
    import msgpack
    return msgpack.packb(value, default=_pack_array)


def _unpack(value):
    import msgpack
    return msgpack.unpackb(value, object_hook=_unpack_array)


class MsgPackWebSocketTransport:
    """Synchronous WebSocket adapter for OpenPI's NumPy MsgPack schema.

    Provides a metadata handshake, bounded request timeout, and
    cancellation semantics independently of the generic policy transport.
    """

    def __init__(self, uri, *, api_token=None, connect_kwargs=None,
                 metadata_timeout=30, request_timeout=300):
        self.uri = uri
        self.api_token = api_token
        self.connect_kwargs = dict(connect_kwargs or {})
        self.metadata_timeout = metadata_timeout
        self.request_timeout = request_timeout
        self._lock = threading.Lock()
        self._closed = False
        self._connection = None
        self._packer = None

    def connect(self):
        import msgpack
        from websockets.sync.client import connect

        headers = {"Authorization": f"Api-Key {self.api_token}"} if self.api_token else None
        connection = connect(
            self.uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            **self.connect_kwargs,
        )
        with self._lock:
            if self._closed:
                connection.close()
                raise RuntimeError("Policy transport closed during connection")
            self._connection = connection
        self._packer = msgpack.Packer(default=_pack_array)
        metadata = connection.recv(timeout=self.metadata_timeout)
        if isinstance(metadata, str):
            raise RuntimeError(f"Expected binary policy metadata, received text: {metadata}")
        return _unpack(metadata)

    def request(self, request):
        connection = self._connection
        if connection is None or self._packer is None:
            raise RuntimeError("Policy transport is not connected")
        connection.send(self._packer.pack(request))
        response = connection.recv(timeout=self.request_timeout)
        if isinstance(response, str):
            raise RuntimeError(f"Policy endpoint returned an error: {response}")
        return _unpack(response)

    def close(self):
        with self._lock:
            self._closed = True
            connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()
