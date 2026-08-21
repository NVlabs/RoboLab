# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared synchronous MsgPack/NumPy WebSocket transport for policy clients.

The transport owns only the wire-level behavior shared by policy backends:
NumPy-aware MsgPack encoding, an initial metadata handshake, optional Bearer
authentication, and one request/response exchange. Backend clients retain
their own retry, timeout, and session-lifecycle policies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import websockets.sync.client


class MsgPackNumpy:
    """Small MsgPack wrapper compatible with OpenPI NumPy markers."""

    def __init__(self) -> None:
        import msgpack

        self._msgpack = msgpack

    def pack(self, obj: Any) -> bytes:
        # strict_types=True rejects tuples, which are valid in metadata.
        return self._msgpack.packb(obj, default=self._encode_numpy)

    def unpack(self, data: bytes) -> Any:
        return self._msgpack.unpackb(data, object_hook=self._decode_numpy, strict_map_key=False)

    @staticmethod
    def _encode_numpy(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind in ("V", "O", "c"):
                raise ValueError(f"Unsupported dtype: {obj.dtype}")
            if not obj.flags.c_contiguous:
                obj = np.ascontiguousarray(obj)
            return {
                b"__ndarray__": True,
                b"data": obj.tobytes(),
                b"dtype": obj.dtype.str,
                b"shape": obj.shape,
            }

        if isinstance(obj, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": obj.item(),
                b"dtype": obj.dtype.str,
            }

        raise TypeError(f"Unsupported type: {type(obj)!r}")

    @staticmethod
    def _decode_numpy(obj: dict[Any, Any]) -> Any:
        if b"__ndarray__" in obj:
            array = np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"])).copy()
            return array.reshape(tuple(obj[b"shape"]))
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj


class MsgPackWebSocketTransport:
    """Synchronous policy transport with metadata handshake and Bearer auth."""

    def __init__(
        self,
        uri: str,
        *,
        api_token: str | None = None,
        connect_kwargs: dict[str, Any] | None = None,
        metadata_timeout: float | None = None,
    ) -> None:
        self.uri = uri
        self.packer = MsgPackNumpy()
        self.server_metadata: Any = None
        self._auth_headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        self._connect_kwargs = {
            "compression": None,
            "max_size": None,
            **(connect_kwargs or {}),
        }
        self._metadata_timeout = metadata_timeout
        self._ws: Any = None

    @property
    def connected(self) -> bool:
        try:
            return self._ws is not None and self._ws.socket is not None
        except Exception:
            return False

    def connect(self) -> Any:
        """Connect and consume the server's first binary metadata message."""
        kwargs = dict(self._connect_kwargs)
        kwargs["additional_headers"] = self._auth_headers
        try:
            self._ws = websockets.sync.client.connect(self.uri, **kwargs)
        except TypeError:
            # Older websockets releases bundled with some Isaac Sim versions
            # do not expose the sync-client ping tuning arguments.
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("ping_interval", None)
            fallback_kwargs.pop("ping_timeout", None)
            self._ws = websockets.sync.client.connect(self.uri, **fallback_kwargs)

        try:
            raw_metadata = self._recv(timeout=self._metadata_timeout)
            if isinstance(raw_metadata, str):
                raise RuntimeError(f"Policy server returned text metadata: {raw_metadata}")
            self.server_metadata = self.packer.unpack(raw_metadata)
            return self.server_metadata
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def send_recv(self, data: bytes, *, timeout: float | None = None) -> bytes | str:
        if self._ws is None:
            raise ConnectionError("Policy WebSocket is not connected")
        self._ws.send(data)
        return self._recv(timeout=timeout)

    def request(self, request: Any, *, timeout: float | None = None) -> Any:
        raw = self.send_recv(self.packer.pack(request), timeout=timeout)
        if isinstance(raw, str):
            raise RuntimeError(f"Policy server returned a text error:\n{raw}")
        return self.packer.unpack(raw)

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None

    def _recv(self, *, timeout: float | None) -> bytes | str:
        if timeout is None:
            return self._ws.recv()
        return self._ws.recv(timeout=timeout)
