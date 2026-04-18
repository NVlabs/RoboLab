# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# This file is the source of truth. A verbatim copy lives at
# droid_plus/eval/base_client.py — keep both in sync when editing.

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class InferenceClient(ABC):
    """Root client for policy inference.

    Subclass override surface, in order of increasing commitment:

    1. Implement the four hooks (``_extract_observation``, ``_pack_request``,
       ``_query_server``, ``_unpack_response``). Chunking, env-id bookkeeping,
       visualization, and reset are handled by the base.
    2. Additionally override ``_postprocess_chunk`` or ``_build_visualization``
       for action-space / logging quirks (gripper binarization, 7->8 padding).
    3. Override ``infer`` entirely if your flow isn't query-then-step-chunk
       (e.g. server-side session state, pre-step caching).

    Two hooks are meant to be split per concern:

      ``_extract_observation``  <- repo-specific (real-robot flat numpy dict vs
                                   sim nested torch batched dict)
      ``_pack_request``         <- backend-specific (wire keys, image sizes)

    Keeping these separate lets the same backend client be paired with
    different observation sources without duplicating the wire format.
    """

    # Subclasses override to match their server's chunk length.
    # horizon=1 is correct for single-action servers.
    open_loop_horizon: int = 1

    def __init__(self) -> None:
        # Per-env chunking state. Subclasses may ignore and manage state however
        # they want.
        self._chunks: dict[int, np.ndarray] = {}
        self._counters: dict[int, int] = {}


    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        """Return ``{"action": np.ndarray, "viz": np.ndarray | None}``.

        Default flow: extract -> pack -> query -> unpack -> postprocess ->
        cache chunk -> step one action. Override entirely if your client needs
        a different control loop.
        """
        extracted = self._extract_observation(obs, env_id=env_id)

        if self._needs_refresh(env_id):
            request = self._pack_request(extracted, instruction)
            response = self._query_server(request)
            chunk = self._unpack_response(response)
            chunk = self._postprocess_chunk(chunk)
            self._set_chunk(env_id, chunk)

        action = self._next_action(env_id)
        viz = self._build_visualization(extracted)
        return {"action": action, "viz": viz}

    def reset(self, *, env_id: int | None = None) -> None:
        """Clear per-episode state. ``env_id=None`` resets all envs.

        Subclasses with server-side session state should override to notify
        the server, then call ``super().reset(env_id=env_id)``.
        """
        if env_id is None:
            self._chunks.clear()
            self._counters.clear()
        else:
            self._chunks.pop(env_id, None)
            self._counters.pop(env_id, None)

    def close(self) -> None:
        """Release transport resources. Default: no-op."""
        return None

    def visualize(self, obs: Any, *, env_id: int = 0) -> np.ndarray | None:
        """Public convenience wrapper for callers that want the viz image."""
        return self._build_visualization(self._extract_observation(obs, env_id=env_id))

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_observation(self, raw_obs: Any, *, env_id: int = 0) -> dict:
        """Convert the caller's native obs into a flat dict of numpy arrays.

        Repo-specific seam. Return whatever keys ``_pack_request`` expects;
        the contract between these two methods is owned by the subclass pair.
        """

    @abstractmethod
    def _pack_request(self, extracted_obs: dict, instruction: str) -> Any:
        """Build the server's wire-format request. Backend-specific."""

    @abstractmethod
    def _query_server(self, request: Any) -> Any:
        """Send the request and return the raw response. Transport-specific."""

    @abstractmethod
    def _unpack_response(self, response: Any) -> np.ndarray:
        """Return a ``(horizon, action_dim)`` numpy array."""

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Action post-processing (binarization, padding, sign flips).

        Default: identity.
        """
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray | None:
        """Image for logging/recording. Default: None."""
        return None

    # ------------------------------------------------------------------
    # Chunking helpers (usable or ignorable by subclasses)
    # ------------------------------------------------------------------

    def _needs_refresh(self, env_id: int) -> bool:
        return env_id not in self._chunks or self._counters[env_id] >= self.open_loop_horizon

    def _set_chunk(self, env_id: int, chunk: np.ndarray) -> None:
        self._chunks[env_id] = chunk
        self._counters[env_id] = 0

    def _next_action(self, env_id: int) -> np.ndarray:
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return action
