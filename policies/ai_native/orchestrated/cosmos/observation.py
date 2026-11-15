# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-free adapter for RoboLab's official Cosmos3 observation codec."""
from __future__ import annotations


class _Cosmos3ObservationCodec:
    """Provide only the state required by the pinned observation methods."""

    def __init__(self, client_type, episode_idx: int):
        self.client_type = client_type
        self._image_h = client_type.IMAGE_H
        self._image_w = client_type.IMAGE_W
        self._eval_episode_idx = int(episode_idx)

    def pack(self, raw, instruction: str, env_id: int):
        extracted = self.client_type._extract_observation(self, raw, env_id=env_id)
        return self.client_type._pack_request(self, extracted, instruction)


def pack_cosmos3_observation(raw, instruction: str, *, env_id: int, episode_idx: int):
    """Pack one native observation without creating a policy transport."""
    from policies.cosmos3.client import Cosmos3Client

    return _Cosmos3ObservationCodec(Cosmos3Client, episode_idx).pack(
        raw, instruction, env_id)
