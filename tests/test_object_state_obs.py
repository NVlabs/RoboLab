# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the opt-in ground-truth object-state observation terms."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from robolab.core.observations.observation_utils import (
    generate_object_state_obs,
    object_pos,
    object_quat,
    object_vel,
)
from robolab.core.utils.isaaclab_compat import quat_wxyz_to_isaaclab


class _FakeScene:
    def __init__(self, entities):
        self._entities = entities
        self.env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    def __getitem__(self, name):
        return self._entities[name]


def _make_env():
    banana = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([[0.5, 0.0, 0.1], [10.5, 0.0, 0.2]]),
        root_quat_w=quat_wxyz_to_isaaclab(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        ),
        root_vel_w=torch.zeros(2, 6),
    ))
    return SimpleNamespace(scene=_FakeScene({"banana": banana}))


def test_object_state_funcs_report_env_local_pos_and_world_quat():
    env = _make_env()
    cfg = SimpleNamespace(name="banana")

    pos = object_pos(env, cfg)
    quat = object_quat(env, cfg)
    vel = object_vel(env, cfg)

    # Position is env-local: env 1's world x of 10.5 minus its origin at x=10.
    torch.testing.assert_close(pos, torch.tensor([[0.5, 0.0, 0.1], [0.5, 0.0, 0.2]]))
    torch.testing.assert_close(quat[1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert vel.shape == (2, 6)


def test_generate_object_state_obs_creates_terms_per_object():
    group = generate_object_state_obs(["banana", "bowl"])()

    for name in ("banana", "bowl"):
        for suffix, func in (("pos", object_pos), ("quat", object_quat), ("vel", object_vel)):
            term = getattr(group, f"{name}_{suffix}")
            assert term.func is func
            assert term.params["asset_cfg"].name == name
    assert group.concatenate_terms is False
