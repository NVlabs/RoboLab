# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode success comes from the ``success*`` termination terms (no simulator scene required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from robolab.core.environments.env import RobolabEnv
from robolab.core.utils.termination_utils import success_terms_fired


class _FakeTerminationManager:
    """The parts of Isaac Lab's TerminationManager that RoboLab reads, for one or more envs."""

    def __init__(self, fired: dict[str, list[bool]], time_out_terms: frozenset[str] = frozenset({"time_out"})):
        self._fired = {name: torch.tensor(values) for name, values in fired.items()}
        self.active_terms = list(fired)
        num_envs = len(next(iter(fired.values())))
        # Isaac Lab: ``terminated`` is the OR of every term not marked time_out=True.
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)
        for name, values in self._fired.items():
            if name not in time_out_terms:
                self.terminated |= values

    def get_term(self, name: str) -> torch.Tensor:
        return self._fired[name]


def _recorded_success(fired: dict[str, bool]) -> bool | None:
    """Run one env to the end of its episode with the given terms fired; return the recorded success."""
    env = RobolabEnv.__new__(RobolabEnv)
    env.__dict__.update(
        # num_envs and device are read-only properties on ManagerBasedEnv (scene.num_envs, sim.device).
        scene=SimpleNamespace(num_envs=1),
        sim=SimpleNamespace(device="cpu"),
        _is_closed=True,  # ManagerBasedEnv.__del__ calls close() unless this is set
        _frozen_envs=torch.zeros(1, dtype=torch.bool),
        _pre_step_frozen=torch.zeros(1, dtype=torch.bool),
        _env_results={},
        _env_term_step={},
        _has_stepped=True,
        episode_length_buf=torch.tensor([50]),
        recorder_manager=None,
        termination_manager=_FakeTerminationManager({name: [value] for name, value in fired.items()}),
    )
    env._reset_idx(torch.tensor([0]))
    result = env.get_env_results()[0]
    assert result["step"] == 50
    return result["success"]


@pytest.mark.parametrize(
    ("fired", "expected"),
    [
        pytest.param({"time_out": False, "success": True, "object_fell_off_table": False}, True, id="success-term"),
        pytest.param({"time_out": True, "success": False, "object_fell_off_table": False}, False, id="time-out"),
        pytest.param({"time_out": False, "success": False, "object_fell_off_table": True}, False, id="failure-term"),
    ],
)
def test_success_comes_from_success_terms(fired, expected):
    assert _recorded_success(fired) is expected


def test_any_of_several_success_terms_counts():
    """Tasks such as grab_a_bagel.py define one ``success_*`` term per acceptable object."""
    fired = {"time_out": False, "success_bagel_02": False, "success_bagel_01": True, "success_bagel_07": False}
    assert _recorded_success(fired) is True


def test_success_terms_fired_per_env():
    """The HDF5 recorder passes several env ids at once."""
    manager = _FakeTerminationManager(
        {
            "time_out": [False, True, False, False],
            "success": [True, False, False, True],
            "object_fell_off_table": [False, False, True, True],
        }
    )
    assert success_terms_fired(manager, [0, 1, 2, 3]).tolist() == [True, False, False, True]
    assert success_terms_fired(manager, [2, 0]).tolist() == [False, True]
