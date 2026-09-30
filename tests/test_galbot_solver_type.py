# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Galbot's contact solver selection and unrelated robot defaults."""

import pytest

from robolab.core.environments.config import parse_env_cfg
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
from robolab.registrations.galbot.auto_env_registrations_jointpos import auto_register_galbot_envs


def test_galbot_registration_selects_pgs_by_default():
    postfix = "SolverTypeDefault"
    auto_register_galbot_envs(
        task="BananaInBowlTask",
        env_postfix=postfix,
    )

    cfg = parse_env_cfg(f"BananaInBowlTask{postfix}", num_envs=1)
    assert cfg.sim.physx.solver_type == 0


def test_galbot_registration_allows_tgs_override():
    postfix = "SolverTypeTGS"
    auto_register_galbot_envs(
        task="BananaInBowlTask",
        env_postfix=postfix,
        solver_type=1,
    )

    cfg = parse_env_cfg(f"BananaInBowlTask{postfix}", num_envs=1)
    assert cfg.sim.physx.solver_type == 1


def test_droid_registration_keeps_global_solver_default():
    auto_register_droid_envs(task="BananaInBowlTask")

    cfg = parse_env_cfg("BananaInBowlTask", num_envs=1)
    assert cfg.sim.physx.solver_type == 1
