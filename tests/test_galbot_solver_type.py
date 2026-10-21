# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Galbot's contact solver selection and unrelated robot defaults."""

import pytest

from robolab.core.environments.config import parse_env_cfg
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
from robolab.registrations.galbot.auto_env_registrations_abs_ik import auto_register_galbot_abs_ik_envs
from robolab.registrations.galbot.auto_env_registrations_jointpos import auto_register_galbot_envs


@pytest.mark.parametrize("register", [auto_register_galbot_envs, auto_register_galbot_abs_ik_envs])
@pytest.mark.parametrize(
    ("options", "expected_solver"),
    [({}, 0), ({"solver_type": 0}, 0), ({"solver_type": 1}, 1), ({"solver_type": None}, 1)],
    ids=["default-pgs", "explicit-pgs", "explicit-tgs", "global-default"],
)
def test_galbot_solver_selection_preserves_iterations(register, options, expected_solver):
    postfix = f"SolverType{register.__name__}{len(options)}{options.get('solver_type')}"
    register(task="BananaInBowlTask", env_postfix=postfix, **options)

    cfg = parse_env_cfg(f"BananaInBowlTask{postfix}", num_envs=1)
    assert cfg.sim.physx.solver_type == expected_solver
    for axis, expected in (("position", 128), ("velocity", 4)):
        fields = [f"num_{axis}_iterations", f"max_{axis}_iteration_count"]
        available = [field for field in fields if hasattr(cfg.sim.physx, field)]
        assert available, f"No {axis} iteration setting found"
        for field in available:
            assert getattr(cfg.sim.physx, field) == expected
        assert getattr(cfg.scene.robot.spawn.articulation_props, f"solver_{axis}_iteration_count") == expected


def test_droid_registration_keeps_global_solver_default():
    auto_register_droid_envs(task="BananaInBowlTask")
    cfg = parse_env_cfg("BananaInBowlTask", num_envs=1)
    assert cfg.sim.physx.solver_type == 1
