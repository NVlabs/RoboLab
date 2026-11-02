# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native scene regressions for the reported Galbot startup explosions."""

import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from robolab.constants import set_output_dir
from robolab.core.environments.config import parse_env_cfg
from robolab.core.environments.runtime import create_env
from robolab.registrations.galbot.auto_env_registrations_jointpos import auto_register_galbot_envs
from robolab.robots.galbot_golf import (
    GalbotGolfLeftEgoCameraCfg,
    GalbotGolfLeftWristCameraCfg,
    GalbotGolfRightWristCameraCfg,
    GalbotGolfTabletopCfg,
    make_galbot_golf_tabletop_cfg,
)
from robolab.robots.galbot_golf_definitions import WHOLE_BODY_JOINTS


@pytest.mark.parametrize("decimation, control_hz", [(None, 10), (16, 15)])
def test_cosmos3_cli_registers_contact_profile(monkeypatch, app_launcher_class, decimation, control_hz):
    app_module = sys.modules["isaaclab.app"]
    original_launcher = app_launcher_class

    class ReuseLauncher:
        add_app_launcher_args = original_launcher.add_app_launcher_args

        def __init__(self, _args):
            self.app = None  # Registration probe; main() and inference are not run.

    client_module = ModuleType("policies.cosmos3.client")
    client_module.Cosmos3Client = SimpleNamespace
    client_module.Cosmos3GalbotClient = SimpleNamespace
    monkeypatch.setitem(sys.modules, "policies.cosmos3.client", client_module)
    monkeypatch.setattr(app_module, "AppLauncher", ReuseLauncher, raising=False)
    argv = ["run.py", "--embodiment", "galbot", "--task", "BananaInBowlTask", "--headless"]
    if decimation is not None:
        argv += ["--galbot-decimation", str(decimation)]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_module("policies.cosmos3.run", run_name="_contact_profile_probe")
    cfg = parse_env_cfg("BananaInBowlTaskGalbotGolfFixedBase", num_envs=1)
    assert cfg.sim.dt == pytest.approx(1 / 240)
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1 / control_hz)
    assert cfg.sim.physx.solver_type == 1
    assert cfg.sim.physx.max_velocity_iteration_count == 1
    assert cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count == 1
    assert cfg.scene.robot.actuators["grippers"].effort_limit_sim == 0.5


def test_tgs_iteration_override_keeps_scene_and_robot_consistent():
    robot_cfg = make_galbot_golf_tabletop_cfg(gripper_effort=0.5, solver_iterations=(128, 1))
    auto_register_galbot_envs(
        task="BananaInBowlTask", env_postfix="ContactBudgetTest", robot_cfg=robot_cfg,
        solver_type=1, solver_iterations=(128, 1), dt=1 / 240, decimation=16, render_interval=16,
    )
    cfg = parse_env_cfg("BananaInBowlTaskContactBudgetTest", num_envs=1)
    assert cfg.sim.physx.max_position_iteration_count == 128
    assert cfg.sim.physx.max_velocity_iteration_count == 1
    assert cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count == 1
    assert cfg.scene.robot.actuators["grippers"].effort_limit_sim == 0.5
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1 / 15)
    assert GalbotGolfTabletopCfg().robot.spawn.articulation_props.solver_velocity_iteration_count == 4


@pytest.mark.parametrize("iterations", [(0, 1), (256, 1), (128, -1), (128, 256), (128, 1.0), (128, True)])
def test_rejects_invalid_articulation_iteration_budget(iterations):
    with pytest.raises(ValueError, match="solver_iterations"):
        make_galbot_golf_tabletop_cfg(solver_iterations=iterations)


@pytest.mark.parametrize("task", ["TakeMeasuringSpoonOutTask", "CookingClearPlateTask", "FruitsOnionToPlateTask"])
def test_scene_stays_at_rest_across_full_and_partial_resets(task, tmp_path):
    set_output_dir(str(tmp_path))
    postfix = "StartupContactTest"
    auto_register_galbot_envs(
        task=task, env_postfix=postfix,
        cameras=[GalbotGolfLeftEgoCameraCfg, GalbotGolfLeftWristCameraCfg, GalbotGolfRightWristCameraCfg],
        robot_cfg=make_galbot_golf_tabletop_cfg(
            init_pose="dataset_mean", gain_profile="reduced", gripper_effort=0.5, solver_iterations=(128, 1)),
        solver_type=1, solver_iterations=(128, 1), dt=1 / 240, decimation=16, render_interval=16,
    )
    cfg = parse_env_cfg(task + postfix, num_envs=2, seed=1)
    success = cfg.terminations.success
    cfg.recorders = None
    for name in vars(cfg.terminations):
        if not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    env, _ = create_env(cfg)
    original_update = env.scene.update
    initial = {}
    maxima = {}

    def sample(dt):
        original_update(dt)
        for name, obj in env.scene.rigid_objects.items():
            state = obj.data.root_state_w
            assert torch.isfinite(state).all(), name
            maxima[name][0] = max(maxima[name][0], float(state[:, 7:10].norm(dim=-1).max()))
            maxima[name][1] = max(maxima[name][1], float((state[:, :3] - initial[name]).norm(dim=-1).max()))

    try:
        robot = env.scene["robot"]
        ids, _ = robot.find_joints(WHOLE_BODY_JOINTS, preserve_order=True)
        for reset_ids in [None, None, torch.tensor([1], device=env.device)]:
            env.scene.update = original_update
            env.reset_eval_state()
            env.reset(env_ids=reset_ids)
            assert not success.func(env, **success.params).any(), "Task starts already solved"
            initial = {n: obj.data.root_pos_w.clone() for n, obj in env.scene.rigid_objects.items()}
            maxima = {n: [0.0, 0.0] for n in initial}
            env.scene.update = sample
            action = robot.data.joint_pos[:, ids].clone()
            action[:, -2:] = 0
            for _ in range(8):
                env.step(action)
            assert not success.func(env, **success.params).any(), "Task becomes solved during passive settling"
            for name, (speed, displacement) in maxima.items():
                assert speed < 0.15, f"{task}/{name}: startup speed {speed:.3f} m/s"
                assert displacement < 0.006, f"{task}/{name}: startup drift {displacement:.4f} m"
    finally:
        env.scene.update = original_update
        env.close()
