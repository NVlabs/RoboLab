# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin down the coordinate-frame contract (docs/frames.md).

Everything on the frame-contract branch is a "no behavior change for
Franka-family robots" claim; these tests make a regression loud:

- the pre-contract fallback rule lives in exactly one place and works,
- the robot-root EE math is correct when the root is NOT at the env origin
  (no floor-standing robot exists on main yet, so this is synthetic),
- a recorded episode carries robot_root_pose, and for DROID the robot-root
  EE values equal the historical env-local ones (the Franka coincidence).
"""

import glob
import os

import h5py
import numpy as np
import torch
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from robolab.core.logging.frame_compat import demo_robot_root_pose
from robolab.core.utils.isaaclab_compat import (
    as_torch,
    quat_isaaclab_to_wxyz,
    quat_wxyz_to_isaaclab,
)


def test_frame_compat_fallback(tmp_path):
    """Demos without robot_root_pose are pre-contract → identity root pose."""
    path = tmp_path / "fake.hdf5"
    with h5py.File(path, "w") as f:
        new = f.create_group("data/demo_new")
        new.create_dataset("actions", data=np.zeros((7, 8)))
        root = new.create_group("robot_root_pose")
        root.create_dataset("position", data=np.tile([0.1, -0.2, -0.65], (7, 1)))
        root.create_dataset("orientation", data=np.tile([0.0, 0.0, 0.0, 1.0], (7, 1)))
        old = f.create_group("data/demo_old")
        old.create_dataset("actions", data=np.zeros((5, 8)))
        f.create_group("data/demo_bare")

    with h5py.File(path, "r") as f:
        pos, quat = demo_robot_root_pose(f["data/demo_new"])
        assert pos.shape == (7, 3)
        np.testing.assert_allclose(pos[0], [0.1, -0.2, -0.65])
        np.testing.assert_allclose(quat[0], [0.0, 0.0, 0.0, 1.0])

        pos, quat = demo_robot_root_pose(f["data/demo_old"])
        assert pos.shape == (5, 3) and np.abs(pos).max() == 0.0
        np.testing.assert_allclose(quat, np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)))

        pos, quat = demo_robot_root_pose(f["data/demo_bare"])
        assert pos.shape == (1, 3) and quat[0, 0] == 1.0

        pos, _ = demo_robot_root_pose(f["data/demo_old"], num_steps=9)
        assert pos.shape == (9, 3)


def test_ee_frame_math_with_offset_root():
    """The robot-root EE math is correct when the root is away from the origin.

    Synthetic stand-in for a floor-standing embodiment: root translated and
    yawed 90 deg. subtract_frame_transforms must recover the local EE pose,
    and composing with the root pose must round-trip to the world pose.
    """
    sqrt2_2 = float(np.sqrt(2.0) / 2.0)
    root_pos = torch.tensor([[1.0, 2.0, -0.7]])
    root_quat = torch.tensor([[sqrt2_2, 0.0, 0.0, sqrt2_2]])  # (w, x, y, z), yaw +90 deg
    ee_local_pos = torch.tensor([[0.5, 0.0, 0.3]])
    ee_local_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    # World EE pose, built independently: yaw+90 maps (x, y) -> (-y, x).
    ee_world_pos = torch.tensor([[1.0, 2.5, -0.4]])
    ee_world_quat = root_quat.clone()

    root_quat_internal = quat_wxyz_to_isaaclab(root_quat)
    ee_world_quat_internal = quat_wxyz_to_isaaclab(ee_world_quat)
    pos, quat_internal = subtract_frame_transforms(
        root_pos, root_quat_internal, ee_world_pos, ee_world_quat_internal
    )
    quat = quat_isaaclab_to_wxyz(quat_internal)
    torch.testing.assert_close(pos, ee_local_pos, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(quat, ee_local_quat, atol=1e-6, rtol=0.0)

    # Round trip: robot-root EE + recorded root pose -> world again.
    back_pos, back_quat_internal = combine_frame_transforms(
        root_pos, root_quat_internal, pos, quat_internal
    )
    back_quat = quat_isaaclab_to_wxyz(back_quat_internal)
    torch.testing.assert_close(back_pos, ee_world_pos, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(back_quat, ee_world_quat, atol=1e-6, rtol=0.0)


def test_recorded_episode_frame_contract():
    """A recorded DROID episode satisfies the frame contract end to end.

    Arms the recorder the way the eval loop does (a bare create_env +
    end_episode never writes an HDF5, so without arming this test would pass
    vacuously on nothing).
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA device required for a recorded episode")

    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env, end_episode
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    auto_register_droid_envs(task="BananaInBowlTask")
    env_name = get_envs(task="BananaInBowlTask")[0]
    env, _ = create_env(env_name, num_envs=1, use_fabric=True)
    num_steps = 5
    expected_ee_pos = []
    expected_ee_quat = []
    try:
        env.reset()
        env.recorder_manager.set_episode_index(0, env_ids=[0])
        actions = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
        for _ in range(num_steps):
            env.step(actions)
            # Historical env-local values, computed independently of the code
            # under test. For DROID the root is at the env origin with identity
            # rotation, so robot-root must equal these (the Franka coincidence).
            robot = env.scene["robot"]
            body_idx = robot.data.body_names.index("base_link")
            expected_ee_pos.append(
                (as_torch(robot.data.body_pos_w)[:, body_idx, :] - env.scene.env_origins[:, 0:3]).cpu().numpy()[0]
            )
            expected_ee_quat.append(
                quat_isaaclab_to_wxyz(as_torch(robot.data.body_quat_w)[:, body_idx, :]).cpu().numpy()[0]
            )
        export_dir = env.recorder_manager.cfg.dataset_export_dir_path
        end_episode(env)
    finally:
        env.close()

    files = sorted(glob.glob(os.path.join(export_dir, "**", "*.hdf5"), recursive=True), key=os.path.getmtime)
    assert files, f"no HDF5 exported under {export_dir}"
    with h5py.File(files[-1], "r") as f:
        demo = f["data"][sorted(f["data"].keys())[0]]
        assert "robot_root_pose" in demo, "per-step root pose channel missing"
        root_pos, root_quat = demo_robot_root_pose(demo)
        assert root_pos.shape[0] == demo["actions"].shape[0], "root pose is not per-step"
        assert np.abs(root_pos).max() < 1e-4, "DROID root must sit at the env origin"
        np.testing.assert_allclose(root_quat, np.tile([1.0, 0.0, 0.0, 0.0], (root_quat.shape[0], 1)), atol=1e-5)

        ee = demo["ee_pose"]
        np.testing.assert_allclose(ee["position"][-num_steps:], np.stack(expected_ee_pos), atol=1e-5)
        np.testing.assert_allclose(ee["orientation"][-num_steps:], np.stack(expected_ee_quat), atol=1e-5)
