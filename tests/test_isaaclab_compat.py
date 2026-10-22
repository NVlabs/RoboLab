# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import torch

import robolab.core.utils.isaaclab_compat as compat


def test_quaternion_conversions_are_inverse():
    wxyz = (0.5, 0.1, 0.2, 0.3)
    xyzw = compat.quat_wxyz_to_isaaclab(wxyz, uses_xyzw=True)
    assert xyzw == (0.1, 0.2, 0.3, 0.5)
    assert compat.quat_isaaclab_to_wxyz(xyzw, uses_xyzw=True) == wxyz


def test_pose_conversion_preserves_position():
    pose = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1, 0.2, 0.3]])
    internal = compat.pose_wxyz_to_isaaclab(pose, uses_xyzw=True)
    torch.testing.assert_close(internal, torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.5]]))
    torch.testing.assert_close(compat.pose_isaaclab_to_wxyz(internal, uses_xyzw=True), pose)


def test_scene_state_round_trip(monkeypatch):
    monkeypatch.setattr(compat, "ISAACLAB_USES_XYZW", True)
    state = {
        "articulation": {
            "robot": {
                "root_pose": torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]]),
                "joint_position": torch.tensor([[0.25]]),
            }
        },
        "rigid_object": {
            "cube": {"root_pose": torch.tensor([[4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]])}
        },
    }
    internal = compat.scene_state_to_isaaclab(state)
    assert internal["articulation"]["robot"]["root_pose"][0].tolist() == [1, 2, 3, 0, 0, 0, 1]
    restored = compat.scene_state_from_isaaclab(internal)
    torch.testing.assert_close(restored["articulation"]["robot"]["root_pose"], state["articulation"]["robot"]["root_pose"])
    torch.testing.assert_close(restored["rigid_object"]["cube"]["root_pose"], state["rigid_object"]["cube"]["root_pose"])


def test_config_rotations_are_prepared_once_and_recorded_as_wxyz(monkeypatch):
    monkeypatch.setattr(compat, "ISAACLAB_USES_XYZW", True)

    class Config:
        def __init__(self):
            self.rot = (1.0, 0.0, 0.0, 0.0)
            self.child = {"rot": [0.5, 0.1, 0.2, 0.3]}

        def to_dict(self):
            return deepcopy(vars(self))

    cfg = Config()
    compat.prepare_env_cfg(cfg)
    assert cfg.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.child["rot"] == [0.1, 0.2, 0.3, 0.5]
    compat.prepare_env_cfg(cfg)
    assert cfg.rot == (0.0, 0.0, 0.0, 1.0)
    recorded = compat.env_cfg_to_recording_dict(cfg)
    assert recorded["rot"] == (1.0, 0.0, 0.0, 0.0)
    assert recorded["child"]["rot"] == [0.5, 0.1, 0.2, 0.3]


def test_droid_relative_ik_authors_identity_at_wxyz_boundary(monkeypatch):
    """The relative-IK offset must not inherit Isaac Lab 3's XYZW default."""
    from robolab.robots.droid import DroidRelIKActionCfg

    cfg = DroidRelIKActionCfg()
    assert cfg.arm_action.body_offset.rot == (1.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(compat, "ISAACLAB_USES_XYZW", True)
    compat.prepare_env_cfg(cfg)
    assert cfg.arm_action.body_offset.rot == (0.0, 0.0, 0.0, 1.0)
