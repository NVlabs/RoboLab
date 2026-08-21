# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for policy-proxy interoperability metadata."""

from __future__ import annotations

import numpy as np

from policies.volo.client import VoloCosmos3Client
from policies.volo.metadata import OrchestratorMetadataMixin
from robolab.eval.base_client import InferenceClient


class _Tensor:
    """Small torch-like test double for dependency-light tests."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return _Tensor(self.value[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Client(InferenceClient):
    def __init__(self):
        super().__init__()
        self.requests = []

    def _extract_observation(self, raw_obs, *, env_id=0):
        return raw_obs

    def _pack_request(self, extracted_obs, instruction):
        return {"prompt": instruction}

    def _query_server(self, request):
        self.requests.append(request)
        return {"actions": [[0.0]]}

    def _unpack_response(self, response):
        return np.asarray(response["actions"])


class _VoloClient(OrchestratorMetadataMixin, _Client):
    """Fake proxy client: metadata mixin over the dependency-light base."""


def test_base_client_wire_format_is_unchanged():
    client = _Client()
    client.begin_episode(7)

    client.infer({}, "task")

    assert client.requests[0] == {"prompt": "task"}


def test_volo_mixin_forwards_episode_id_and_metadata():
    client = _VoloClient()
    client.begin_episode(7)

    obs = {
        "image_obs": {
            "over_shoulder_left_camera_pos": _Tensor(
                np.asarray([[9.0, 9.0, 9.0], [1.0, 2.0, 3.0]])
            ),
        },
    }
    client.infer(obs, "task", env_id=1)

    request = client.requests[0]
    assert request["__episode_id"] == 7
    assert request["prompt"] == "task"
    np.testing.assert_array_equal(request["observation/camera_pos"], [1.0, 2.0, 3.0])


def test_begin_episode_is_part_of_the_core_contract():
    # run_episode calls this on every client; the default stores the index
    # that proxy-aware clients forward as __episode_id.
    client = _Client()
    client.begin_episode(5)
    assert client._eval_episode_idx == 5


def test_volo_cosmos3_keeps_backend_session_and_orchestrator_episode_ids():
    client = VoloCosmos3Client.__new__(VoloCosmos3Client)
    InferenceClient.__init__(client)
    client._image_h = 2
    client._image_w = 2
    client.begin_episode(7)
    extracted = {
        "left_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "right_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "joint_position": np.zeros(7, dtype=np.float32),
        "gripper_position": np.zeros(1, dtype=np.float32),
        "session_id": "robolab-episode-7-env-3",
        "_orchestrator_keys": {},
    }

    request = client._pack_request(extracted, "pick")

    assert request["session_id"] == "robolab-episode-7-env-3"
    assert request["__episode_id"] == 7


def test_orchestrator_keys_are_optional():
    assert _VoloClient()._orchestrator_keys({}) == {}


def test_core_client_has_no_volo_wire_knowledge():
    # The wire map and its collection logic are VoLo-owned; the core ABC
    # exposes only generic observation helpers.
    assert not hasattr(InferenceClient, "ORCHESTRATOR_KEY_MAP")
    assert not hasattr(InferenceClient, "_orchestrator_keys")


def test_orchestrator_keys_forward_camera_depth_rgb_and_gt_state():
    client = _VoloClient()
    env1_state = {"objects": {"cup": {"pos": [0.1, 0.2, 0.3]}}}
    gt_state = {0: {"objects": {}}, 1: env1_state}
    obs = {
        "image_obs": {
            "over_shoulder_left_camera_depth": _Tensor(
                np.ones((2, 3, 4, 1), dtype=np.float32)
            ),
            "over_shoulder_left_camera_pos": _Tensor(
                np.asarray([[9.0, 9.0, 9.0], [1.0, 2.0, 3.0]])
            ),
            "over_shoulder_left_camera_quat": _Tensor(
                np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            ),
            "over_shoulder_left_camera_K": _Tensor(
                np.stack([np.eye(3), np.eye(3) * 2.0])
            ),
        },
        "viewport_cam": {
            "egocentric_mirrored_camera": _Tensor(
                np.zeros((2, 5, 6, 3), dtype=np.uint8)
            ),
            "egocentric_mirrored_camera_depth": _Tensor(
                np.full((2, 5, 6, 1), 2.0, dtype=np.float32)
            ),
            "egocentric_mirrored_camera_pos": _Tensor(
                np.asarray([[9.0, 9.0, 9.0], [4.0, 5.0, 6.0]])
            ),
        },
        "gt_state": gt_state,
    }

    result = client._orchestrator_keys(obs, env_id=1)

    assert result["observation/depth_external"].shape == (3, 4, 1)
    assert result["observation/depth_front"].shape == (5, 6, 1)
    assert result["observation/front_image_left_raw"].shape == (5, 6, 3)
    # Camera metadata is per-env: env_id=1 selects the second row.
    np.testing.assert_array_equal(result["observation/camera_pos"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result["observation/camera_quat"], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(result["observation/camera_K"], np.eye(3) * 2.0)
    np.testing.assert_array_equal(
        result["observation/camera_pos_front"], [4.0, 5.0, 6.0]
    )
    # GT state is keyed per env; only this env's entry is forwarded, with
    # the mixin's derived fields added on top of the raw snapshot.
    forwarded = result["gt_state"]
    np.testing.assert_array_equal(forwarded["objects"]["cup"]["pos"], [0.1, 0.2, 0.3])
    assert forwarded["objects"]["cup"]["lifted"] is False
    # The shared snapshot itself is not mutated.
    assert "lifted" not in env1_state["objects"]["cup"]


def test_gt_state_missing_env_entry_is_skipped():
    client = _VoloClient()

    result = client._orchestrator_keys({"gt_state": {0: {"objects": {}}}}, env_id=1)

    assert "gt_state" not in result


def _raw_state(pos, closedness=0.0, contacts=()):
    """Raw core-exporter snapshot: poses only, no derived fields."""
    return {
        "objects": {"banana": {"pos": np.asarray(pos, dtype=np.float32),
                               "quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                               "vel": np.zeros(6, dtype=np.float32)}},
        "robot": {"ee_pos": np.zeros(3, dtype=np.float32),
                  "ee_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                  "gripper_closedness": np.float32(closedness),
                  "objects_in_contact": list(contacts)},
        "subtask": {}, "scene_objects": ["banana"], "step": 1,
    }


def _send(client, state, env_id):
    client.infer({"gt_state": {env_id: state}}, "task", env_id=env_id)
    return client.requests[-1]["gt_state"]


def test_mixin_derives_lift_tracking_per_env():
    client = _VoloClient()
    client.begin_episode(0)

    first = _send(client, _raw_state([0.0, 0.0, 0.10]), env_id=0)
    _send(client, _raw_state([0.0, 0.0, 0.10]), env_id=1)
    raised = _send(client, _raw_state([0.0, 0.0, 0.15]), env_id=0)
    lowered = _send(client, _raw_state([0.0, 0.0, 0.10]), env_id=0)
    other_env = _send(client, _raw_state([0.0, 0.0, 0.10]), env_id=1)

    banana = first["objects"]["banana"]
    assert banana["z_lift"] == 0.0 and not banana["lifted"]
    banana = raised["objects"]["banana"]
    np.testing.assert_allclose(banana["z_lift"], 0.05, atol=1e-6)
    np.testing.assert_allclose(banana["displacement"], 0.05, atol=1e-6)
    assert banana["lifted"]
    # Back down: z_lift returns to 0 but max_z_lift and lifted persist.
    banana = lowered["objects"]["banana"]
    np.testing.assert_allclose(banana["z_lift"], 0.0, atol=1e-6)
    np.testing.assert_allclose(banana["max_z_lift"], 0.05, atol=1e-6)
    assert banana["lifted"]
    # Env 1 never moved: its tracking is independent of env 0's.
    assert not other_env["objects"]["banana"]["lifted"]


def test_mixin_gates_grasp_on_gripper_closure():
    client = _VoloClient()
    client.begin_episode(0)

    open_grip = _send(client, _raw_state([0, 0, 0.1], closedness=0.2, contacts=["banana"]), env_id=0)
    closed_grip = _send(client, _raw_state([0, 0, 0.1], closedness=0.3, contacts=["banana"]), env_id=0)

    # Open gripper: contact suppressed, matching the historical wire format.
    assert open_grip["robot"]["grasped_object"] is None
    assert open_grip["robot"]["objects_in_contact"] == []
    assert closed_grip["robot"]["grasped_object"] == "banana"
    assert closed_grip["robot"]["objects_in_contact"] == ["banana"]


def test_mixin_reset_clears_one_env_only():
    client = _VoloClient()
    client.begin_episode(0)

    _send(client, _raw_state([0.0, 0.0, 0.10]), env_id=0)
    _send(client, _raw_state([0.0, 0.0, 0.10]), env_id=1)
    _send(client, _raw_state([0.0, 0.0, 0.15]), env_id=0)
    _send(client, _raw_state([0.0, 0.0, 0.15]), env_id=1)

    client.reset(env_id=1)

    # Env 1 re-anchors at its current height; env 0 keeps its history.
    env1 = _send(client, _raw_state([0.0, 0.0, 0.15]), env_id=1)
    env0 = _send(client, _raw_state([0.0, 0.0, 0.15]), env_id=0)
    assert not env1["objects"]["banana"]["lifted"]
    assert env0["objects"]["banana"]["lifted"]


def test_mixin_does_not_mutate_the_shared_snapshot():
    client = _VoloClient()
    client.begin_episode(0)
    state = _raw_state([0.0, 0.0, 0.10], closedness=0.9, contacts=["banana"])

    _send(client, state, env_id=0)

    assert "lifted" not in state["objects"]["banana"]
    assert "grasped_object" not in state["robot"]
