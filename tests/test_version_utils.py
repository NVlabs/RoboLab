# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay warnings for simulator and GPU differences."""

from robolab.core.utils import version_utils


def _mock_versions(monkeypatch, recorded, current_gpu):
    monkeypatch.setattr(version_utils, "load_hdf5_provenance", lambda *_args, **_kwargs: recorded)
    monkeypatch.setattr(version_utils, "get_gpu_info", lambda _device=None: current_gpu)
    monkeypatch.setattr(
        version_utils.importlib.metadata,
        "version",
        lambda package: {"isaaclab": "2.2.0", "isaacsim": "5.0.0"}[package],
    )


def test_matching_recording_environment_prints_no_warning(monkeypatch, capsys):
    gpu_info = {"gpu_name": "GPU A", "cuda_version": "12.8", "driver_version": "550.1"}
    _mock_versions(
        monkeypatch,
        {"isaaclab_version": "2.2.0", "isaacsim_version": "5.0.0", **gpu_info},
        gpu_info,
    )

    version_utils.warn_on_recording_mismatch("recording.hdf5", episode=3, device="cuda:1")

    assert capsys.readouterr().out == ""


def test_simulator_and_gpu_differences_are_reported(monkeypatch, capsys):
    _mock_versions(
        monkeypatch,
        {
            "isaaclab_version": "2.1.0",
            "isaacsim_version": "5.0.0",
            "gpu_name": "GPU A",
            "cuda_version": "12.8",
            "driver_version": "550.1",
        },
        {"gpu_name": "GPU B", "cuda_version": "12.9", "driver_version": "560.2"},
    )

    version_utils.warn_on_recording_mismatch("recording.hdf5", device="cuda:0")

    output = capsys.readouterr().out
    assert "recording used isaaclab 2.1.0" in output
    assert "GPU: recorded GPU A, current GPU B" in output
    assert "PyTorch CUDA build: recorded 12.8, current 12.9" in output
    assert "NVIDIA driver: recorded 550.1, current 560.2" in output


def test_selected_episode_and_device_are_forwarded(monkeypatch):
    calls = {}

    def _load(path, episode=None):
        calls["load"] = (path, episode)
        return {
            "isaaclab_version": "2.2.0",
            "isaacsim_version": "5.0.0",
            "gpu_name": "GPU A",
        }

    def _gpu_info(device=None):
        calls["gpu"] = device
        return {"gpu_name": "GPU A"}

    monkeypatch.setattr(version_utils, "load_hdf5_provenance", _load)
    monkeypatch.setattr(version_utils, "get_gpu_info", _gpu_info)
    monkeypatch.setattr(
        version_utils.importlib.metadata,
        "version",
        lambda package: {"isaaclab": "2.2.0", "isaacsim": "5.0.0"}[package],
    )

    version_utils.warn_on_recording_mismatch("recording.hdf5", episode=4, device="cuda:1")

    assert calls == {"load": ("recording.hdf5", 4), "gpu": "cuda:1"}


def test_gpu_recording_replayed_on_cpu_warns(monkeypatch, capsys):
    _mock_versions(
        monkeypatch,
        {"isaaclab_version": "2.2.0", "isaacsim_version": "5.0.0", "gpu_name": "GPU A"},
        {},
    )

    version_utils.warn_on_recording_mismatch("recording.hdf5", device="cpu")

    assert "replay is configured to use the CPU" in capsys.readouterr().out


def test_legacy_recording_warns_without_returning_early(monkeypatch, capsys):
    _mock_versions(monkeypatch, {}, {"gpu_name": "GPU A"})

    version_utils.warn_on_recording_mismatch("recording.hdf5", device="cuda:0")

    output = capsys.readouterr().out
    assert "no simulator-version information" in output
    assert "no GPU information" in output
