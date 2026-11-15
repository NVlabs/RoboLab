# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU information queries and HDF5 recording integration."""

import subprocess
from types import SimpleNamespace

import h5py
import pytest
import torch

from robolab.core.logging import recorder_manager
from robolab.core.logging.streaming_hdf5_handler import StreamingHDF5DatasetFileHandler
from robolab.core.utils import hardware_utils
from robolab.core.utils.file_utils import load_hdf5_provenance


def _mock_cuda(monkeypatch, *, count: int = 2, current: int = 0):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)


def test_gpu_info_uses_selected_nonzero_cuda_device(monkeypatch):
    _mock_cuda(monkeypatch)
    requested_indices = []

    def _get_device_name(index):
        requested_indices.append(index)
        return f"GPU {index}"

    monkeypatch.setattr(torch.cuda, "get_device_name", _get_device_name)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(hardware_utils, "get_driver_version", lambda: "550.90.07")

    assert hardware_utils.get_gpu_info("cuda:1") == {
        "gpu_name": "GPU 1",
        "cuda_version": "12.8",
        "driver_version": "550.90.07",
    }
    assert requested_indices == [1]


def test_gpu_info_uses_current_device_when_index_is_omitted(monkeypatch):
    _mock_cuda(monkeypatch, current=1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU {index}")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(hardware_utils, "get_driver_version", lambda: "550.90.07")

    assert hardware_utils.get_gpu_info("cuda")["gpu_name"] == "GPU 1"


@pytest.mark.parametrize("device", ["cpu", "cuda:2", "not-a-device"])
def test_gpu_info_returns_empty_for_non_cuda_or_invalid_device(monkeypatch, device):
    _mock_cuda(monkeypatch)
    monkeypatch.setattr(
        hardware_utils,
        "get_driver_version",
        lambda: pytest.fail("driver should not be queried without a selected CUDA GPU"),
    )

    assert hardware_utils.get_gpu_info(device) == {}


def test_gpu_info_returns_empty_without_a_usable_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", "12.8")

    assert hardware_utils.get_gpu_info() == {}


def test_gpu_info_omits_individual_query_failures(monkeypatch):
    monkeypatch.setattr(hardware_utils, "_cuda_device_index", lambda _device: 0)
    monkeypatch.setattr(
        hardware_utils,
        "get_gpu_name",
        lambda _device: (_ for _ in ()).throw(RuntimeError("name unavailable")),
    )
    monkeypatch.setattr(hardware_utils, "get_cuda_version", lambda _device: "12.8")
    monkeypatch.setattr(hardware_utils, "get_driver_version", lambda: None)

    assert hardware_utils.get_gpu_info("cuda:0") == {"cuda_version": "12.8"}


def test_driver_version_falls_back_when_nvml_probe_raises(monkeypatch):
    monkeypatch.setattr(
        hardware_utils,
        "_get_driver_version_from_nvml",
        lambda: (_ for _ in ()).throw(AttributeError("missing NVML symbol")),
    )
    monkeypatch.setattr(hardware_utils, "_get_driver_version_from_nvidia_smi", lambda: "550.90.07")

    assert hardware_utils.get_driver_version() == "550.90.07"


def test_nvml_probe_handles_missing_symbols(monkeypatch):
    monkeypatch.setattr(hardware_utils.ctypes, "CDLL", lambda _name: object())

    assert hardware_utils._get_driver_version_from_nvml() is None


def test_nvidia_smi_probe_handles_timeout(monkeypatch):
    monkeypatch.setattr(hardware_utils.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        hardware_utils.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("nvidia-smi", 5)),
    )

    assert hardware_utils._get_driver_version_from_nvidia_smi() is None


def test_nvidia_smi_probe_returns_first_nonempty_version(monkeypatch):
    monkeypatch.setattr(hardware_utils.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        hardware_utils.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="\n550.90.07\n550.90.07\n"),
    )

    assert hardware_utils._get_driver_version_from_nvidia_smi() == "550.90.07"


def _write_empty_episode(file_path, gpu_info, episode):
    handler = StreamingHDF5DatasetFileHandler()
    handler.create(str(file_path), env_name="TestTask", gpu_info=gpu_info)
    handler.begin_episode(seed=episode, episode_index=episode)
    handler.end_episode(success=True, episode_index=episode)
    handler.close()


def test_handler_stamps_file_and_episode_gpu_info(tmp_path):
    file_path = tmp_path / "recording.hdf5"
    gpu_info = {
        "gpu_name": "GPU A",
        "cuda_version": "12.8",
        "driver_version": "550.90.07",
    }

    _write_empty_episode(file_path, gpu_info, episode=0)

    file_info = load_hdf5_provenance(str(file_path))
    episode_info = load_hdf5_provenance(str(file_path), episode=0)
    assert {key: file_info[key] for key in gpu_info} == gpu_info
    assert {key: episode_info[key] for key in gpu_info} == gpu_info
    with h5py.File(file_path, "r") as file:
        assert "recorded_at" in file["data"].attrs
        assert "recorded_at" in file["data/demo_0"].attrs


def test_resumed_file_uses_per_episode_gpu_info(tmp_path):
    file_path = tmp_path / "recording.hdf5"
    gpu_a = {"gpu_name": "GPU A", "cuda_version": "12.8", "driver_version": "550.1"}
    gpu_b = {"gpu_name": "GPU B", "cuda_version": "12.9", "driver_version": "560.2"}

    _write_empty_episode(file_path, gpu_a, episode=0)
    _write_empty_episode(file_path, gpu_b, episode=1)

    assert load_hdf5_provenance(str(file_path))["gpu_name"] == "GPU A"
    for episode, expected in ((0, gpu_a), (1, gpu_b)):
        loaded = load_hdf5_provenance(str(file_path), episode=episode)
        assert {key: loaded[key] for key in expected} == expected


def test_episode_lookup_falls_back_to_file_level_gpu_info(tmp_path):
    file_path = tmp_path / "old_recording.hdf5"
    with h5py.File(file_path, "w") as file:
        data = file.create_group("data")
        data.attrs["gpu_name"] = "GPU A"
        data.create_group("demo_0")

    assert load_hdf5_provenance(str(file_path), episode=0)["gpu_name"] == "GPU A"


def test_explicit_empty_gpu_info_does_not_probe_host_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "robolab.core.logging.streaming_hdf5_handler.get_gpu_info",
        lambda: pytest.fail("host GPU should not be queried for a CPU recording"),
    )
    file_path = tmp_path / "cpu_recording.hdf5"

    _write_empty_episode(file_path, {}, episode=0)

    assert not {"gpu_name", "cuda_version", "driver_version"} & set(
        load_hdf5_provenance(str(file_path), episode=0)
    )


def test_recorder_manager_queries_the_simulation_device_once(monkeypatch, tmp_path):
    calls = []

    class _Handler:
        def close(self):
            pass

        def create(self, file_path, env_name=None, gpu_info=None):
            calls.append((file_path, env_name, gpu_info))

    manager = object.__new__(recorder_manager.RobolabRecorderManager)
    manager.cfg = SimpleNamespace(
        dataset_export_mode=recorder_manager.DatasetExportMode.EXPORT_ALL,
        dataset_export_dir_path=str(tmp_path),
        dataset_filename="recording",
    )
    manager._recording_device = "cuda:1"
    manager._gpu_info = None
    manager._env_name = "TestTask"
    manager._dataset_file_handler = None
    manager._failed_episode_dataset_file_handler = None
    manager._hdf5_initialized = False

    queried_devices = []

    def _get_gpu_info(device):
        queried_devices.append(device)
        return {"gpu_name": "GPU 1"}

    monkeypatch.setattr(recorder_manager, "get_gpu_info", _get_gpu_info)
    monkeypatch.setattr(recorder_manager, "StreamingHDF5DatasetFileHandler", _Handler)

    manager._ensure_hdf5_handler()
    manager._ensure_hdf5_handler("second")

    assert queried_devices == ["cuda:1"]
    assert calls == [
        (str(tmp_path / "recording"), "TestTask", {"gpu_name": "GPU 1"}),
        (str(tmp_path / "second"), "TestTask", {"gpu_name": "GPU 1"}),
    ]


def test_legacy_file_without_provenance_yields_empty_dict(tmp_path):
    file_path = tmp_path / "legacy.hdf5"
    with h5py.File(file_path, "w") as f:
        f.create_group("data").attrs["total"] = 0
    assert load_hdf5_provenance(str(file_path)) == {}
