# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort information about the GPU used by a recording."""

import ctypes
import shutil
import subprocess
from collections.abc import Callable

_NVML_DRIVER_VERSION_BUFFER_SIZE = 80
GPU_INFO_KEYS = ("gpu_name", "cuda_version", "driver_version")


def _nonempty_string(query: Callable[[], object]) -> str | None:
    """Run an optional information query without letting it break recording."""
    try:
        value = query()
    except Exception:
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _cuda_device_index(device: str | None) -> int | None:
    """Resolve a CUDA device string to a valid logical device index."""
    try:
        import torch

        resolved = torch.device(device or "cuda")
        if resolved.type != "cuda" or not torch.cuda.is_available():
            return None
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            return None
        return index
    except Exception:
        return None


def get_gpu_name(device: str | None = None) -> str | None:
    """Return the selected CUDA device's model name, or None when unavailable."""
    index = _cuda_device_index(device)
    if index is None:
        return None

    def _query() -> object:
        import torch

        return torch.cuda.get_device_name(index)

    return _nonempty_string(_query)


def get_cuda_version(device: str | None = None) -> str | None:
    """Return PyTorch's CUDA build version when the selected CUDA device is usable."""
    if _cuda_device_index(device) is None:
        return None

    def _query() -> object:
        import torch

        return torch.version.cuda

    return _nonempty_string(_query)


def get_driver_version() -> str | None:
    """Return the installed NVIDIA driver version, or None when unavailable."""
    return _nonempty_string(_get_driver_version_from_nvml) or _nonempty_string(
        _get_driver_version_from_nvidia_smi
    )


def get_gpu_info(device: str | None = None) -> dict[str, str]:
    """Return information about the GPU selected for simulation.

    The returned keys are listed in :data:`GPU_INFO_KEYS`; missing values are
    omitted. A CPU device, invalid CUDA device, or machine without a usable
    CUDA GPU returns an empty dictionary. All queries are best-effort so
    optional metadata can never prevent a recording from being created.
    """
    if _cuda_device_index(device) is None:
        return {}

    info = {
        "gpu_name": _nonempty_string(lambda: get_gpu_name(device)),
        "cuda_version": _nonempty_string(lambda: get_cuda_version(device)),
        "driver_version": _nonempty_string(get_driver_version),
    }
    return {key: value for key, value in info.items() if value is not None}


def _get_driver_version_from_nvml() -> str | None:
    """Query the system-wide driver version through NVML."""
    initialized = False
    shutdown = None
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        init = getattr(nvml, "nvmlInit_v2", None)
        get_version = getattr(nvml, "nvmlSystemGetDriverVersion", None)
        shutdown = getattr(nvml, "nvmlShutdown", None)
        if init is None or get_version is None or shutdown is None:
            return None

        init.argtypes = []
        init.restype = ctypes.c_int
        get_version.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
        get_version.restype = ctypes.c_int
        shutdown.argtypes = []
        shutdown.restype = ctypes.c_int

        if init() != 0:
            return None
        initialized = True

        buffer = ctypes.create_string_buffer(_NVML_DRIVER_VERSION_BUFFER_SIZE)
        if get_version(buffer, len(buffer)) != 0:
            return None
        return buffer.value.decode("utf-8").strip() or None
    except Exception:
        return None
    finally:
        if initialized and shutdown is not None:
            try:
                shutdown()
            except Exception:
                pass


def _get_driver_version_from_nvidia_smi() -> str | None:
    """Query the system-wide driver version through ``nvidia-smi``."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError, UnicodeError):
        return None
    if result.returncode != 0:
        return None
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return versions[0] if versions else None
