# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording-environment checks used by replay.

Recordings carry simulator versions and GPU information as HDF5 attributes.
These checks compare them with the current replay environment.
"""

import importlib.metadata

from robolab.core.utils.file_utils import load_hdf5_provenance
from robolab.core.utils.hardware_utils import GPU_INFO_KEYS, get_gpu_info


def _warn(message: str) -> None:
    print(f"\033[93mNOTE: {message}\033[0m")


def warn_on_recording_mismatch(
    hdf5_path: str,
    *,
    episode: int | None = None,
    device: str | None = None,
) -> None:
    """Warn when recording and replay environments may produce different physics.

    Missing attributes are supported for older recordings. When an episode is
    selected, its GPU attributes take precedence over the file-level values so
    files resumed on another machine remain accurately described.
    """
    recorded = load_hdf5_provenance(hdf5_path, episode=episode)
    if not recorded.get("isaaclab_version") and not recorded.get("isaacsim_version"):
        _warn(
            "recording has no simulator-version information. If it was recorded "
            "with a different IsaacSim/IsaacLab stack, replay may diverge."
        )
    else:
        for package in ("isaaclab", "isaacsim"):
            recorded_version = recorded.get(f"{package}_version")
            try:
                current_version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                continue
            if recorded_version is not None and str(recorded_version) != current_version:
                _warn(
                    f"recording used {package} {recorded_version}, but the current environment "
                    f"uses {package} {current_version}. Replay may diverge."
                )

    recorded_gpu = {key: str(recorded[key]) for key in GPU_INFO_KEYS if recorded.get(key) is not None}
    if not recorded_gpu:
        _warn("recording has no GPU information; hardware compatibility cannot be checked.")
        return

    current_gpu = get_gpu_info(device)
    if not current_gpu:
        if device is not None and str(device).lower().startswith("cpu"):
            detail = "the replay is configured to use the CPU"
        else:
            detail = "the current GPU could not be identified"
        _warn(f"recording used a CUDA GPU, but {detail}. Replay may diverge.")
        return

    labels = {
        "gpu_name": "GPU",
        "cuda_version": "PyTorch CUDA build",
        "driver_version": "NVIDIA driver",
    }
    differences = [
        f"{labels[key]}: recorded {recorded_gpu[key]}, current {current_gpu[key]}"
        for key in GPU_INFO_KEYS
        if key in recorded_gpu and key in current_gpu and recorded_gpu[key] != current_gpu[key]
    ]
    if differences:
        _warn("GPU environment differs (" + "; ".join(differences) + "). Replay may diverge.")


def warn_on_stack_mismatch(hdf5_path: str) -> None:
    """Backward-compatible wrapper for the original simulator-only check."""
    warn_on_recording_mismatch(hdf5_path)
