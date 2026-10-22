# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for Isaac Lab 2.x and 3.x.

RoboLab's public and recorded quaternion convention remains ``(w, x, y, z)``.
Isaac Lab 3.x changed its simulation, sensor, and math APIs to ``(x, y, z, w)``
and exposes most state through ``ProxyArray`` objects.  Keeping those details in
one module makes recordings portable between Isaac Sim 5 and 6.

This module deliberately does not import Isaac Lab.  It is safe to import before
``AppLauncher`` creates the Isaac Sim application.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _uses_xyzw_quaternions() -> bool:
    """Return whether the installed Isaac Lab uses its 3.x XYZW convention."""
    try:
        major = int(version("isaaclab").split(".", 1)[0])
        # Isaac Lab's 3.0 beta wheels use the 6.x unified package version.
        return major >= 3
    except (PackageNotFoundError, ValueError):
        # NGC development images may expose Isaac Lab from a source checkout
        # without distribution metadata. ProxyArray was introduced with the
        # XYZW migration, and checking its path does not import Isaac Lab before
        # AppLauncher starts Kit.
        try:
            spec = find_spec("isaaclab")
        except ValueError:
            # A stand-in module named ``isaaclab`` with no ``__spec__`` (the offline tests
            # register one so detector code imports without Isaac Sim): not the 3.x stack.
            return False
        locations = spec.submodule_search_locations if spec is not None else None
        return bool(
            locations
            and any((Path(location) / "utils" / "warp" / "proxy_array.py").is_file() for location in locations)
        )


ISAACLAB_USES_XYZW = _uses_xyzw_quaternions()


def as_torch(value: Any) -> torch.Tensor:
    """Return a tensor view for a tensor, ProxyArray, or Warp array."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(value)
    tensor = getattr(value, "torch", None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    try:
        import warp as wp

        return wp.to_torch(value)
    except (ImportError, RuntimeError, TypeError) as exc:
        raise TypeError(f"Cannot convert {type(value).__name__} to torch.Tensor") from exc


def _reorder_quaternion(value: Any, indices: tuple[int, int, int, int]):
    """Reorder the last dimension without changing the container type."""
    if isinstance(value, torch.Tensor):
        return value[..., list(indices)]
    if isinstance(value, np.ndarray):
        return value[..., list(indices)]
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices)
    if isinstance(value, list):
        return [value[index] for index in indices]
    tensor = as_torch(value)
    return tensor[..., list(indices)]


def quat_wxyz_to_isaaclab(value: Any, *, uses_xyzw: bool | None = None):
    """Convert a RoboLab WXYZ quaternion to the installed Isaac Lab convention."""
    if uses_xyzw is None:
        uses_xyzw = ISAACLAB_USES_XYZW
    return _reorder_quaternion(value, (1, 2, 3, 0)) if uses_xyzw else value


def quat_isaaclab_to_wxyz(value: Any, *, uses_xyzw: bool | None = None):
    """Convert an Isaac Lab quaternion to RoboLab's stable WXYZ convention."""
    if uses_xyzw is None:
        uses_xyzw = ISAACLAB_USES_XYZW
    return _reorder_quaternion(value, (3, 0, 1, 2)) if uses_xyzw else value


def _reorder_pose(value: Any, quat_indices: tuple[int, int, int, int]):
    if isinstance(value, torch.Tensor):
        return torch.cat((value[..., :3], value[..., 3:7][..., list(quat_indices)]), dim=-1)
    if isinstance(value, np.ndarray):
        return np.concatenate((value[..., :3], value[..., 3:7][..., list(quat_indices)]), axis=-1)
    if isinstance(value, tuple):
        return tuple(value[:3]) + tuple(value[3:7][index] for index in quat_indices)
    if isinstance(value, list):
        return list(value[:3]) + [value[3:7][index] for index in quat_indices]
    return _reorder_pose(as_torch(value), quat_indices)


def pose_wxyz_to_isaaclab(value: Any, *, uses_xyzw: bool | None = None):
    """Convert a ``(..., 7)`` XYZ+WXYZ pose to the Isaac Lab convention."""
    if uses_xyzw is None:
        uses_xyzw = ISAACLAB_USES_XYZW
    return _reorder_pose(value, (1, 2, 3, 0)) if uses_xyzw else value


def pose_isaaclab_to_wxyz(value: Any, *, uses_xyzw: bool | None = None):
    """Convert a ``(..., 7)`` Isaac Lab pose to XYZ+WXYZ."""
    if uses_xyzw is None:
        uses_xyzw = ISAACLAB_USES_XYZW
    return _reorder_pose(value, (3, 0, 1, 2)) if uses_xyzw else value


def _convert_scene_state(state: dict, *, to_isaaclab: bool) -> dict:
    """Copy a scene-state tree while converting its root-pose quaternions."""
    converted = {}
    for family, assets in state.items():
        if not isinstance(assets, dict):
            converted[family] = assets
            continue
        converted[family] = {}
        for asset_name, asset_state in assets.items():
            if not isinstance(asset_state, dict):
                converted[family][asset_name] = asset_state
                continue
            values = dict(asset_state)
            if family in {"articulation", "rigid_object"} and "root_pose" in values:
                converter = pose_wxyz_to_isaaclab if to_isaaclab else pose_isaaclab_to_wxyz
                values["root_pose"] = converter(values["root_pose"])
            converted[family][asset_name] = values
    return converted


def scene_state_to_isaaclab(state: dict) -> dict:
    """Convert a recorded/public scene state before passing it to Isaac Lab."""
    return _convert_scene_state(state, to_isaaclab=True)


def scene_state_from_isaaclab(state: dict) -> dict:
    """Convert an Isaac Lab scene state to RoboLab's stable recording format."""
    return _convert_scene_state(state, to_isaaclab=False)


def _convert_config_rotations(value: Any, *, to_isaaclab: bool, seen: set[int]) -> None:
    """Recursively convert config fields named ``rot`` in place."""
    if value is None or isinstance(value, (str, bytes, int, float, bool, torch.Tensor, np.ndarray)):
        return
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    converter = quat_wxyz_to_isaaclab if to_isaaclab else quat_isaaclab_to_wxyz
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "rot" and isinstance(child, (tuple, list)) and len(child) == 4:
                value[key] = converter(child)
            else:
                _convert_config_rotations(child, to_isaaclab=to_isaaclab, seen=seen)
        return
    if isinstance(value, list):
        for child in value:
            _convert_config_rotations(child, to_isaaclab=to_isaaclab, seen=seen)
        return
    if isinstance(value, tuple):
        for child in value:
            _convert_config_rotations(child, to_isaaclab=to_isaaclab, seen=seen)
        return
    if hasattr(value, "__dict__"):
        for key, child in vars(value).items():
            if key == "rot" and isinstance(child, (tuple, list)) and len(child) == 4:
                setattr(value, key, converter(child))
            else:
                _convert_config_rotations(child, to_isaaclab=to_isaaclab, seen=seen)


def prepare_env_cfg(env_cfg: Any) -> Any:
    """Convert authored WXYZ config rotations for Isaac Lab 3, once per config."""
    if not ISAACLAB_USES_XYZW or getattr(env_cfg, "_robolab_quaternions_prepared", False):
        return env_cfg
    _convert_config_rotations(env_cfg, to_isaaclab=True, seen=set())
    env_cfg._robolab_quaternions_prepared = True
    return env_cfg


def env_cfg_to_recording_dict(env_cfg: Any) -> dict:
    """Serialize a config with version-independent WXYZ ``rot`` fields."""
    result = env_cfg.to_dict()
    if ISAACLAB_USES_XYZW:
        _convert_config_rotations(result, to_isaaclab=False, seen=set())
    result.pop("_robolab_quaternions_prepared", None)
    return result


def write_root_pose(asset: Any, root_pose: torch.Tensor, env_ids=None) -> None:
    if hasattr(asset, "write_root_pose_to_sim_index"):
        asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    else:
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)


def write_root_velocity(asset: Any, root_velocity: torch.Tensor, env_ids=None) -> None:
    if hasattr(asset, "write_root_velocity_to_sim_index"):
        asset.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
    else:
        asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def write_joint_state(asset: Any, position: torch.Tensor, velocity: torch.Tensor, env_ids=None) -> None:
    if hasattr(asset, "write_joint_state_to_sim_index"):
        asset.write_joint_state_to_sim_index(position=position, velocity=velocity, env_ids=env_ids)
    else:
        asset.write_joint_state_to_sim(position, velocity, env_ids=env_ids)


def write_nodal_state(asset: Any, nodal_state: torch.Tensor, env_ids=None) -> None:
    if hasattr(asset, "write_nodal_state_to_sim_index"):
        asset.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)
    else:
        asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
