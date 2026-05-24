# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Extract per-episode time-series from a RoboLab data.hdf5.

The HDF5 layout written by IsaacLab's recorder is
``/data/<episode_key>/{obs/<name>, actions, ...}`` where each leaf is a
[T, D] array. We surface a handful of well-known signals; everything is
downsampled to a max length to keep JSON payloads cheap.
"""

from pathlib import Path

import h5py
import numpy as np

# obs signals we know about; first match wins. Used for the IsaacLab recorder layout
# where /data/<ep>/obs/proprio_obs/<name>.
_OBS_SIGNALS = (
    ("proprio_obs/arm_joint_pos", "arm_joint_pos"),
    ("proprio_obs/gripper_pos", "gripper_pos"),
    ("proprio_obs/eef_pos", "eef_pos"),
    ("proprio_obs/eef_quat", "eef_quat"),
    ("proprio_obs/ee_pos", "ee_pos"),
    ("proprio_obs/ee_quat", "ee_quat"),
)

# Flat per-episode layout: episode-level datasets, no /obs group.
_FLAT_SIGNALS = ("ee_pose", "states", "bbox")

# ee_pose/* gets surfaced as flat ee_<leaf> series.
_EE_POSE_SIGNALS = (
    ("ee_pose/position",          "ee_position"),
    ("ee_pose/orientation",       "ee_orientation"),
    ("ee_pose/linear_velocity",   "ee_linear_velocity"),
    ("ee_pose/angular_velocity",  "ee_angular_velocity"),
)


def _state_label(parts: tuple[str, ...]) -> str:
    """Cleaned chart label for a path under ``states/``.

    Drop boilerplate intermediate group names so a path like
    ``articulation/robot/joint_position`` reads as ``state.robot.joint_position``
    rather than the full nested form. Same for ``rigid_object/<obj>/<name>``.
    """
    cleaned = [p for p in parts if p not in ("articulation", "rigid_object")]
    return "state." + ".".join(cleaned)


def _walk_state_group(group, prefix: tuple[str, ...] = ()) -> list[tuple[str, "h5py.Dataset"]]:
    """Yield (label, dataset) pairs for every 2D+ leaf under a states group."""
    import h5py
    out: list[tuple[str, "h5py.Dataset"]] = []
    for k in group.keys():
        child = group[k]
        new_prefix = prefix + (k,)
        if isinstance(child, h5py.Group):
            out.extend(_walk_state_group(child, new_prefix))
        elif isinstance(child, h5py.Dataset) and child.ndim >= 2:
            out.append((_state_label(new_prefix), child))
    return out


def list_episode_keys(path: Path) -> list[str]:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            return []
        return list(f["data"].keys())


def episode_timeseries(path: Path, episode_key: str, max_points: int = 400,
                       dt_override: float | None = None) -> dict:
    """Return downsampled time-series for a single episode.

    Output shape::

        {
            "dt": float,
            "num_steps": int,
            "series": {
                "<signal>": {"labels": ["d0", ...], "data": [[t, v0, v1, ...], ...]}
            }
        }
    """
    out: dict = {"dt": None, "num_steps": 0, "series": {}}
    with h5py.File(path, "r") as f:
        if "data" not in f or episode_key not in f["data"]:
            return out
        ep = f["data"][episode_key]
        # dt comes from env_cfg.json (sim.dt × decimation) when provided —
        # that's the authoritative engine timestep. The HDF5 attribute is a
        # last resort.
        if dt_override:
            dt = float(dt_override)
        else:
            dt = float(ep.attrs.get("dt", 0.0)) or None
        out["dt"] = dt

        def add(name: str, arr: np.ndarray) -> None:
            t = arr.shape[0]
            if t == 0:
                return
            stride = max(1, t // max_points)
            sampled = arr[::stride]
            ts = np.arange(0, t, stride, dtype=float)
            if dt:
                ts = ts * dt
            labels = [f"d{i}" for i in range(sampled.shape[1] if sampled.ndim > 1 else 1)]
            cols = sampled if sampled.ndim > 1 else sampled.reshape(-1, 1)
            data = np.concatenate([ts.reshape(-1, 1), cols.astype(float)], axis=1)
            out["series"][name] = {"labels": labels, "data": data.tolist()}

        if "obs" in ep:
            obs = ep["obs"]
            for key, label in _OBS_SIGNALS:
                try:
                    arr = obs[key][...]
                except (KeyError, ValueError):
                    continue
                if arr.ndim >= 2:
                    add(label, arr)
        if "actions" in ep:
            arr = ep["actions"][...]
            if arr.ndim >= 2:
                add("actions", arr)
        # Flat per-episode datasets (ee_pose/bbox handled as groups via
        # _EE_POSE_SIGNALS / object walks below; only `states` may be flat).
        flat_states = ep.get("states")
        if isinstance(flat_states, h5py.Dataset) and flat_states.ndim >= 2:
            add("state", flat_states[...])
        # pi05-style nested states/ — walk and emit one chart per leaf.
        if isinstance(flat_states, h5py.Group):
            for label, ds in _walk_state_group(flat_states):
                try:
                    arr = ds[...]
                except (KeyError, ValueError):
                    continue
                if arr.ndim >= 2:
                    add(label, arr)
        # ee_pose/ leaves.
        for path, label in _EE_POSE_SIGNALS:
            try:
                node = ep[path]
            except KeyError:
                continue
            if not isinstance(node, h5py.Dataset):
                continue
            try:
                arr = node[...]
            except (KeyError, ValueError):
                continue
            if arr.ndim >= 2:
                add(label, arr)

        # num steps = first signal length
        if out["series"]:
            first = next(iter(out["series"].values()))
            out["num_steps"] = len(first["data"])
    return out
