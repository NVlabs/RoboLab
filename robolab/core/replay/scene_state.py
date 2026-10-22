# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene-state restore and validation against recorded episodes.

``restore_recorded_initial_state`` puts the scene in the exact state a
recording started from, so an open-loop action replay evolves the same way the
recording did. ``StateValidator`` measures how closely a replay tracks the
recorded per-step states — a debug tool that turns "the replay diverged" from
a guess into a measurement.
"""

import numpy as np
import torch

from robolab.core.utils.file_utils import load_hdf5_initial_state, load_hdf5_states
from robolab.core.utils.isaaclab_compat import scene_state_from_isaaclab, scene_state_to_isaaclab


def restore_recorded_initial_state(env, hdf5_path: str, episode: int) -> None:
    """Reset ``env`` to the initial scene state recorded for ``episode``.

    The recorded ``initial_state`` follows the ``InteractiveScene.get_state()``
    layout with env-relative poses. Row 0 is tiled across ``env.num_envs`` so
    every env replays from the exact state the actions were recorded against.
    The recorded state is overlaid onto the env's current full state because
    ``InteractiveScene.reset_to`` requires an entry for every scene asset,
    while the recorder only saves dynamic assets (e.g. no static table).
    """
    def _overlay(current, recorded):
        for key, value in recorded.items():
            if key not in current:
                print(f"WARNING: recorded initial state has '{key}' which is not in the scene; skipping.")
                continue
            if isinstance(value, dict):
                _overlay(current[key], value)
            else:
                row = torch.as_tensor(value, device=env.device)[0:1]
                current[key] = row.repeat(env.num_envs, *([1] * (row.ndim - 1)))

    try:
        recorded_state = load_hdf5_initial_state(hdf5_path, episode)
    except ValueError as err:
        print(f"WARNING: no recorded initial state to restore ({err}); "
              "replaying from default reset state, which may diverge from the recording.")
        return
    state = scene_state_from_isaaclab(env.scene.get_state(is_relative=True))
    _overlay(state, recorded_state)
    env.reset_to(scene_state_to_isaaclab(state), env_ids=None, is_relative=True)


def _flatten_state_tree(tree: dict, prefix: str = "") -> dict:
    """Flatten a nested ``InteractiveScene.get_state()``-style dict into ``{"a/b/c": leaf}``."""
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_state_tree(value, path))
        else:
            flat[path] = value
    return flat


class StateValidator:
    """Compare per-step sim state against the recorded ``states`` group (debug tool).

    Recorded states are post-step snapshots in the ``InteractiveScene.get_state()``
    layout, one row per step. Comparison uses env 0 only, matching the
    single-env record/replay recipe. Quaternion sign flips (q vs -q) are not
    normalized, so a reported pose drift of ~2.0 on an otherwise tracking
    replay usually means a sign flip, not a real divergence.
    """

    def __init__(self, hdf5_path: str, episode: int, tolerance: float = 0.01):
        self.tolerance = tolerance
        self.recorded = _flatten_state_tree(load_hdf5_states(hdf5_path, episode))
        self.num_steps = min(leaf.shape[0] for leaf in self.recorded.values())
        self.max_drift = 0.0
        self.max_drift_step = None
        self.max_drift_path = None
        self.first_exceed_step = None
        self.paths_over_tolerance = set()

    def check_step(self, env, step: int) -> None:
        if step >= self.num_steps:
            return
        current = _flatten_state_tree(scene_state_from_isaaclab(env.scene.get_state(is_relative=True)))
        for path, series in self.recorded.items():
            if path not in current:
                continue
            simulated = current[path][0].detach().cpu().numpy()
            drift = float(np.max(np.abs(simulated - series[step])))
            if self.max_drift_path is None or drift > self.max_drift:
                self.max_drift, self.max_drift_step, self.max_drift_path = drift, step, path
            if drift > self.tolerance:
                self.paths_over_tolerance.add(path)
                if self.first_exceed_step is None:
                    self.first_exceed_step = step
                    print(f"\033[93mSTATE VALIDATION: drift first exceeded tolerance {self.tolerance} "
                          f"at step {step} ({path}: {drift:.4f}).\033[0m")

    def report(self) -> None:
        if self.first_exceed_step is None:
            print(f"STATE VALIDATION: replay tracked the recording over {self.num_steps} steps; "
                  f"max drift {self.max_drift:.4f} on {self.max_drift_path} (tolerance {self.tolerance}).")
        else:
            over = sorted(self.paths_over_tolerance)
            print(f"\033[93mSTATE VALIDATION: replay diverged from the recording. Max drift "
                  f"{self.max_drift:.4f} at step {self.max_drift_step} on {self.max_drift_path}; "
                  f"first exceeded tolerance {self.tolerance} at step {self.first_exceed_step}. "
                  f"Fields over tolerance ({len(over)}/{len(self.recorded)}): {', '.join(over)}\033[0m")
