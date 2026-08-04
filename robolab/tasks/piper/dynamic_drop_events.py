# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reset-time sequential drop for generated Piper scenes.

The generated USD already contains every object, so cloning and contact
sensors are constructed exactly once.  This event keeps the not-yet-released
objects at off-camera holding poses, releases one object at a time, and drives
only the low-level physics loop.  Consequently the policy cannot act and no
task step/score is recorded until :meth:`env.reset` returns.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence

import torch

logger = logging.getLogger(__name__)


def sequential_drop_reset(env, env_ids: Sequence[int], *, plan: Mapping) -> None:
    """Release dynamic objects serially and settle them before policy reset ends.

    The routine intentionally uses ``env.sim.step`` rather than ``env.step``.
    It therefore bypasses the action, observation, recorder, termination and
    reward managers while setup is underway.  A video callback may be attached
    by :func:`robolab.eval.episode.run_episode` as
    ``env._dynamic_setup_capture``; it receives rendered setup frames only and
    does not record task actions.
    """
    if not getattr(env, "_dynamic_setup_enabled", True):
        return

    object_names = [str(name) for name in plan.get("object_names", [])]
    if not object_names:
        return
    if int(plan.get("version", 0)) != 1:
        raise ValueError(f"Unsupported dynamic sequential-drop plan version: {plan.get('version')}")

    device = env.device
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    if ids.numel() == 0:
        return

    assets = {}
    for name in object_names:
        try:
            asset = env.scene[name]
        except KeyError as exc:
            raise RuntimeError(f"Sequential-drop object '{name}' is not present in env.scene.") from exc
        if not hasattr(asset, "write_root_pose_to_sim") or not hasattr(asset, "write_root_velocity_to_sim"):
            raise RuntimeError(
                f"Sequential-drop object '{name}' is not a dynamic RigidObject. "
                "It must be authored with a rigid body before environment creation."
            )
        assets[name] = asset

    release_poses = _world_poses(env, ids, plan["release_poses"], object_names)
    holding_poses = _world_poses(env, ids, plan["holding_poses"], object_names)
    zero_velocity = torch.zeros((len(ids), 6), dtype=torch.float32, device=device)

    # Start every body in the same safe state.  This is deliberately repeated
    # even though the USD is authored in holding poses: it makes every reset
    # deterministic and protects against a prior episode's final state.
    for name, asset in assets.items():
        asset.write_root_pose_to_sim(holding_poses[name], env_ids=ids)
        asset.write_root_velocity_to_sim(zero_velocity, env_ids=ids)

    # A map of objects still being held for every cloned environment.  It lets
    # each env use a different seeded release order while still batching PhysX
    # writes by object asset.
    unreleased: dict[str, set[int]] = {name: set(int(eid) for eid in ids.tolist()) for name in object_names}
    released: dict[int, set[str]] = {int(eid): set() for eid in ids.tolist()}
    release_orders = _release_orders(object_names, ids.tolist(), plan.get("seed"))

    render_interval = max(1, int(env.cfg.sim.render_interval))
    step_counter = 0
    captured_frames = 0

    def capture() -> None:
        nonlocal captured_frames
        callback = getattr(env, "_dynamic_setup_capture", None)
        if callback is None:
            return
        callback(env)
        captured_frames += 1

    # Record one clean background/holding frame before the first release when
    # requested.  It also guarantees the camera sees the reset poses.
    _hold_robot(env, ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    if getattr(env, "_dynamic_setup_capture", None) is not None:
        env.sim.render()
        env.scene.update(dt=env.physics_dt)
        capture()

    def advance_one() -> None:
        """Advance one physics tick while pinning all unreleased objects."""
        nonlocal step_counter
        _pin_unreleased(assets, holding_poses, unreleased, ids, zero_velocity)
        _hold_robot(env, ids)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        step_counter += 1
        should_render = getattr(env, "_dynamic_setup_capture", None) is not None and step_counter % render_interval == 0
        if should_render:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)
        if should_render:
            capture()

    steps_per_object = int(plan["steps_per_object"])
    stable_frames = int(plan["stable_frames"])
    linear_threshold = float(plan["stable_linear_velocity"])
    angular_threshold = float(plan["stable_angular_velocity"])

    for drop_index in range(len(object_names)):
        # Every environment releases precisely one object for this round.  The
        # selected object can differ across environments because their order is
        # seeded with ``seed + env_id``.
        release_groups: dict[str, list[int]] = {}
        for eid in ids.tolist():
            name = release_orders[int(eid)][drop_index]
            release_groups.setdefault(name, []).append(int(eid))

        for name, group_ids in release_groups.items():
            group = torch.as_tensor(group_ids, dtype=torch.long, device=device)
            group_rows = _rows_for(ids, group)
            assets[name].write_root_pose_to_sim(release_poses[name][group_rows], env_ids=group)
            assets[name].write_root_velocity_to_sim(
                torch.zeros((len(group), 6), dtype=torch.float32, device=device), env_ids=group
            )
            for eid in group_ids:
                unreleased[name].discard(eid)
                released[eid].add(name)

        reached_stability, used_steps = _advance_until_stable(
            advance_one=advance_one,
            assets=assets,
            released=released,
            env_ids=ids,
            max_steps=steps_per_object,
            stable_frames=stable_frames,
            linear_threshold=linear_threshold,
            angular_threshold=angular_threshold,
        )
        names = ", ".join(sorted(release_groups))
        status = "stable" if reached_stability else f"not stable after {used_steps} physics steps"
        print(
            f"\033[96m[RoboLab] Runtime sequential drop {drop_index + 1}/{len(object_names)}: "
            f"{names} ({status})\033[0m"
        )

    # Preserve the legacy --dynamic-settle-final-steps as a guaranteed minimum
    # post-release settle time, then wait for velocity stability up to the
    # explicit cap.  A cap prevents a pathological object from stalling reset.
    for _ in range(int(plan["final_steps"])):
        advance_one()
    final_stable, final_used = _advance_until_stable(
        advance_one=advance_one,
        assets=assets,
        released=released,
        env_ids=ids,
        max_steps=int(plan["max_final_steps"]),
        stable_frames=stable_frames,
        linear_threshold=linear_threshold,
        angular_threshold=angular_threshold,
    )
    if not final_stable:
        logger.warning(
            "Sequential drop reached the final settle cap (%d physics steps) before every object was stable.",
            final_used,
        )

    # Setup uses its own physical clock, never env._sim_step_counter, so it is
    # excluded from task timeout, HDF5 action history and policy timing.
    setup_sim_s = step_counter * float(env.physics_dt)
    video_frame_s = render_interval * float(env.physics_dt)
    env._dynamic_setup_sim_duration_s = {int(eid): setup_sim_s for eid in ids.tolist()}
    env._dynamic_setup_video_duration_s = {
        int(eid): captured_frames * video_frame_s for eid in ids.tolist()
    }
    env._dynamic_setup_steps = {int(eid): step_counter for eid in ids.tolist()}
    env._dynamic_setup_completed = True
    print(
        f"\033[96m[RoboLab] Runtime sequential drop complete: {len(object_names)} object(s), "
        f"{setup_sim_s:.2f}s physics, final={'stable' if final_stable else 'settle-cap'}\033[0m"
    )


def _world_poses(env, env_ids: torch.Tensor, raw_poses: Mapping, object_names: list[str]) -> dict[str, torch.Tensor]:
    """Convert local plan poses into per-env world-frame root poses."""
    poses: dict[str, torch.Tensor] = {}
    origins = env.scene.env_origins[env_ids]
    for name in object_names:
        try:
            entry = raw_poses[name]
            pos = torch.tensor(entry["pos"], dtype=torch.float32, device=env.device)
            rot = torch.tensor(entry["rot"], dtype=torch.float32, device=env.device)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Sequential-drop plan has no valid pose for '{name}'.") from exc
        if pos.numel() != 3 or rot.numel() != 4:
            raise ValueError(f"Sequential-drop pose for '{name}' must be 3D position + wxyz quaternion.")
        world_pos = pos.unsqueeze(0).expand(len(env_ids), -1) + origins
        world_rot = rot.unsqueeze(0).expand(len(env_ids), -1)
        poses[name] = torch.cat((world_pos, world_rot), dim=-1)
    return poses


def _release_orders(object_names: list[str], env_ids: list[int], seed: int | None) -> dict[int, list[str]]:
    """Return independently seeded but reproducible per-env drop orders."""
    base_seed = 0 if seed is None else int(seed)
    orders = {}
    for eid in env_ids:
        order = list(object_names)
        random.Random(base_seed + 1_000_003 * int(eid)).shuffle(order)
        orders[int(eid)] = order
    return orders


def _rows_for(all_ids: torch.Tensor, selected_ids: torch.Tensor) -> torch.Tensor:
    """Rows of ``selected_ids`` within the (small) event-local ``all_ids`` tensor."""
    rows = []
    for eid in selected_ids.tolist():
        match = (all_ids == eid).nonzero(as_tuple=False)
        if match.numel() != 1:
            raise RuntimeError(f"Sequential-drop event id {eid} was not found exactly once.")
        rows.append(int(match.item()))
    return torch.as_tensor(rows, dtype=torch.long, device=all_ids.device)


def _pin_unreleased(
    assets: Mapping,
    holding_poses: Mapping[str, torch.Tensor],
    unreleased: Mapping[str, set[int]],
    all_ids: torch.Tensor,
    zero_velocity: torch.Tensor,
) -> None:
    """Restore holding pose/zero velocity immediately before each physics tick."""
    for name, held_ids in unreleased.items():
        if not held_ids:
            continue
        selected = torch.as_tensor(sorted(held_ids), dtype=torch.long, device=all_ids.device)
        rows = _rows_for(all_ids, selected)
        assets[name].write_root_pose_to_sim(holding_poses[name][rows], env_ids=selected)
        assets[name].write_root_velocity_to_sim(zero_velocity[rows], env_ids=selected)


def _hold_robot(env, env_ids: torch.Tensor) -> None:
    """Keep robots at their current joints without invoking the action manager."""
    for articulation in env.scene.articulations.values():
        joint_pos = articulation.data.joint_pos[env_ids].clone()
        articulation.set_joint_position_target(joint_pos, env_ids=env_ids)
        articulation.set_joint_velocity_target(torch.zeros_like(joint_pos), env_ids=env_ids)


def _advance_until_stable(
    *,
    advance_one,
    assets: Mapping,
    released: Mapping[int, set[str]],
    env_ids: torch.Tensor,
    max_steps: int,
    stable_frames: int,
    linear_threshold: float,
    angular_threshold: float,
) -> tuple[bool, int]:
    """Drive low-level physics until all released bodies are quiet for N ticks."""
    consecutive = 0
    for step in range(1, max_steps + 1):
        advance_one()
        if _released_objects_are_stable(
            assets, released, env_ids, linear_threshold, angular_threshold
        ):
            consecutive += 1
            if consecutive >= stable_frames:
                return True, step
        else:
            consecutive = 0
    return False, max_steps


def _released_objects_are_stable(
    assets: Mapping,
    released: Mapping[int, set[str]],
    env_ids: torch.Tensor,
    linear_threshold: float,
    angular_threshold: float,
) -> bool:
    for name, asset in assets.items():
        selected_ids = [int(eid) for eid in env_ids.tolist() if name in released[int(eid)]]
        if not selected_ids:
            continue
        selected = torch.as_tensor(selected_ids, dtype=torch.long, device=env_ids.device)
        linear = asset.data.root_lin_vel_w[selected]
        angular = asset.data.root_ang_vel_w[selected]
        if torch.any(torch.linalg.vector_norm(linear, dim=-1) > linear_threshold):
            return False
        if torch.any(torch.linalg.vector_norm(angular, dim=-1) > angular_threshold):
            return False
    return True
