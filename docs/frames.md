# Coordinate Frames

RoboLab distinguishes three frames. Historically all three coincided for the
Franka-family robots, so much of the stack was written as if they were one
frame. This document is the contract that removes that assumption.

| Frame | Definition |
|-------|------------|
| **world** | The simulator's global frame. Multi-env replication places envs at `env_origins` offsets in this frame. |
| **env-local** | World minus the env's `scene.env_origins` translation. Replication is pure translation, so env-local and world share axes and orientations. Task scenes, objects, and the ground plane are authored in this frame; the origin sits at the table-mount plane (the ground is authored *below* it, at the canonical z = −0.697; a few legacy scenes keep −0.65, see `tests/test_scene_ground.py`). |
| **robot-root** | The robot articulation's root-link pose. For Franka-family robots the root is at the env origin with identity rotation, so robot-root and env-local coincide **numerically** — this is a coincidence of those embodiments, not a rule. Floor-standing robots (e.g. Galbot One Golf) author their root elsewhere. |

**Conventions:** translations in meters; quaternions `(w, x, y, z)`; env-local
= world with `env_origins` subtracted from positions only (orientations are
unaffected by the replication offset). One exception to the quaternion
ordering: recorded camera extrinsics use IsaacLab's `quat_w_ros`, which is ROS
order `(x, y, z, w)` — see the table below.

## Robot placement

Where a robot's root goes is part of the robot declaration, not an assumption:

- Default: root at the env origin (Franka-family).
- Floor-standing robots author a fixed root z in the robot cfg against the
  canonical scene ground (z = -0.697, see docs/robots.md), e.g.
  `GALBOT_GOLF_TABLETOP_ROOT_POS`.

Whatever the placement, the resulting root pose is recorded per step (see
`robot_root_pose` below). Consumers must read it rather than assume the root is
at the origin.

## Channel frame table

Robot-centric channels are expressed in the **robot-root** frame; scene-centric
channels are **env-local**; the recorded root pose is the bridge between them.

| Channel | Location | Position frame | Orientation frame |
|---------|----------|----------------|-------------------|
| `proprio_obs/ee_pos`, `ee_quat` | observations | robot-root | robot-root |
| `proprio_obs/eef_pos`, `eef_quat` | observations | robot-root | robot-root |
| EE-pose channels (`ee_pose/*`; per-arm e.g. `left_ee_pose/*`, `right_ee_pose/*`) | HDF5 per step | robot-root | robot-root (velocities: world axes) |
| `robot_root_pose/*` | HDF5 per step | env-local | world/env axes |
| `states/**` (incl. `articulation/robot/root_pose`) | HDF5 per step | env-local | world/env axes |
| `initial_state/**` | HDF5 per episode | env-local | world/env axes |
| camera extrinsics | HDF5 per episode | env-local | world/env axes, **ROS order (x, y, z, w)** |
| `WorldState` / object poses / predicates | runtime | env-local | world/env axes |

## Action frames

Policy actions with Cartesian content are interpreted **in the robot-root
frame — always, for every robot**. This is not configurable: IsaacLab's
differential-IK solves in the articulation's base frame and never references
the env origin, and RoboLab relies on that. It also means floor-standing
robots need no action-side special-casing — the solver never assumed the root
was at the origin in the first place.

Joint-space and gripper actions have no Cartesian frame; the rule does not
apply to them.

If you add a Cartesian action term of a new kind (not IsaacLab differential
IK), it must interpret its targets in the robot-root frame too — convert
before the action term if a policy speaks another frame. State the frame in
the action config's docstring, as the existing IK action groups do.

(A note for anyone tempted to make this machine-readable: a frame field
cannot go on the action *group* config — IsaacLab's `ActionManager` iterates
every attribute of the group and raises on anything that is not an action
term. Put it on the term config, and only together with code that actually
reads it.)

## Compatibility with pre-contract recordings

Recordings made before this contract carry no frame metadata. They were all
produced with Franka-family robots (root at env origin, identity rotation), so
their env-local values are byte-identical to robot-root values — every old file
remains valid under the new interpretation. The single compatibility rule,
implemented in one place (`robolab/core/logging/frame_compat.py`): **a demo
without a `robot_root_pose` group is Franka-era; its root pose is identity.**
Do not re-implement this assumption anywhere else.
