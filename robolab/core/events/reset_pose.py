# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import time

import isaaclab.sim.utils as sim_utils
import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import robolab.constants
import robolab.core.utils.usd_utils as usd_utils
from robolab.core.events.init_pose_layout import (
    Footprint,
    Group,
    GroupMotion,
    Workspace,
    build_groups,
    convex_hull,
    default_yaw_range_deg,
    episode_rng,
    sample_layout,
)
from robolab.core.utils.isaaclab_compat import (
    as_torch,
    quat_isaaclab_to_wxyz,
    quat_wxyz_to_isaaclab,
    write_joint_state,
    write_nodal_state,
    write_root_pose,
    write_root_velocity,
)

########################################################
#  Reset helpers
########################################################

def _parse_asset_cfg(asset_cfg: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str) -> list[str]:
    """Parse asset_cfg into a list of asset names.

    Args:
        asset_cfg: Asset configuration - can be a SceneEntityCfg, string, or list of either.

    Returns:
        List of asset name strings.
    """
    if isinstance(asset_cfg, SceneEntityCfg):
        return [asset_cfg.name]
    elif isinstance(asset_cfg, str):
        return [asset_cfg]
    elif isinstance(asset_cfg, list):
        return [each.name if isinstance(each, SceneEntityCfg) else each for each in asset_cfg if isinstance(each, (SceneEntityCfg, str))]
    else:
        return list(asset_cfg)


def _reset_assets_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_names_set: set[str],
):
    """Internal helper to reset specified assets to their default pose.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        asset_names_set: Set of asset names to reset.
    """
    # Reset rigid bodies that are in the asset list
    for name, rigid_object in env.scene.rigid_objects.items():
        if name not in asset_names_set:
            continue
        # obtain default and deal with the offset for env origins
        default_root_state = as_torch(rigid_object.data.default_root_state)[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        write_root_pose(rigid_object, default_root_state[:, :7], env_ids)
        write_root_velocity(rigid_object, default_root_state[:, 7:], env_ids)

    # Reset articulations that are in the asset list
    for name, articulation_asset in env.scene.articulations.items():
        if name not in asset_names_set:
            continue
        # obtain default and deal with the offset for env origins
        default_root_state = as_torch(articulation_asset.data.default_root_state)[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        write_root_pose(articulation_asset, default_root_state[:, :7], env_ids)
        write_root_velocity(articulation_asset, default_root_state[:, 7:], env_ids)
        # obtain default joint positions
        default_joint_pos = as_torch(articulation_asset.data.default_joint_pos)[env_ids].clone()
        default_joint_vel = as_torch(articulation_asset.data.default_joint_vel)[env_ids].clone()
        # set into the physics simulation
        write_joint_state(articulation_asset, default_joint_pos, default_joint_vel, env_ids)

    # Reset deformable objects that are in the asset list
    for name, deformable_object in env.scene.deformable_objects.items():
        if name not in asset_names_set:
            continue
        # obtain default and set into the physics simulation
        nodal_state = as_torch(deformable_object.data.default_nodal_state_w)[env_ids].clone()
        write_nodal_state(deformable_object, nodal_state, env_ids)


def reset_pose_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str,
):
    """Reset the specified assets' pose to the default state specified in the scene configuration.

    Only resets assets that are in the asset_cfg list.
    Every other asset is left unchanged.
    Adapted from isaaclab.envs.mdp.events.reset_scene_to_default.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        asset_cfg: The asset(s) to reset to default pose. Can be a single asset or list of assets.
    """
    asset_names_set = set(_parse_asset_cfg(asset_cfg))
    _reset_assets_to_default(env, env_ids, asset_names_set)


########################################################
#  Initial-pose randomization
########################################################

INIT_POSE_EVENT_NAME = "randomize_init_pose"
TABLE_OBJECT_NAME = "table"
ROBOT_ENTITY_NAME = "robot"
# Moved groups start this much higher than their authored height and drop into place.
# Authored scenes often sink objects a few millimeters into the table, which is stable where
# they were settled but gets them ejected elsewhere on a not-quite-flat table collider.
SPAWN_LIFT_M = 0.005
# Scene objects whose top is lower than the table's top minus this are below the table: never
# moved, and only obstacles if simulated. Table boxes can overstate the surface height by a few
# centimeters, so this is deliberately loose; it only has to exclude the ground plane and
# objects that fell through the floor.
OBSTACLE_BELOW_TABLE_TOP_M = 0.1


def make_init_pose_event(
    noise_m: float,
    yaw_range_deg: float | None = None,
    objects: list[str] | None = None,
    workspace: tuple[float, float, float, float] | None = None,
) -> dict[str, EventTerm]:
    """Build the reset event that randomizes initial object poses; pass it to ``create_env(events=...)``.

    Args:
        noise_m: Each group moves up to this far (meters) in x and y from its default pose.
        yaw_range_deg: Yaw half-range in degrees, 0-180. None couples it to the noise
            (see :func:`default_yaw_range_deg`).
        objects: Names of the objects that may move. None means every simulated rigid object.
        workspace: ``(x_min, x_max, y_min, y_max)`` in env-local meters. Every moved object
            stays inside it. None uses the table top.
    """
    if noise_m < 0:
        raise ValueError(f"noise_m must be >= 0, got {noise_m}")
    if yaw_range_deg is not None and not 0.0 <= yaw_range_deg <= 180.0:
        raise ValueError(f"yaw_range_deg must be in [0, 180], got {yaw_range_deg}")
    if workspace is not None:
        Workspace(*workspace)  # validate early
    return {
        INIT_POSE_EVENT_NAME: EventTerm(
            func=RandomizeInitPose,
            mode="reset",
            params={"noise_m": noise_m, "yaw_range_deg": yaw_range_deg, "objects": objects, "workspace": workspace},
        )
    }


def _rotate_wxyz(quat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate (n, 3) points by one w-x-y-z quaternion."""
    w, xyz = quat[0], quat[1:]
    t = 2.0 * np.cross(xyz, points)
    return points + w * t + np.cross(xyz, t)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (..., 4) w-x-y-z quaternions."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


class RandomizeInitPose(ManagerTermBase):
    """Reset event that moves groups of objects to random, collision-free poses on the table.

    On the first reset it reads every scene object's default outline and height range from
    the stage, adds the robot's links as obstacles, and builds the groups (see :func:`build_groups`;
    the task's ``object_groups`` field adds links, and its ``fixed_objects`` field keeps objects in
    place). On every reset, each env samples its own layout from a generator seeded by
    ``(env.cfg.seed, task name, episode index)``, where the episode index is
    ``env.run_index * num_envs + env_id``, so a given episode gets the same layout
    regardless of how episodes are split across envs and runs. Positions are
    env-local meters. Each moved group gets a new x, y and yaw, and starts SPAWN_LIFT_M above
    its authored height.

    Runs after the default ``reset`` event, which has already put every object at its
    default pose; fixed groups are left there.
    """

    def __init__(self, cfg: EventTerm, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.footprints: list[Footprint] | None = None
        self.groups: list[Group] | None = None
        self.workspace: Workspace | None = None
        self.table_top_z: float | None = None
        self.setup_seconds: float | None = None
        self.reset_seconds: list[float] = []
        self.last_motions: dict[int, list[GroupMotion | None]] = {}
        # Fraction of the noise each env's last layout used (< 1 when the table was too crowded).
        self.last_noise_scales: dict[int, float] = {}
        # Env-local root positions (meters) written at the last reset: name -> (len(env_ids), 3).
        self.last_written_positions: dict[str, np.ndarray] = {}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        noise_m: float,
        yaw_range_deg: float | None = None,
        objects: list[str] | None = None,
        workspace: tuple[float, float, float, float] | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        if self.groups is None:
            start = time.perf_counter()
            self._build(env, objects, workspace)
            self.setup_seconds = time.perf_counter() - start
            self._report(env, noise_m, yaw_range_deg)

        start = time.perf_counter()
        yaw_range_rad = math.radians(default_yaw_range_deg(noise_m) if yaw_range_deg is None else yaw_range_deg)
        task_name = self._task_name(env)
        run_index = int(getattr(env, "run_index", 0))
        seed = int(env.cfg.seed or 0)
        env_id_list = env_ids.tolist()
        motions_per_env = []
        for env_id in env_id_list:
            rng = episode_rng(seed, task_name, run_index * env.num_envs + env_id)
            layout = sample_layout(self.groups, self.footprints, self.workspace, noise_m, yaw_range_rad, rng)
            motions_per_env.append(layout.motions)
            self.last_motions[env_id] = layout.motions
            self.last_noise_scales[env_id] = layout.noise_scale
        self._write_poses(env, env_ids, motions_per_env)
        self.reset_seconds.append(time.perf_counter() - start)

    @staticmethod
    def _task_name(env: ManagerBasedEnv) -> str:
        return getattr(type(env.cfg), "_task_name", None) or type(env.cfg).__name__

    def _build(self, env: ManagerBasedEnv, objects: list[str] | None, workspace: tuple | None) -> None:
        """Read default outlines from env 0's stage and build the groups."""
        from pxr import Gf, Usd, UsdGeom

        rigid_objects = env.scene.rigid_objects
        if objects is not None:
            unknown = sorted(set(objects) - set(rigid_objects))
            if unknown:
                raise ValueError(f"--init-pose-objects names {unknown}, which are not simulated rigid objects "
                                 f"of this task; available: {sorted(rigid_objects)}")

        scene_cfg = env.scene.cfg.scene
        scene_prims = [p for p in sim_utils.find_matching_prims(scene_cfg.prim_path) if "env_0/" in str(p.GetPath()) + "/"]
        if not scene_prims:
            raise ValueError(f"Scene prim '{scene_cfg.prim_path}' not found for env 0")
        scene_root = scene_prims[0]
        stage = scene_root.GetStage()
        origin = env.scene.env_origins[0].cpu().numpy()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        def box_corners(prim) -> np.ndarray | None:
            """A prim's oriented bounding box at its default pose: 8 env-local corners, (8, 3)."""
            bound = bbox_cache.ComputeWorldBound(prim)
            box = bound.GetRange()
            if box.IsEmpty():
                return None
            matrix = bound.GetMatrix()
            low, high = box.GetMin(), box.GetMax()
            corners = [matrix.Transform(Gf.Vec3d(x, y, z))
                       for x, y, z in itertools.product((low[0], high[0]), (low[1], high[1]), (low[2], high[2]))]
            return np.array(corners) - origin

        def default_box(name: str) -> np.ndarray | None:
            prim = stage.GetPrimAtPath(scene_root.GetPath().AppendChild(name))
            if not prim.IsValid() or not prim.IsActive():
                return None
            return box_corners(prim)

        def outline(name: str) -> np.ndarray:
            """Convex outline from above of every mesh under the object (its box for other shapes)."""
            prim = stage.GetPrimAtPath(scene_root.GetPath().AppendChild(name))
            time_code = Usd.TimeCode.Default()
            points = []
            for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
                if child.IsA(UsdGeom.Mesh):
                    vertices = UsdGeom.Mesh(child).GetPointsAttr().Get(time_code)
                    if vertices is None or len(vertices) == 0:
                        continue
                    matrix = np.array(UsdGeom.Xformable(child).ComputeLocalToWorldTransform(time_code))
                    world = np.asarray(vertices, dtype=float) @ matrix[:3, :3] + matrix[3, :3]
                    points.append(convex_hull(world[:, :2] - origin[:2]))
                elif child.IsA(UsdGeom.Gprim):  # cubes, spheres, cylinders, ...
                    corners = box_corners(child)
                    if corners is not None:
                        points.append(corners[:, :2])
            if not points:
                points.append(default_box(name)[:, :2])
            return convex_hull(np.concatenate(points))

        table_corners = default_box(TABLE_OBJECT_NAME)
        if table_corners is None:
            raise ValueError(f"Initial-pose randomization needs a '{TABLE_OBJECT_NAME}' object in the scene "
                             f"({scene_cfg.spawn.usd_path}) to find the support surface")
        table_low, table_high = table_corners.min(axis=0), table_corners.max(axis=0)
        self.table_top_z = float(table_high[2])
        if workspace is None:
            self.workspace = Workspace(float(table_low[0]), float(table_high[0]), float(table_low[1]), float(table_high[1]))
        else:
            self.workspace = Workspace(*workspace)

        scene_objects = usd_utils.get_usd_objects_info(scene_cfg.spawn.usd_path)
        task_fixed = set(getattr(env.cfg, "fixed_objects", None) or [])
        unknown = sorted(task_fixed - {info["name"] for info in scene_objects})
        if unknown:
            raise ValueError(f"fixed_objects names unknown scene objects {unknown}; scene objects: "
                             f"{sorted(info['name'] for info in scene_objects)}")
        footprints = []
        for info in scene_objects:
            name = info["name"]
            corners = default_box(name) if name != TABLE_OBJECT_NAME else None
            if corners is None:
                continue
            low, high = corners.min(axis=0), corners.max(axis=0)
            if name in rigid_objects and high[2] < self.table_top_z - OBSTACLE_BELOW_TABLE_TOP_M:
                # E.g. an object that fell through the floor before the scene was saved. Its height
                # range keeps it from blocking anything on the table.
                reason = f"below the table (z {low[2]:.2f} to {high[2]:.2f} m)"
            elif name in rigid_objects and name in task_fixed:
                reason = "fixed by the task"
            elif name in rigid_objects:
                reason = "" if objects is None or name in objects else "not selected"
            else:
                # Static, kinematic, or simulated-but-untracked objects over the table are obstacles.
                above_table = high[2] > self.table_top_z - OBSTACLE_BELOW_TABLE_TOP_M
                over_table = np.all(low[:2] < table_high[:2]) and np.all(high[:2] > table_low[:2])
                if not (above_table and over_table):
                    continue
                if info["static_body"]:
                    reason = "static"
                elif info["kinematic"]:
                    reason = "kinematic"
                else:
                    reason = "not tracked by the task"
            footprints.append(Footprint(name=name, polygon_xy=outline(name), fixed_reason=reason,
                                        z_range=(float(low[2]), float(high[2]))))
        footprints.extend(self._robot_footprints(env, table_low, table_high))
        self.footprints = footprints
        self.groups = build_groups(footprints, self.workspace,
                                   getattr(env.cfg, "object_groups", None))
        if objects is not None:
            blocked = [(name, g.fixed_reason) for g in self.groups if not g.moves for name in g.members
                       if name in objects]
            if blocked:
                raise ValueError(f"--init-pose-objects asked to move objects that must stay fixed: {blocked}")

    @staticmethod
    def _robot_footprints(env: ManagerBasedEnv, table_low: np.ndarray, table_high: np.ndarray) -> list[Footprint]:
        """Fixed footprints, with height ranges, for every robot link over the table.

        Links are posed with the simulation's body poses at the first reset (the default
        joint configuration); their shapes come from env 0's stage. Their height ranges let
        short objects sit under a hovering gripper while keeping tall ones away from it.
        """
        robot = env.scene.articulations.get(ROBOT_ENTITY_NAME)
        if robot is None:
            return []
        from pxr import Usd

        robot_prims = [p for p in sim_utils.find_matching_prims(robot.cfg.prim_path)
                       if "env_0/" in str(p.GetPath()) + "/"]
        if not robot_prims:
            return []
        link_prims = {}
        for prim in Usd.PrimRange(robot_prims[0]):
            link_prims.setdefault(prim.GetName(), prim)
        origin = env.scene.env_origins[0].cpu().numpy()
        data = robot.data
        pos_w = data.body_link_pos_w if hasattr(data, "body_link_pos_w") else data.body_pos_w
        quat_w = data.body_link_quat_w if hasattr(data, "body_link_quat_w") else data.body_quat_w
        body_pos = as_torch(pos_w)[0].cpu().numpy() - origin
        body_quat = quat_isaaclab_to_wxyz(as_torch(quat_w)[0].cpu().numpy())
        footprints = []
        for index, body_name in enumerate(robot.body_names):
            prim = link_prims.get(body_name)
            if prim is None:
                continue
            try:
                corners, _ = usd_utils.get_bbox(prim)
            except Exception:  # links without geometry have no bound
                continue
            local = np.array([[c[0], c[1], c[2]] for c in corners], dtype=float)
            if not np.all(np.isfinite(local)) or np.ptp(local, axis=0).max() <= 0.0:
                continue
            world = _rotate_wxyz(body_quat[index], local) + body_pos[index]
            low, high = world.min(axis=0), world.max(axis=0)
            over_table = np.all(low[:2] < table_high[:2]) and np.all(high[:2] > table_low[:2])
            if not over_table:
                continue
            footprints.append(Footprint(name=f"robot/{body_name}", polygon_xy=convex_hull(world[:, :2]),
                                        fixed_reason="robot", z_range=(float(low[2]), float(high[2]))))
        return footprints

    def report(self, noise_m: float, yaw_range_deg: float | None) -> dict:
        """Groups, settings and setup time as a JSON-serializable dict."""
        return {
            "noise_m": noise_m,
            "yaw_range_deg": default_yaw_range_deg(noise_m) if yaw_range_deg is None else yaw_range_deg,
            "workspace_m": vars(self.workspace),
            "table_top_z_m": self.table_top_z,
            "setup_seconds": self.setup_seconds,
            "groups": [{"members": g.members, "moves": g.moves, "fixed_reason": g.fixed_reason} for g in self.groups],
        }

    def _report(self, env: ManagerBasedEnv, noise_m: float, yaw_range_deg: float | None) -> None:
        """Print the groups once per task and save them next to the task's results."""
        import json
        import os

        report = {"task": self._task_name(env), **self.report(noise_m, yaw_range_deg)}
        print(f"[RoboLab] Initial-pose randomization for {report['task']}: noise {noise_m} m, "
              f"yaw ±{report['yaw_range_deg']:.0f} deg, workspace {report['workspace_m']}, "
              f"setup {self.setup_seconds * 1e3:.0f} ms")
        for group in report["groups"]:
            status = "moves" if group["moves"] else f"fixed ({group['fixed_reason']})"
            print(f"[RoboLab]   {' + '.join(group['members'])}: {status}")
        output_dir = robolab.constants.get_output_dir()
        if output_dir and os.path.isdir(output_dir):
            with open(os.path.join(output_dir, "init_pose_groups.json"), "w") as f:
                json.dump(report, f, indent=2)

    def _write_poses(self, env: ManagerBasedEnv, env_ids: torch.Tensor, motions_per_env: list[list]) -> None:
        """Apply each env's group motions to the member objects' default root poses."""
        origins = env.scene.env_origins[env_ids].cpu().numpy()
        for group_index, group in enumerate(self.groups):
            if not group.moves:
                continue
            centers = np.stack([motions[group_index].center_xy for motions in motions_per_env])
            yaws = np.array([motions[group_index].yaw_rad for motions in motions_per_env])
            cos, sin = np.cos(yaws), np.sin(yaws)
            yaw_quats = np.stack([np.cos(yaws / 2), np.zeros_like(yaws), np.zeros_like(yaws), np.sin(yaws / 2)], -1)
            for name in group.members:
                asset = env.scene.rigid_objects[name]
                default = as_torch(asset.data.default_root_state)[env_ids].cpu().numpy()
                offset = default[:, :2] - group.center_xy
                position = default[:, :3].copy()
                position[:, 0] = cos * offset[:, 0] - sin * offset[:, 1] + centers[:, 0]
                position[:, 1] = sin * offset[:, 0] + cos * offset[:, 1] + centers[:, 1]
                position[:, 2] += SPAWN_LIFT_M
                self.last_written_positions[name] = position.copy()
                position += origins
                quat_wxyz = _quat_mul_wxyz(yaw_quats, quat_isaaclab_to_wxyz(default[:, 3:7]))
                pose = np.concatenate([position, quat_wxyz_to_isaaclab(quat_wxyz)], axis=-1)
                write_root_pose(asset, torch.tensor(pose, dtype=torch.float32, device=env.device), env_ids)
                write_root_velocity(asset, torch.zeros(len(env_ids), 6, device=env.device), env_ids)
