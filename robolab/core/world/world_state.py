# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
world_state: "What is the current state?"

Entity access: get_body(), get_articulation(), objects, articulations
Pose/transform: get_pose(), get_velocity(), get_frame_pose()
Geometry: get_dimensions(), get_aabb(), get_bbox(), get_centroid()
Robot state: get_joint_positions(), get_joint_velocity()
Contact (physics-based): in_contact(), get_contact_force(), is_supported_on_surface()

All getters support an optional env_id parameter:
  env_id=None (default) → return batched results for ALL envs, shape (num_envs, ...)
  env_id=<int>          → return results for a single env (backward compat)
"""

import math
from typing import Any

import isaaclab.sim.utils as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, AssetBase, DeformableObject, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils.math import transform_points
from isaacsim.core.prims import XFormPrim
from pxr import Gf, Usd, UsdGeom

import robolab.constants
import robolab.core.utils.usd_utils as usd_utils
from robolab.core.sensors.contact_sensor_utils import (
    get_batch_contact_sensor,
    get_contact_sensor,
    get_contact_sensor_with_order,
)
from robolab.core.utils import vis_utils
from robolab.core.utils.debug_utils import get_caller_info

# Global factory instance for easy access
_global_world = None

def get_world(env: ManagerBasedRLEnv=None):
    """
    Get or create the global WorldState singleton.

    Args:
        env: The ManagerBasedRLEnv instance (or an existing WorldState to pass through)

    Returns:
        The global WorldState instance
    """
    global _global_world

    if isinstance(env, WorldState):
        return env

    if _global_world is None or _global_world.env != env:
        _global_world = WorldState(env)
    return _global_world


def clear_world_cache():
    """Clear the global WorldState singleton (e.g., on env teardown)."""
    global _global_world
    _global_world = None


class WorldState:
    env: ManagerBasedRLEnv = None

    def __init__(
        self,
        env: ManagerBasedRLEnv,
    ):
        self._local_geometry_cache: dict[str, dict] = {}
        self.set_world(env)

    def set_world(self, env: ManagerBasedRLEnv):
        if env != self.env:
            self.env = env
            self._local_geometry_cache.clear()
            if robolab.constants.DEBUG:
                caller_info = get_caller_info()
                print(f"[{caller_info}] setting world for env. Objects: {list(self.entities.keys())}")
                if robolab.constants.VERBOSE:
                    for each in self.entities.keys():
                        print(f"\t{each}, {type(self.entities[each])}")

    @property
    def time(self) -> float:
        return self.env.episode_length_buf * self.env.cfg.sim.dt

    @property
    def entities(self) -> dict[str, Any]:
        """Return a dictionary containing all entities in the scene (extras, objects, articulations, deformable objects)"""
        entities = {}
        entities.update(self.objects)
        entities.update(self.articulations)
        entities.update(self.deformable_objects)
        entities.update(self.extras)
        return entities

    @property
    def extras(self) -> dict[str, XFormPrim]:
        return self.env.scene.extras

    @property
    def objects(self) -> dict[str, RigidObject]:
        """Get list of movable object names"""
        return self.env.scene.rigid_objects

    @property
    def articulations(self) -> dict[str, Articulation]:
        """Get list of movable object names"""
        return self.env.scene.articulations

    @property
    def deformable_objects(self) -> dict[str, DeformableObject]:
        """Get list of movable object names"""
        return self.env.scene.deformable_objects

    @property
    def frames(self) -> dict[str, FrameTransformer]:
        """Get list of frame names"""
        return self.env.scene.frames

    def get_body(self, body: str):
        """ Return object from the scene"""
        if body in self.objects.keys():
            return self.objects.get(body)
        elif body in self.extras.keys():
            return self.extras.get(body)
        elif body in self.articulations.keys():
            return self.articulations.get(body)
        elif body in self.deformable_objects.keys():
            return self.deformable_objects.get(body)
        else:
            raise ValueError(f"[WorldState] Object '{body}' not found in scene; available objects: {list(self.objects.keys()) + list(self.extras.keys()) + list(self.articulations.keys()) + list(self.deformable_objects.keys())}")

    #########################################################
    # Local geometry cache (static per rigid body)
    #########################################################

    def _get_local_geometry(self, body_name: str) -> dict:
        """Cache local AABB corners, centroid offset, dimensions for a body.

        These are STATIC for rigid objects — identical across envs and timesteps.
        Computed once from env_id=0's USD prim and reused for all envs.
        """
        if body_name in self._local_geometry_cache:
            return self._local_geometry_cache[body_name]

        prim = self._get_prim(body_name, env_id=0)

        # Get local corners and centroid at origin (no world pose applied)
        local_corners_gf, local_centroid_gf = usd_utils.get_bbox(prim)

        # Convert to torch tensors on device
        local_corners = torch.tensor(
            [[c[0], c[1], c[2]] for c in local_corners_gf],
            dtype=torch.float32, device=self.env.device
        )  # (8, 3)
        local_centroid = torch.tensor(
            [local_centroid_gf[0], local_centroid_gf[1], local_centroid_gf[2]],
            dtype=torch.float32, device=self.env.device
        )  # (3,)

        # get_bbox now uses ComputeWorldBound + world_xform.GetInverse(), so
        # corners and centroid are already in the prim's own local frame.
        # No env_origin correction needed.

        # Get local AABB and dimensions
        lower, upper = usd_utils.get_aabb(prim)
        scale = usd_utils.get_scale(prim)
        scale_np = np.array([scale[0], scale[1], scale[2]])
        dimensions = (upper - lower) * scale_np

        self._local_geometry_cache[body_name] = {
            'corners': local_corners,         # (8, 3) torch on device
            'centroid': local_centroid,        # (3,) torch on device
            'aabb_lower': lower * scale_np,   # (3,) np
            'aabb_upper': upper * scale_np,   # (3,) np
            'dimensions': dimensions,         # (3,) np
        }
        return self._local_geometry_cache[body_name]

    #########################################################
    # Robot
    #########################################################
    def get_articulation(self, articulation_name: str) -> Articulation:
        """ Return articulation from the scene"""
        if articulation_name in self.articulations.keys():
            return self.articulations[articulation_name]
        else:
            raise ValueError(f"[WorldState] articulation '{articulation_name}' not found in scene; available articulations: {list(self.articulations.keys())}")

    def get_articulation_link_names(self, articulation_name: str) -> list[str]:
        """ Return link names from the articulation"""
        articulation = self.get_articulation(articulation_name)
        return articulation.body_names

    def get_articulation_link_index(self, articulation_name: str, link_name: str) -> int:
        """ Return link index from the articulation"""
        return self.get_articulation_link_names(articulation_name).index(link_name)

    def get_articulation_link_pose(self, articulation_name: str, link_name: str, env_id: int | None = None) -> torch.Tensor:
        """Return link pose from the articulation.

        Args:
            env_id: None → (num_envs, 7), int → (7,)
        """
        articulation = self.get_articulation(articulation_name)
        link_data = articulation.data.body_link_state_w  # (num_envs, num_bodies, 13)
        link_idx = self.get_articulation_link_index(articulation_name, link_name)
        if env_id is None:
            return link_data[:, link_idx, :7].clone().detach()  # (num_envs, 7)
        else:
            return link_data[env_id, link_idx, :7].clone().detach()  # (7,)

    def get_joint_names(self, body_name: str) -> list[str]:
        """Get joint names for articulated body"""
        body = self.get_body(body_name)
        if not isinstance(body, Articulation):
            raise ValueError(f"Object {body_name} is not an articulation")
        return body.data.joint_names

    def get_joint_positions(self, body_name: str, env_id: int | None = None) -> torch.Tensor:
        """Get current joint positions.

        Args:
            env_id: None → (num_envs, num_joints), int → (num_joints,)
        """
        body = self.get_body(body_name)
        if not isinstance(body, Articulation):
            raise ValueError(f"Object {body_name} is not an articulation")
        if env_id is None:
            return body.data.joint_pos.clone().detach()
        return body.data.joint_pos[env_id].clone().detach()

    def get_joint_velocity(self, body_name: str, env_id: int | None = None) -> torch.Tensor:
        """Get current joint velocities.

        Args:
            env_id: None → (num_envs, num_joints), int → (num_joints,)
        """
        body = self.get_body(body_name)
        if not isinstance(body, Articulation):
            raise ValueError(f"Object {body_name} is not an articulation")
        if env_id is None:
            return body.data.joint_vel.clone().detach()
        return body.data.joint_vel[env_id].clone().detach()

    #########################################################
    # Frames
    #########################################################
    def get_frames(self, frame_cfg_name: str="frames") -> FrameTransformer:
        """Get frame from the scene"""
        return self.env.scene[frame_cfg_name]

    def get_frame_pose(self, frame: str, frame_cfg_name: str="frames", as_matrix: bool=False, env_id: int | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Get pose for a frame.

        Args:
            env_id: None → batched (num_envs, ...), int → single env
        """
        frames = self.get_frames(frame_cfg_name)
        frame_names = frames.data.target_frame_names
        frame_idx = frame_names.index(frame)

        frame_pos_w = frames.data.target_pos_w[:, frame_idx, :].clone().detach()
        frame_quat_w = frames.data.target_quat_w[:, frame_idx, :].clone().detach()

        if env_id is not None:
            frame_pos_w = frame_pos_w[env_id]
            frame_quat_w = frame_quat_w[env_id]

        if as_matrix:
            from robolab.core.utils.geometry_utils import pose_from_pos_quat
            frame_pose_w = pose_from_pos_quat(frame_pos_w, frame_quat_w)
            return frame_pose_w
        else:
            return frame_pos_w, frame_quat_w


    def get_frame_relative_transform(self, frame: str, frame_cfg_name: str="frames", as_matrix: bool=False, env_id: int | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Get transform for a frame.

        Args:
            env_id: None → batched (num_envs, ...), int → single env
        """
        frames = self.get_frames(frame_cfg_name)
        frame_names = frames.data.target_frame_names
        frame_idx = frame_names.index(frame)

        frame_pos_tf = frames.data.target_pos_source[:, frame_idx, :].clone().detach()
        frame_quat_tf = frames.data.target_quat_source[:, frame_idx, :].clone().detach()

        if env_id is not None:
            frame_pos_tf = frame_pos_tf[env_id]
            frame_quat_tf = frame_quat_tf[env_id]

        if as_matrix:
            from robolab.core.utils.geometry_utils import pose_from_pos_quat
            frame_pose_tf = pose_from_pos_quat(frame_pos_tf, frame_quat_tf)
            return frame_pose_tf
        else:
            return frame_pos_tf, frame_quat_tf

    #########################################################
    # Bodies
    #########################################################
    def _get_prim(self, body_name: str, env_id: int = 0) -> Usd.Prim:
        """Get USD prim for a body in a specific env. Used internally for init-time
        geometry caching and visualization. Not called per-step."""
        body = self.get_body(body_name)
        if isinstance(body, XFormPrim):
            idx = min(env_id, len(body.prims) - 1)
            return body.prims[idx]
        prim_path = body.cfg.prim_path
        env_id_str = "env_" + str(env_id)
        prims = sim_utils.find_matching_prims(prim_path)
        for prim in prims:
            if env_id_str in str(prim.GetPath()):
                return prim
        raise ValueError(f"[WorldState] Prim at path '{prim_path}' not found in scene")

    def get_pose(self, body_name: str, is_relative: bool = True, as_matrix: bool = False, env_id: int | None = None) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Get pose in xyz, wxyz format.

        Args:
            env_id: None → (num_envs, 3) and (num_envs, 4), int → (3,) and (4,)
                    If as_matrix: None → (num_envs, 4, 4), int → (4, 4)
        """
        body = self.get_body(body_name)
        if isinstance(body, AssetBase):
            if env_id is not None:
                pos = body.data.root_pos_w[env_id].clone().detach()
                quat = body.data.root_quat_w[env_id].clone().detach()
                if is_relative:
                    pos = pos - self.env.scene.env_origins[env_id]
            else:
                pos = body.data.root_pos_w.clone().detach()  # (N, 3)
                quat = body.data.root_quat_w.clone().detach()  # (N, 4)
                if is_relative:
                    pos = pos - self.env.scene.env_origins  # (N, 3)
        elif isinstance(body, XFormPrim):
            num_prims = len(body._prim_paths) if hasattr(body, '_prim_paths') else body.count
            if env_id is not None:
                # Clamp index — static extras may have fewer prims than envs
                idx = min(env_id, num_prims - 1)
                positions, orientations = body.get_world_poses(indices=[idx])
                pos = positions[0]
                quat = orientations[0]
                if not isinstance(pos, torch.Tensor):
                    pos = torch.tensor(pos, dtype=torch.float32, device=self.env.device)
                    quat = torch.tensor(quat, dtype=torch.float32, device=self.env.device)
                else:
                    pos = pos.clone().detach()
                    quat = quat.clone().detach()
                if is_relative:
                    pos = pos - self.env.scene.env_origins[env_id]
            else:
                # XFormPrim — get all available poses then map to envs.
                # Static extras may have fewer prims than num_envs (e.g., shelf).
                # In that case, compute relative pose from prim 0 and replicate.
                positions, orientations = body.get_world_poses()
                if num_prims >= self.env.num_envs:
                    # One prim per env — straightforward
                    all_pos, all_quat = [], []
                    for i in range(self.env.num_envs):
                        p = positions[i]
                        q = orientations[i]
                        if not isinstance(p, torch.Tensor):
                            p = torch.tensor(p, dtype=torch.float32, device=self.env.device)
                            q = torch.tensor(q, dtype=torch.float32, device=self.env.device)
                        else:
                            p = p.clone().detach()
                            q = q.clone().detach()
                        if is_relative:
                            p = p - self.env.scene.env_origins[i]
                        all_pos.append(p)
                        all_quat.append(q)
                    pos = torch.stack(all_pos)  # (N, 3)
                    quat = torch.stack(all_quat)  # (N, 4)
                else:
                    # Fewer prims than envs — static object, same relative pose in all envs
                    p0 = positions[0]
                    q0 = orientations[0]
                    if not isinstance(p0, torch.Tensor):
                        p0 = torch.tensor(p0, dtype=torch.float32, device=self.env.device)
                        q0 = torch.tensor(q0, dtype=torch.float32, device=self.env.device)
                    else:
                        p0 = p0.clone().detach()
                        q0 = q0.clone().detach()
                    rel_pos = p0 - self.env.scene.env_origins[0] if is_relative else p0
                    pos = rel_pos.unsqueeze(0).expand(self.env.num_envs, -1).clone()  # (N, 3)
                    quat = q0.unsqueeze(0).expand(self.env.num_envs, -1).clone()  # (N, 4)
        else:
            raise ValueError(f"[WorldState] Object '{body_name}' is not a valid body")

        if as_matrix:
            from robolab.core.utils.geometry_utils import pose_from_pos_quat
            pose_w = pose_from_pos_quat(pos, quat)
            return pose_w
        else:
            return pos, quat

    def get_velocity(self, body_name: str, env_id: int | None = None) -> torch.Tensor:
        """Get velocity for a body. Returns zero velocity for XFormPrim extras.

        Args:
            env_id: None → (num_envs, 6), int → (6,)
        """
        body = self.get_body(body_name)
        if isinstance(body, XFormPrim):
            if env_id is None:
                return torch.zeros(self.env.num_envs, 6, dtype=torch.float32, device=self.env.device)
            return torch.zeros(6, dtype=torch.float32, device=self.env.device)
        if env_id is None:
            return body.data.root_vel_w.clone().detach()
        return body.data.root_vel_w[env_id].clone().detach()

    def get_dimensions(self, body: str) -> np.ndarray:
        """Get dimensions from cached local geometry. Returns (3,) np array."""
        geom = self._get_local_geometry(body)
        return geom['dimensions'].copy()

    def get_aabb(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the Axis-Aligned Bounding Box (AABB) for a body in local coordinates.

        Returns:
            tuple[np.ndarray, np.ndarray]: (lower, upper) each shape (3,)
        """
        geom = self._get_local_geometry(body)
        return geom['aabb_lower'].copy(), geom['aabb_upper'].copy()

    def get_bbox(self, body: str, env_id: int | None = None) -> tuple:
        """
        Get the Oriented Bounding Box (OBB) for a body in world coordinates.

        Uses cached local geometry + vectorized transform_points for efficiency.

        Args:
            body: Name of the body/object in the scene
            env_id: None → (Tensor(N, 8, 3), Tensor(N, 3))
                    int  → (list[Gf.Vec3d] len=8, np.ndarray(3,)) for backward compat

        Returns:
            (corners, centroid) — shapes depend on env_id
        """
        geom = self._get_local_geometry(body)
        pos, quat = self.get_pose(body, env_id=env_id)

        if env_id is not None:
            # Single env — transform and return legacy format
            corners_world = transform_points(
                geom['corners'].unsqueeze(0),   # (1, 8, 3)
                pos=pos.unsqueeze(0),            # (1, 3)
                quat=quat.unsqueeze(0)           # (1, 4)
            ).squeeze(0)  # (8, 3)
            centroid_world = transform_points(
                geom['centroid'].reshape(1, 1, 3),  # (1, 1, 3)
                pos=pos.unsqueeze(0),
                quat=quat.unsqueeze(0)
            ).squeeze(0).squeeze(0)  # (3,)
            # Convert to legacy format
            corners_list = [Gf.Vec3d(*c.cpu().tolist()) for c in corners_world]
            centroid_np = centroid_world.cpu().numpy()
            return corners_list, centroid_np
        else:
            # All envs — fully vectorized
            num_envs = pos.shape[0]
            corners_world = transform_points(
                geom['corners'].unsqueeze(0).expand(num_envs, -1, -1),  # (N, 8, 3)
                pos=pos,    # (N, 3)
                quat=quat   # (N, 4)
            )  # (N, 8, 3)
            centroid_world = transform_points(
                geom['centroid'].reshape(1, 1, 3).expand(num_envs, -1, -1),  # (N, 1, 3)
                pos=pos,
                quat=quat
            ).squeeze(1)  # (N, 3)
            return corners_world, centroid_world

    def get_local_geometric_center(self, body: str) -> np.ndarray:
        """
        Calculate the geometric center of a body's AABB in local coordinates.

        Returns:
            np.ndarray: shape (3,)
        """
        lower, upper = self.get_aabb(body)
        return (lower + upper) / 2

    def get_centroid(self, body: str, env_id: int | None = None):
        """
        Get the geometric center of a body's OBB in world coordinates.

        Args:
            env_id: None → Tensor(num_envs, 3), int → np.ndarray(3,)
        """
        _, centroid = self.get_bbox(body, env_id=env_id)
        return centroid

    #########################################################
    # Contact
    #########################################################
    def in_contact(self, body1: str, body2: str, force_threshold: float = 0.1, env_id: int | None = None):
        """Check if two bodies are in contact.

        Args:
            env_id: None → Tensor(num_envs,) bool, int → bool
        """
        contact_sensor = get_contact_sensor(self.env.scene, body1, body2)
        if env_id is not None:
            force_matrix = contact_sensor.data.force_matrix_w[env_id]
            return torch.any(torch.abs(force_matrix) > force_threshold).item()
        else:
            # force_matrix_w: (num_envs, num_bodies, num_filter_bodies, 3)
            force_matrix = contact_sensor.data.force_matrix_w
            # Reduce over bodies, filter_bodies, and xyz → per-env bool
            above = torch.abs(force_matrix) > force_threshold  # (N, B, M, 3)
            return above.any(dim=-1).any(dim=-1).any(dim=-1)   # (N,)

    def get_objects_in_contact_with(
        self,
        body: str,
        candidates: list[str],
        force_threshold: float = 0.1,
        env_id: int | None = None,
    ) -> list[str]:
        """
        Get all objects from candidates that are currently in contact with body.

        Note: This returns a list of object names and is inherently per-env.
        When env_id=None, defaults to env_id=0 for backward compat.
        """
        if env_id is None:
            env_id = 0

        batch_sensor = get_batch_contact_sensor(self.env.scene, body)

        if batch_sensor is None:
            objects_in_contact = [obj for obj in candidates if self.in_contact(body, obj, force_threshold, env_id=env_id)]
            if robolab.constants.VERBOSE:
                from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
                available_sensors = list(get_contact_sensors(self.env.scene).keys())
                print(f"[WorldState] Batch sensor for '{body}' not found. Available sensors: {available_sensors}. Found '{body}' in contact with: {objects_in_contact}")
            return objects_in_contact

        force_matrix = batch_sensor.data.force_matrix_w[env_id]
        force_above_threshold = torch.abs(force_matrix) > force_threshold
        any_force_per_body = torch.any(force_above_threshold, dim=-1)
        in_contact_mask = torch.any(any_force_per_body, dim=0)

        num_filter_bodies = in_contact_mask.shape[0]
        if len(candidates) != num_filter_bodies:
            if robolab.constants.VERBOSE:
                print(f"[WorldState] Warning: candidates ({len(candidates)}) != sensor filter bodies ({num_filter_bodies}). Falling back to individual queries.")
            objects_in_contact = [obj for obj in candidates if self.in_contact(body, obj, force_threshold, env_id=env_id)]
            if robolab.constants.VERBOSE:
                print(f"[WorldState] '{body}' in contact with: {objects_in_contact}")
            return objects_in_contact

        objects_in_contact = [candidates[i] for i in range(len(candidates)) if in_contact_mask[i].item()]

        if robolab.constants.VERBOSE:
            print(f"[WorldState] '{body}' in contact with: {objects_in_contact}")

        return objects_in_contact

    def get_contact_force(self, body1: str, body2: str, env_id: int | None = None) -> torch.Tensor:
        """
        Get the contact force vector between two bodies.

        Args:
            env_id: None → (num_envs, 3), int → (3,)
        """
        contact_sensor, is_reversed = get_contact_sensor_with_order(self.env.scene, body1, body2)
        if env_id is not None:
            force_matrix = contact_sensor.data.force_matrix_w[env_id]
            net_force = force_matrix.sum(dim=(0, 1))  # (3,)
        else:
            # (num_envs, num_bodies, num_filter_bodies, 3) → (num_envs, 3)
            force_matrix = contact_sensor.data.force_matrix_w
            net_force = force_matrix.sum(dim=(1, 2))  # (N, 3)

        if is_reversed:
            net_force = -net_force

        return net_force

    def is_supported_on_surface(
        self,
        obj: str,
        surface: str,
        cone_half_angle_deg: float = 45.0,
        force_threshold: float = 0.1,
        env_id: int | None = None,
    ):
        """
        Check if an object is stably supported on a surface by analyzing contact forces.

        Args:
            env_id: None → Tensor(num_envs,) bool, int → bool
        """
        contact_force = self.get_contact_force(obj, surface, env_id=env_id)
        cos_theta_max = math.cos(math.radians(cone_half_angle_deg))

        if env_id is not None:
            force_magnitude = torch.norm(contact_force).item()
            if force_magnitude < force_threshold:
                return False
            fz = contact_force[2].item()
            if fz <= 0:
                return False
            if fz < force_magnitude * cos_theta_max:
                return False
            return True
        else:
            # Vectorized: contact_force is (N, 3)
            force_magnitude = torch.norm(contact_force, dim=-1)  # (N,)
            fz = contact_force[:, 2]  # (N,)
            has_contact = force_magnitude >= force_threshold
            force_upward = fz > 0
            in_cone = fz >= force_magnitude * cos_theta_max
            return has_contact & force_upward & in_cone  # (N,) bool tensor

    def get_objects_supported_on(
        self,
        surface: str,
        candidates: list[str],
        cone_half_angle_deg: float = 45.0,
        force_threshold: float = 0.1,
        env_id: int | None = None,
    ) -> list[str]:
        """
        Get all objects from candidates that are stably supported on a surface.

        Note: Returns list of names, inherently per-env. Defaults to env_id=0 when None.
        """
        if env_id is None:
            env_id = 0
        return [
            obj for obj in candidates
            if self.is_supported_on_surface(obj, surface, cone_half_angle_deg, force_threshold, env_id=env_id)
        ]


    #########################################################
    # Visualization
    #########################################################

    def visualize(self, objects: list[str] | None = None, env_id: int = 0):
        """
        Visualizes the bounding box and axes of the objects for a given env.

        Args:
            objects: List of object names, or None for all objects.
            env_id: Which env to visualize (default 0).
        """
        if isinstance(objects, str):
            objects = [objects]
        elif objects is None:
            objects = list(self.objects.keys())

        for each in objects:
            self.visualize_object(each, env_id=env_id)

    def visualize_object(self, object_name: str, env_id: int = 0):
        """Visualizes one object's bounding box and axes."""
        corners, centroid = self.get_bbox(object_name, env_id=env_id)
        vis_utils.visualize_bbox(corners, object_name, color='red')
        pos, quat = self.get_pose(object_name, env_id=env_id)
        vis_utils.visualize_axes(pos.cpu().numpy().tolist(), quat.cpu().numpy().tolist(), object_name)
