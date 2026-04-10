# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import omni.usd
import torch
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
from pxr import Gf, UsdGeom

from robolab.constants import DEVICE

COLORS = {'red': Gf.Vec3f(1, 0, 0),
          'green': Gf.Vec3f(0, 1, 0),
          'blue': Gf.Vec3f(0, 0, 1),
          'yellow': Gf.Vec3f(1, 1, 0), }

def create_visualization_marker(prim_path: str, scale: float = 0.1) -> VisualizationMarkers:
    """Create a visualization marker at the given prim path."""
    marker_cfg = FRAME_MARKER_CFG.replace(prim_path=prim_path)
    marker_cfg.markers["frame"].scale = (scale, scale, scale)
    return VisualizationMarkers(marker_cfg)

def visualize_bbox(corners, object_name: str, color: str='red'):
    """
    Visualizes the bounding box for an object using custom AABB function.
    The corners are assumed to be in the following format:
    corners = [
    Gf.Vec3d(x0, y0, z0),
    Gf.Vec3d(x1, y1, z1),
    ...
    Gf.Vec3d(x7, y7, z7)
    ]

    Example:
        state = WorldState(env, env_id)
        banana_bbox, centroid = state.get_bbox('banana')
        vis_utils.visualize_bbox(banana_bbox, 'banana', color='yellow')

    Args:
        object_name (str): Name of the target object
    """

    # Get current USD stage
    stage = omni.usd.get_context().get_stage()

    # Create parent prim for bounding boxes
    bbox_parent = "/Visuals/BoundingBoxes"
    if not stage.GetPrimAtPath(bbox_parent):
        UsdGeom.Xform.Define(stage, bbox_parent)

    # Create unique prim path for this bbox
    prim_path = f"{bbox_parent}/{object_name}_bbox"

    # Remove existing prim if re-creating
    if stage.GetPrimAtPath(prim_path):
        stage.RemovePrim(prim_path)

    # Create wireframe visualization
    bbox_prim = stage.DefinePrim(prim_path, "BasisCurves")
    curves = UsdGeom.BasisCurves(bbox_prim)


    # # corners: (8, 3) numpy array from your first function
    usd_order = [0, 4, 6, 2, 1, 5, 7, 3]
    gf_corners = [Gf.Vec3f(*corners[i]) for i in usd_order]

    # Define box edges (12 edges)
    points = []
    for edge in [(0,1), (1,2), (2,3), (3,0),
                 (4,5), (5,6), (6,7), (7,4),
                 (0,4), (1,5), (2,6), (3,7)]:
        points.append(gf_corners[edge[0]])
        points.append(gf_corners[edge[1]])

    # Configure curve properties
    curves.CreatePointsAttr().Set(points)
    curves.CreateCurveVertexCountsAttr().Set([2]*12)  # 12 edges
    curves.CreateTypeAttr().Set("linear")
    curves.CreateBasisAttr().Set("bspline")
    curves.CreateDisplayColorAttr().Set([COLORS[color]])  # Red color
    curves.CreateWidthsAttr().Set([0.005]*len(points))  # Line thickness

    # Set as guide geometry
    # bbox_prim.GetAttribute("purpose").Set("guide")

def delete_visual_axes(name: str = None, axes_parent: str = "/Visuals/Axes"):
    """
    Delete visual axes. If name is provided, delete the specific axes for that object.
    If name is None, delete all visual axes in the axes_parent directory.

    Args:
        name: Optional name of the specific object axes to delete. If None, deletes all axes.
        axes_parent: Parent directory containing the visual axes.
    """
    stage = omni.usd.get_context().get_stage()

    if name is not None:
        # Delete specific axes for the given object name
        xform_path = f"{axes_parent}/{name}"
        if stage.GetPrimAtPath(xform_path):
            stage.RemovePrim(xform_path)
    else:
        # Delete all visual axes in the parent directory
        parent_prim = stage.GetPrimAtPath(axes_parent)
        if parent_prim:
            # Get all child prims and delete them
            for child in parent_prim.GetChildren():
                stage.RemovePrim(child.GetPath())

def visualize_axes(pos: torch.Tensor | np.ndarray | list,
                   quat_wxyz: torch.Tensor | np.ndarray | list,
                   object_name: str,
                   axis_length: float = 0.05,
                   axes_parent: str = "/Visuals/Axes"):
    """
    Visualizes the local axes (X, Y, Z) as RGB lines at the given pose.

    Example:
        state = WorldState(env, env_id)
        bowl_pose = state.get_pose('bowl').cpu().numpy().tolist()
        vis_utils.visualize_axes(bowl_pose[:3], bowl_pose[3:],'bowl')

    Args:
        pos: (3,) array-like, world position [x, y, z]
        quat_wxyz: (4,) array-like, quaternion [w, x, y, z]
        object_name: str, unique name for this axes visualization
        axis_length: float, length of each axis line
    """
    # Get USD stage
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(axes_parent):
        UsdGeom.Xform.Define(stage, axes_parent)

    # Define a dedicated Xform for this frame
    xform_path = f"{axes_parent}/{object_name}"
    if not stage.GetPrimAtPath(xform_path):
        UsdGeom.Xform.Define(stage, xform_path)
    else:
        stage.RemovePrim(xform_path)
        UsdGeom.Xform.Define(stage, xform_path)

    axes_prim = stage.DefinePrim(xform_path, "BasisCurves")
    curves = UsdGeom.BasisCurves(axes_prim)

    def quat_wxyz_to_matrix(q):
        if isinstance(q, np.ndarray):
            q = q.tolist()
        if isinstance(q, torch.Tensor):
            q = q.cpu().numpy().tolist()
        w, x, y, z = q
        quat = Gf.Quatf(w, Gf.Vec3f(x, y, z))
        return quat

    # Convert quaternion to rotation matrix
    rot = quat_wxyz_to_matrix(quat_wxyz)
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy().tolist()
    if isinstance(pos, np.ndarray):
        pos = pos.tolist()
    trans = Gf.Vec3f(*pos)
    origin = Gf.Vec3f(0, 0, 0)

    # Canonical axes
    axes = [Gf.Vec3f(1,0,0), Gf.Vec3f(0,1,0), Gf.Vec3f(0,0,1)]
    colors = [Gf.Vec3f(1,0,0), Gf.Vec3f(0,1,0), Gf.Vec3f(0,0,1)]  # RGB

    points = []
    for ax in axes:
        tip = origin + (ax * axis_length)
        points.extend([origin, tip])

    curves.CreatePointsAttr().Set(points)
    curves.CreateCurveVertexCountsAttr().Set([2, 2, 2])  # 3 axes, each a line
    curves.CreateTypeAttr().Set("linear")
    curves.CreateBasisAttr().Set("bspline")
    curves.CreateDisplayColorAttr().Set(colors)  # One color per curve
    curves.GetDisplayColorAttr().SetMetadata("interpolation", UsdGeom.Tokens.uniform)
    curves.CreateWidthsAttr().Set([0.007]*6)

    xform = UsdGeom.Xform(stage.GetPrimAtPath(xform_path))
    xform_ops = xform.GetOrderedXformOps()
    # Clear existing ops if any
    for op in xform_ops:
        xform.RemoveXformOp(op)
    xform.AddTranslateOp().Set(trans)
    xform.AddOrientOp().Set(rot)

    # Optionally, mark as guide geometry
    # axes_prim.GetAttribute("purpose").Set("guide")

def visualize_axes_markers(pos: torch.Tensor | np.ndarray | list,
                        quat: torch.Tensor | np.ndarray | list,
                        visualization_marker: VisualizationMarkers) -> None:
    """Visualize the goal frame markers at pos, quat (xyzw)."""
    if isinstance(pos, torch.Tensor):
        pos = pos.unsqueeze(0)
    if isinstance(quat, torch.Tensor):
        quat = quat.unsqueeze(0)
    if isinstance(pos, np.ndarray):
        pos = torch.tensor(pos, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    if isinstance(quat, np.ndarray):
        quat = torch.tensor(quat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    if isinstance(pos, list):
        pos = torch.tensor(pos, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    if isinstance(quat, list):
        quat = torch.tensor(quat, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    visualization_marker.visualize(translations=pos, orientations=quat)
