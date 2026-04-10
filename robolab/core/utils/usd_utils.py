# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# The import below fails when Isaac Sim is not installed or the search paths have
# not been patched with `AppLauncher`.  Wrap it so that utilities that do *not*
# require the simulator (e.g. reading static information from a USD file) can
# still be used in a plain Python environment.

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

try:
    import isaacsim.core.utils.stage as stage_utils  # type: ignore
except ImportError:  # pragma: no cover -- running outside Isaac-Sim
    stage_utils = None  # noqa: N816 – keep original style for clarity

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def create_str_attribute_to_prim(prim, attribute_name: str):
    """Creates attribute for a prim that holds a string.
    See: https://openusd.org/release/api/class_usd_prim.html
    See: https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd.commands/omni.usd.commands.CreateUsdAttributeCommand.html
    Args:
        prim (Usd.Prim): A Prim for holding the attribute.
        attribute_name (str): The name of the attribute to create.
    Returns:
        Usd.Attribute: An attribute created at specific prim.
    """

    import omni
    from pxr import Sdf, Usd
    omni.kit.commands.execute(
        "CreateUsdAttributeCommand",
        prim=prim,
        attr_name=attribute_name,
        attr_type=Sdf.ValueTypeNames.String,
    )

    attr: Usd.Attribute = prim.GetAttribute(attribute_name)


    return attr

def set_str_attribute_to_prim(prim, name: str, value: str):
    attribute = create_str_attribute_to_prim(prim, attribute_name=name)
    attribute.Set(value)

def get_root_prim_path(root_prim_path=None):
    """
    Gets the root prim path. If root_prim_path is provided, use that; otherwise use the default prim.

    Args:
        root_prim_path (str, optional): Specific prim path to use. Defaults to None.

    Raises:
        ValueError: If neither root_prim_path nor default prim are available.
        ValueError: If the specified root_prim_path is not found.

    Returns:
        tuple: (root_prim, root_prim_path) - The prim object and its path
    """
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    root_prim = None
    if root_prim_path is not None:
        # Use provided root_prim_path
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim:
            raise ValueError(f"Root prim path {root_prim_path} not found.")
        print(f"Using provided prim path: {root_prim_path}")
    else:
        # Fall back to default prim
        root_prim = stage.GetDefaultPrim()
        if not root_prim:
            raise ValueError("No default prim found and no root_prim_path provided.")
        root_prim_path = str(root_prim.GetPath())
        print(f"Using default prim: {root_prim_path}")

    return root_prim, root_prim_path

def rename_prim(stage, old_path_str, new_path_str):
    from pxr import Sdf, Usd
    old_path = Sdf.Path(old_path_str)
    new_path = Sdf.Path(new_path_str)

    stage.GetEditTarget().GetLayer().ExportToString()  # force sync if needed

    # Define the new prim at the new path
    old_prim = stage.GetPrimAtPath(old_path)
    if not old_prim.IsValid():
        raise RuntimeError(f"Prim at {old_path} is not valid")

    # Use Sdf to copy prim spec (correct module and function)
    root_layer = stage.GetRootLayer()
    Sdf.CopySpec(root_layer, old_path, root_layer, new_path)

    # Remove the old prim
    stage.RemovePrim(old_path)

    print(f"Renamed prim from {old_path} to {new_path}")
    return stage.GetPrimAtPath(new_path)

def rename_root_prim(new_path_str):
    """
    Renames the root default prim. This function does not save the stage.

    Args:
        new_path_str (_type_): _description_
    """
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    old_path_str = str(stage.GetDefaultPrim().GetPath())
    rename_prim(stage, old_path_str, new_path_str)
    default_prim = stage.GetPrimAtPath(new_path_str)
    stage.SetDefaultPrim(default_prim)
    stage.GetRootLayer().Save()

def get_attribute_by_name(prim: Usd.Prim, name: str) -> str:
    """Extract the first authored attribute found in the USD file."""
    # Search default prim and its direct children for description in attributes
    search_prims = [prim] + list(prim.GetChildren())
    for prim in search_prims:
        # 1) Attribute named "description"
        for attr in prim.GetAttributes():
            if attr.GetName() == name:
                val = attr.Get()
                if val:
                    return val

        # 2) Prim metadata "documentation"
        doc = prim.GetMetadata("description")
        if doc:
            return doc

        # 3) assetInfo[name]
        try:
            asset_desc = prim.GetAssetInfoByKey(name)
        except Exception:
            asset_desc = None
        if asset_desc:
            return asset_desc

        # 4) customData[name]
        try:
            custom_desc = prim.GetCustomDataByKey(name)
        except Exception:
            custom_desc = None
        if custom_desc:
            return custom_desc
    return ""


def get_usd_rigid_body_info(usd_path: str) -> dict:
    """
    Analyze a USD file containing a single object and extract rigid body information.

    Args:
        usd_path: Path to the USD file

    Returns:
        Dictionary containing rigid body and physics properties
    """
    from robolab.core.utils.physics_utils import get_friction

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise ValueError(f"Failed to open stage: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise ValueError(f"No default prim found in: {usd_path}")

    # Size
    lower, upper = get_aabb(default_prim)
    dims = upper - lower
    size = [float(dims[i]) for i in range(3)]

    # Get description
    # attributes = default_prim.GetAttributes()
    # description = ""
    # if attributes:
    #     for attr in attributes:
    #         # Get the attribute name and its value at the default time
    #         attribute_name = attr.GetName()
    #         attribute_value = attr.Get()
    #         if attribute_name == 'description':
    #             description = attribute_value
    #             break
    description = get_attribute_by_name(default_prim, "description")
    class_name = get_attribute_by_name(default_prim, "class")

    # Check if it's a rigid body
    rigid_body_api = UsdPhysics.RigidBodyAPI(default_prim)
    rigid_body = True if (rigid_body_api and rigid_body_api.GetRigidBodyEnabledAttr().Get()) else str(rigid_body_api)

    # Extract mass and density using MassAPI
    mass = None
    density = None
    mass_api = UsdPhysics.MassAPI(default_prim)
    if mass_api:
        mass_attr = mass_api.GetMassAttr()
        if mass_attr and mass_attr.IsValid():
            mass = mass_attr.Get()

        density_attr = mass_api.GetDensityAttr()
        if density_attr and density_attr.IsValid():
            density = density_attr.Get()

    # Extract friction and restitution using MaterialAPI
    friction_info = get_friction(default_prim, stage)
    dynamic_friction = friction_info['dynamic_friction'] if friction_info else None
    static_friction = friction_info['static_friction'] if friction_info else None
    restitution = friction_info['restitution'] if friction_info else None
    if density is None or density == 0:
        density = friction_info['density'] if friction_info else None

    # Get the world transformation matrix
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_xform = xform_cache.GetLocalToWorldTransform(default_prim)

    # Extract translation (position)
    pos = world_xform.ExtractTranslation()

    # Extract rotation as a quaternion
    rot = world_xform.ExtractRotation().GetQuat()
    rot_quat = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])


    object_info = {
        'name': default_prim.GetName(),
        'usd_path': str(usd_path),
        'prim_path': str(default_prim.GetPath()),
        'position': (pos[0], pos[1], pos[2]),
        'quat_wxyz': rot_quat,
        'payload': get_prim_payload(default_prim, as_string=True),
        'rigid_body': rigid_body,
        'static_body': not rigid_body,
        'class': class_name,
        'description': description,
        'dims': size,
        'mass': mass,
        'density': density,
        'dynamic_friction': dynamic_friction,
        'static_friction': static_friction,
        'restitution': restitution,
    }
    return object_info

@lru_cache(maxsize=256)
def _get_usd_objects_info_cached(usd_path: str) -> tuple:
    """
    Cached internal function to analyze a USD file and extract rigid bodies and static bodies.
    Returns a tuple of tuples (immutable) for caching compatibility.

    Args:
        usd_path: Path to the USD file

    Returns:
        Tuple of object info tuples (frozen for cache compatibility)
    """
    from robolab.core.utils.usd_utils import get_prim_payload

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise ValueError(f"Failed to open stage: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise ValueError(f"No default prim found in: {usd_path}")

    scene_objects = []

    # Get all children of the default prim
    children = default_prim.GetChildren()

    for child in children:
        name = child.GetName()

        attributes = child.GetAttributes()
        description = ""
        if attributes:
            for attr in attributes:
                # Get the attribute name and its value at the default time
                attribute_name = attr.GetName()
                attribute_value = attr.Get()
                if attribute_name == 'description':
                    description = attribute_value
                    break


        # Check if it's a rigid body
        rigid_body_api = UsdPhysics.RigidBodyAPI(child)
        rigid_body = rigid_body_api and rigid_body_api.GetRigidBodyEnabledAttr().Get()

        # Check if kinematic
        is_kinematic = False
        if rigid_body:
            kinematic_attr = child.GetAttribute('physics:kinematicEnabled')
            if kinematic_attr and kinematic_attr.Get():
                is_kinematic = True

        # Get the world transformation matrix
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_xform = xform_cache.GetLocalToWorldTransform(child)

        # Extract translation (position)
        pos = world_xform.ExtractTranslation()

        # Extract rotation as a quaternion.
        # ExtractRotation() assumes orthonormal columns and produces wrong
        # results when the matrix contains scale (e.g. scale=0.4).  We strip
        # scale by normalizing the 3x3 column vectors first.
        rot_mat = world_xform.ExtractRotationMatrix()
        col0 = Gf.Vec3d(rot_mat[0][0], rot_mat[1][0], rot_mat[2][0])
        col1 = Gf.Vec3d(rot_mat[0][1], rot_mat[1][1], rot_mat[2][1])
        col2 = Gf.Vec3d(rot_mat[0][2], rot_mat[1][2], rot_mat[2][2])
        sx, sy, sz = col0.GetLength(), col1.GetLength(), col2.GetLength()
        if sx > 1e-9 and sy > 1e-9 and sz > 1e-9:
            norm_mat = Gf.Matrix3d(
                col0[0]/sx, col1[0]/sy, col2[0]/sz,
                col0[1]/sx, col1[1]/sy, col2[1]/sz,
                col0[2]/sx, col1[2]/sy, col2[2]/sz,
            )
            rot = norm_mat.ExtractRotation().GetQuat()
        else:
            rot = Gf.Quatd(1, 0, 0, 0)
        rot_quat = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])

        object_info = {
            'name': name,
            'prim_path': str(child.GetPath()),
            'position': (pos[0], pos[1], pos[2]),
            'rotation': rot_quat,
            'payload': get_prim_payload(child, as_string=True),
            'rigid_body': rigid_body,
            'kinematic': is_kinematic,
            'static_body': not rigid_body,
            "description": description,
        }

        scene_objects.append(object_info)

    # Convert to tuple of frozen dicts for cache compatibility
    return tuple(tuple(sorted(obj.items())) for obj in scene_objects)


def get_usd_objects_info(usd_path: str) -> list:
    """
    Analyze a USD file and extract rigid bodies and static bodies.
    Results are cached internally to avoid re-parsing the same USD file.

    Args:
        usd_path: Path to the USD file

    Returns:
        List of dictionaries containing rigid bodies and static bodies with their properties
    """
    # Get cached result (tuple of tuples) and convert back to list of dicts
    cached_result = _get_usd_objects_info_cached(usd_path)
    return [dict(obj_tuple) for obj_tuple in cached_result]


def clear_usd_cache():
    """Clear the USD objects info cache. Call this if USD files have been modified."""
    _get_usd_objects_info_cached.cache_clear()

def np_to_gf_vec3d(pos: np.ndarray | list) -> Gf.Vec3d:
    return Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))

def np_to_gf_quatf(quat: np.ndarray | list, scalar_first=False) -> Gf.Quatf:
    # Gf.Quatf expects (real, imag)
    if scalar_first:
        return Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3])))
    else:
        return Gf.Quatf(float(quat[3]), Gf.Vec3f(float(quat[0]), float(quat[1]), float(quat[2])))

def translation_only_matrix(matrix: Gf.Matrix4d) -> Gf.Matrix4d:
    # Extract translation
    translation = matrix.ExtractTranslation()
    # Create identity matrix
    no_rot_matrix = Gf.Matrix4d(1.0)
    # Set translation
    no_rot_matrix.SetTranslate(translation)
    return no_rot_matrix


def get_prim_at_path(prim_path: str) -> Usd.Prim:  # noqa: N802 – keep API unchanged
    """Return the *Usd.Prim* located at *prim_path* on the current Isaac stage.

    This helper needs Isaac-Sim (``stage_utils``).  If the simulator has not
    been started yet, a helpful *ImportError* is raised instead of letting a
    bare ``AttributeError`` escape.
    """

    if stage_utils is None:
        raise ImportError(
            "get_prim_at_path requires Isaac-Sim; start it with AppLauncher or "
            "ensure isaacsim python paths are on PYTHONPATH before calling this "
            "function."
        )

    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")

def get_attribute_names(prim: Usd.Prim) -> List[str]:
    """Get the attribute names for a prim."""
    return [attribute.GetName() for attribute in prim.GetAttributes()]

def get_attribute(prim: Usd.Prim, attribute_name: str) -> Optional[Usd.Attribute]:
    """Get an attribute if it exists."""
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        return None
    return attribute

def get_scale(prim: Usd.Prim) -> Gf.Vec3d:
    """
    Get the scale parameter applied to a Usd.Prim.

    This function tries multiple approaches to get the scale:
    1. Directly from the 'xformOp:scale' attribute if it exists
    2. From the transform matrix using ExtractScale() method
    3. Returns (1, 1, 1) as default if no scale is found

    Args:
        prim: The Usd.Prim to get scale from

    Returns:
        Gf.Vec3d: The scale vector (x, y, z)
    """
    # First try to get scale directly from xformOp:scale attribute
    scale_attr = get_attribute(prim, "xformOp:scale")
    if scale_attr and scale_attr.IsValid():
        scale_value = scale_attr.Get()
        if scale_value is not None:
            # Convert to Gf.Vec3d if it's not already
            if isinstance(scale_value, (list, tuple)):
                return Gf.Vec3d(*scale_value)
            elif hasattr(scale_value, '__len__') and len(scale_value) == 3:
                return Gf.Vec3d(scale_value[0], scale_value[1], scale_value[2])
            else:
                return Gf.Vec3d(scale_value, scale_value, scale_value)

    # Default scale if nothing else works
    return Gf.Vec3d(1.0, 1.0, 1.0)

def get_descendant_prims(prim: Usd.Prim):
    """Get the descendants of a prim using a preorder traversal."""
    yield from prim.GetChildren()
    for child_prim in prim.GetChildren():
        yield from get_descendant_prims(child_prim)


def get_dimensions(body_prim: Usd.Prim) -> np.ndarray:
    """
    Calculate the dimensions (width, height, depth) of a USD prim's bounding box.

    This function computes the size of the axis-aligned bounding box that encompasses
    all geometry within the prim and its descendants.

    Args:
        body_prim (Usd.Prim): The USD prim to analyze

    Returns:
        np.ndarray: The dimensions [width, height, depth] corresponding to
                   [x_size, y_size, z_size], shape (3,)

    Example:
        >>> prim = stage.GetPrimAtPath("/World/Cube")
        >>> dims = get_dimensions(prim)
        >>> print(f"Dimensions: {dims}")  # e.g., [2.0, 2.0, 2.0] for 2x2x2 cube
    """
    # prim_helper.export_geometry_as_obj_file
    time_code = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
    bbox_cache.Clear()

    body_range = Gf.Range3d()
    for prim in get_descendant_prims(body_prim):
        # prim_bbox = bbox_cache.ComputeUntransformedBound(body_prim)  # ComputeLocalBound
        prim_bbox = bbox_cache.ComputeRelativeBound(prim, body_prim)
        prim_range = prim_bbox.ComputeAlignedRange()
        body_range.UnionWith(prim_range)
    return np.array(body_range.GetSize())

def get_bbox(body_prim: Usd.Prim, pos=None, quat=None, scalar_first=False):
    """
    Compute the oriented bounding box (OBB) for a USD prim with optional transform.

    This function calculates the 8 corner points of the bounding box around the prim's
    geometry, applies scaling and transformations, and optionally applies an additional
    user-specified transform (useful for placing the bbox at a specific pose).

    Computation steps:
        1. Compute the local bounding box via UsdGeom.BBoxCache
        2. Extract the 8 corners from the bounding range + local matrix
        3. Compute the centroid via ComputeCentroid()
        4. Transform corners and centroid to origin (inverse world transform)
        5. Apply prim scale to both corners and centroid
        6. Optionally apply a user-supplied rotation + translation

    Args:
        body_prim (Usd.Prim): The USD prim to compute bounding box for
        pos (array-like, optional): Additional translation [x, y, z] to apply.
                                   Can be np.ndarray, list, or None.
        quat (array-like, optional): Additional rotation quaternion to apply.
                                    Can be np.ndarray, list, or None.
        scalar_first (bool): If True, quaternion is in [w, x, y, z] format.
                           If False, quaternion is in [x, y, z, w] format.

    Returns:
        tuple[list, Gf.Vec3d]: A tuple containing:
            - corners (list): List of 8 Gf.Vec3d points representing the bounding box corners.
              The corners are ordered as:
              [0-3]: bottom face corners, [4-7]: top face corners
            - centroid (Gf.Vec3d): The center point of the bounding box

    Example:
        >>> prim = stage.GetPrimAtPath("/World/Cube")
        >>> corners, centroid = get_bbox(prim)
        >>> print(f"8 corners: {len(corners)}")
        >>> print(f"Centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
        >>>
        >>> # With additional transform
        >>> pos = [1.0, 2.0, 3.0]
        >>> quat = [0.0, 0.0, 0.0, 1.0]  # no rotation
        >>> corners, centroid = get_bbox(prim, pos, quat, scalar_first=False)
    """
    time_code = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
    bbox_cache.Clear()

    # Use ComputeWorldBound so corners/centroid are in world space.
    # Applying world_xform.GetInverse() to world-space data correctly yields
    # prim-local coordinates. ComputeLocalBound returns parent-frame data, which
    # produces wrong results in multi-env where env_origins != (0,0,0).
    prim_bbox = bbox_cache.ComputeWorldBound(body_prim)

    # Get corners and centroid (now in world space)
    range3d = prim_bbox.GetRange()
    matrix = prim_bbox.GetMatrix()

    corners = [matrix.Transform(range3d.GetCorner(i)) for i in range(8)]
    centroid = prim_bbox.ComputeCentroid()      # GfVec3d [1][2]

    # Transform to prim-local frame using the inverse of the prim's world transform
    xform_cache = UsdGeom.XformCache(time_code)
    world_xform = xform_cache.GetLocalToWorldTransform(body_prim)  # Gf.Matrix4d[2][3]
    transform = world_xform.GetInverse()

    transformed_corners = [transform.Transform(corner) for corner in corners]
    transformed_centroid = transform.Transform(centroid)

    scale = get_scale(body_prim)

    if scale is not Gf.Vec3d(1.0, 1.0, 1.0):
        # Scale corners directly
        scaled_corners = []
        for corner in transformed_corners:
            # Scale each corner directly
            scaled_corner = Gf.Vec3d(
                corner[0] * scale[0],
                corner[1] * scale[1],
                corner[2] * scale[2]
            )
            scaled_corners.append(scaled_corner)

        transformed_corners = scaled_corners

        # Scale centroid to match corners. The centroid from ComputeCentroid()
        # is in the prim's local space, and after inverse-world-transform it
        # still carries the original mesh magnitude. For non-unit-scale prims
        # (e.g., apple_01 at scale=0.01), the centroid must be multiplied by
        # the same scale factors applied to the corners, otherwise spatial
        # predicates (inside, above_top, center_of, etc.) receive a centroid
        # that is orders of magnitude off from the actual scaled geometry.
        transformed_centroid = Gf.Vec3d(
            transformed_centroid[0] * scale[0],
            transformed_centroid[1] * scale[1],
            transformed_centroid[2] * scale[2],
        )

    # User-supplied transform, if any
    if quat is not None and pos is not None:
        if isinstance(pos, np.ndarray) or isinstance(pos, list):
            pos = np_to_gf_vec3d(pos)
        if isinstance(quat, np.ndarray) or isinstance(quat, list):
            quat = np_to_gf_quatf(quat, scalar_first)

        additional_transform = Gf.Matrix4d().SetRotateOnly(quat)
        additional_transform.SetTranslateOnly(pos)
    else:
        # No additional transform
        return transformed_corners, transformed_centroid

    # Apply additional transform
    new_corners = [additional_transform.Transform(corner) for corner in transformed_corners]
    new_centroid = additional_transform.Transform(transformed_centroid)

    return new_corners, new_centroid

    # return transformed_corners, transformed_centroid


def get_aabb(body_prim: Usd.Prim) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Axis-Aligned Bounding Box (AABB) for a USD prim in local coordinates.

    The AABB is the smallest box aligned with the coordinate axes that completely
    contains the prim's geometry. This function returns the minimum and maximum
    corner coordinates of this box.

    Args:
        body_prim (Usd.Prim): The USD prim to compute AABB for

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - lower (np.ndarray): Minimum corner coordinates [x_min, y_min, z_min], shape (3,)
            - upper (np.ndarray): Maximum corner coordinates [x_max, y_max, z_max], shape (3,)

    Example:
        >>> prim = stage.GetPrimAtPath("/World/Cube")
        >>> lower, upper = get_aabb(prim)
        >>> print(f"Lower corner: {lower}")  # e.g., [-1.0, -1.0, -1.0]
        >>> print(f"Upper corner: {upper}")  # e.g., [1.0, 1.0, 1.0]
        >>> size = upper - lower             # [2.0, 2.0, 2.0] for 2x2x2 cube
        >>> center = (lower + upper) / 2     # [0.0, 0.0, 0.0] geometric center

    Note:
        This computes the bounding box in the prim's local coordinate system,
        not in world coordinates. For world-coordinate bounding boxes,
        use get_bbox() with appropriate transforms.
    """
    time_code = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
    bbox_cache.Clear()

    prim_bbox = bbox_cache.ComputeLocalBound(body_prim)
    prim_range = prim_bbox.ComputeAlignedRange()
    lower = np.array(prim_range.GetMin())
    upper = np.array(prim_range.GetMax())
    aabb = (lower, upper)

    return aabb

def get_scene_payloads(usd_path: str) -> dict[str, list[str]]:
    """
    Generate a dc

    Args:
        usd_path (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        dict[str, list[str]]: _description_
    """
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise ValueError(f"Failed to open stage: {usd_path}")

    # Get the root prim (e.g., '/World')
    root = stage.GetDefaultPrim()
    if not root:
        root = stage.GetPseudoRoot()

    payload_dict = {}

    # Iterate over first-level children
    for child in root.GetChildren():
        # Check if this is an Xform prim
        if child.GetTypeName() == "Xform":
            prim_path = str(child.GetPath())
            payload_list = get_prim_payload(child, as_string=True)
            payload_dict[prim_path] = payload_list

    return payload_dict


def get_prim_payload(prim: Usd.Prim, as_string=True) -> list:
    """Get the payload of a prim."""
    payload_list = []
    # Get payloads by inspecting the prim stack
    for primSpec in prim.GetPrimStack():
        payloads = primSpec.payloadList.prependedItems
        if as_string:
            payload_list.extend([payload.assetPath for payload in payloads])
        else:
            payload_list.extend(payloads)

    return payload_list
