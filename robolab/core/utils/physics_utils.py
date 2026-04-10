# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import omni
from pxr import Usd, UsdGeom, UsdPhysics

import robolab.core.utils.usd_utils as usd_utils


def find_physics_material_in_children(parent_prim, stage: Usd.Stage) -> Usd.Prim:
    """Recursively search for physics material bound to mesh prims."""
    for child_prim in parent_prim.GetAllChildren():
        # Check if this child is a mesh
        if child_prim.IsA(UsdGeom.Mesh):
            # Check for physics material using relationship names
            relationship_names = [
                "material:binding:physics",      # USD Material binding for physics purpose
                "material:binding",              # General material binding
                "physxMaterial:physicsMaterial", # NVIDIA Omniverse (PhysX)
                "physics:material:binding"       # USD Physics Schema
            ]

            for rel_name in relationship_names:
                if child_prim.HasRelationship(rel_name):
                    rel = child_prim.GetRelationship(rel_name)
                    targets = rel.GetTargets()
                    if targets:
                        material_prim_path = targets[0].pathString
                        potential_material = stage.GetPrimAtPath(material_prim_path)
                        # Check if this material has physics properties
                        if potential_material.IsValid() and potential_material.HasAPI(UsdPhysics.MaterialAPI):
                            print(f"Physics material found via '{rel_name}': {potential_material.GetPath()}")
                            return potential_material
                        else:
                            print(f"Material found via '{rel_name}' but no physics API: {material_prim_path}")

            print(f"No physics material bound to mesh: {child_prim.GetPath()}")

        # Recursively search in child's children
        result = find_physics_material_in_children(child_prim, stage)
        if result:
            return result
    return None


def get_friction(prim: Usd.Prim, stage: Usd.Stage) -> dict:
    # Traverse all child prims to find meshes with bound physics materials
    material_prim = None

    material_prim = find_physics_material_in_children(prim, stage)

    if not material_prim:
        print(f"No physics material found at {prim.GetPath()}")
        return None

    material_api = UsdPhysics.MaterialAPI(material_prim)

    result = {'prim_path': material_prim.GetPath().pathString}

    static_attr = material_api.GetStaticFrictionAttr()
    if static_attr:
        result['static_friction'] = static_attr.Get()

    dynamic_attr = material_api.GetDynamicFrictionAttr()
    if dynamic_attr:
        result['dynamic_friction'] = dynamic_attr.Get()

    restitution_attr = material_api.GetRestitutionAttr()
    if restitution_attr:
        result['restitution'] = restitution_attr.Get()

    density_attr = material_api.GetDensityAttr()
    if density_attr:
        result['density'] = density_attr.Get()

    return result


def update_friction(prim: Usd.Prim, stage: Usd.Stage, static_friction: float, dynamic_friction: float, restitution: float) -> None:
    material_prim = find_physics_material_in_children(prim, stage)
    if not material_prim:
        print(f"No physics material found at {prim.GetPath()}")
        return

    material_api = UsdPhysics.MaterialAPI(material_prim)
    if static_friction:
        old_static_friction = material_api.GetStaticFrictionAttr().Get()
        print(f"Updating static friction from {old_static_friction} to {static_friction}")
        material_api.GetStaticFrictionAttr().Set(static_friction)
    if dynamic_friction:
        old_dynamic_friction = material_api.GetDynamicFrictionAttr().Get()
        print(f"Updating dynamic friction from {old_dynamic_friction} to {dynamic_friction}")
        material_api.GetDynamicFrictionAttr().Set(dynamic_friction)
    if restitution:
        old_restitution = material_api.GetRestitutionAttr().Get()
        print(f"Updating restitution from {old_restitution} to {restitution}")
        material_api.GetRestitutionAttr().Set(restitution)


def add_friction(static_friction: float=5.0,
                 dynamic_friction: float=5.0,
                 restitution: float=0.25,
                 root_prim_path: str=None, ) -> dict:
    """ 0.8 is rubber like
        restitution is the amount of "bounce". 1.0 means that it always return to the original bounce position. PhysX default is 0.25. We will set to 0.
        Friction combination is by default 'average' (between the two physics_materials interacting with each other). This is not exposed here to be modified.
    """
    from omni.physx.scripts.physicsUtils import add_physics_material_to_prim
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_prim, root_prim_path = usd_utils.get_root_prim_path(root_prim_path)

    data_dict = {}

    from isaacsim.core.api.materials.physics_material import PhysicsMaterial

    material = PhysicsMaterial(
                prim_path=root_prim_path + "/physics_material",
                name=f"physics_material",
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,)

    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            print(f"\tStatic friction: {static_friction} Dynamic friction: {dynamic_friction} Restitution: {restitution}")
            add_physics_material_to_prim(stage, prim, material.prim_path)
            data_dict[root_prim_path] = {'static': static_friction,
                                                   'dynamic':dynamic_friction,
                                                   'restitution': restitution,
                                                   'physics_material': material.prim_path}
    return data_dict

def add_mass_api(mass: float, root_prim_path: str=None) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    root_prim, root_prim_path = usd_utils.get_root_prim_path(root_prim_path)

    if mass <= 0:
        raise ValueError(f"Cannot set mass to <= 0: {mass}")

    data_dict = {}

    # Total mass is applied to the root Xform only.
    if root_prim.IsA(UsdGeom.Xform):
        print(f"\tAdd mass: {mass} to root prim: {root_prim}")
        massAPI = UsdPhysics.MassAPI.Apply(root_prim)
        massAPI.CreateMassAttr(mass)
        data_dict[root_prim_path] = mass

    # Automatically compute mass for the rest of the meshes.
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            massAPI = UsdPhysics.MassAPI.Apply(prim)
            data_dict[root_prim_path] = "auto"

    return data_dict

def get_rigid_body_collider(prim: Usd.Prim, stage: Usd.Stage) -> dict:
    from pxr import UsdGeom, UsdPhysics

    result = {}

    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    if rigid_body_api:
        kin_attr = rigid_body_api.GetKinematicEnabledAttr()
        kin = kin_attr.Get() if kin_attr else None
        result['kinematic'] = kin

    # Traverse all prims in the stage starting at this path
    for subprim in Usd.PrimRange(prim):
        print(f"Processing prim: {subprim.GetPath()}")
        # Only process shapes and meshes
        if (
            subprim.IsA(UsdGeom.Cylinder)
            or subprim.IsA(UsdGeom.Capsule)
            or subprim.IsA(UsdGeom.Cone)
            or subprim.IsA(UsdGeom.Sphere)
            or subprim.IsA(UsdGeom.Cube)
            or subprim.IsA(UsdGeom.Mesh)
        ):
            # Check standard USD Physics collider
            collision_api = UsdPhysics.CollisionAPI(subprim)
            if collision_api:
                enabled_attr = collision_api.GetCollisionEnabledAttr()
                enabled = enabled_attr.Get() if enabled_attr else None
                print(f"  UsdPhysics Collider found. Enabled: {enabled}")
                result['usd_collision_enabled'] = enabled

            # Check PhysX collider (if running in Isaac/Omniverse)
            try:
                from pxr import PhysxSchema
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(subprim)
                if physx_collision_api:
                    approx_attr = physx_collision_api.GetApproximationAttr()
                    approx = approx_attr.Get() if approx_attr else None
                    print(f"  PhysxSchema Collider found. Approximation: {approx}")
                    result['physx_approximation'] = approx
            except ImportError:
                pass  # PhysxSchema not available

    if not result:
        print(f"No rigid body collider (or collision API) found at {prim.GetPath()}")
        return None

    return result

def update_rigid_body_collider(
    prim: Usd.Prim,
    contact_offset: float = 0.02,
    rest_offset: float = 0.01,
    enable_ccd: bool = True,
    kinematic: bool = False
) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    # Check base prim for RigidBodyAPI
    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    if rigid_body_api:
        kin_attr = rigid_body_api.GetKinematicEnabledAttr()
        kin = kin_attr.Get() if kin_attr else None
        print(f"Base prim {prim.GetPath()} has RigidBodyAPI (kinematic: {kin})")
        if kin != kinematic:
            print(f"Updating kinematic from {kin} to {kinematic}")
            rigid_body_api.GetKinematicEnabledAttr().Set(kinematic)
    else:
        print(f"Base prim {prim.GetPath()} has no RigidBodyAPI.")

    rigid_body_collider_params = {'approximation': 'convexDecomposition',
                           'kinematic': False,
                           'attributes': {
                               'physxConvexDecompositionCollision:maxConvexHulls': 256,
                               'physxConvexDecompositionCollision:shrinkWrap': 1,
                               'physxConvexDecompositionCollision:errorPercentage': 1,
                            },
        }
    attributes_dict = rigid_body_collider_params.get('attributes')
    approximation = rigid_body_collider_params.get('approximation')

    # Recursively traverse all children/shapes/meshes
    from omni.physx.scripts import utils
    for child_prim in Usd.PrimRange(prim):
        if (
            child_prim.IsA(UsdGeom.Cylinder)
            or child_prim.IsA(UsdGeom.Capsule)
            or child_prim.IsA(UsdGeom.Cone)
            or child_prim.IsA(UsdGeom.Sphere)
            or child_prim.IsA(UsdGeom.Cube)
            or child_prim.IsA(UsdGeom.Mesh)
        ):
            print(f"Processing child prim: {child_prim.GetPath()}")

              # Remove UsdPhysics.CollisionAPI
            if UsdPhysics.CollisionAPI(child_prim):
                print(f"  Removing UsdPhysics.CollisionAPI from {child_prim.GetPath()}")
                UsdPhysics.CollisionAPI(child_prim).GetPrim().RemoveAPI(UsdPhysics.CollisionAPI)

            # Remove PhysxSchema.PhysxCollisionAPI if present
            try:
                from pxr import PhysxSchema
                if PhysxSchema.PhysxCollisionAPI(child_prim):
                    print(f"  Removing PhysxSchema.PhysxCollisionAPI from {child_prim.GetPath()}")
                    PhysxSchema.PhysxCollisionAPI(child_prim).GetPrim().RemoveAPI(PhysxSchema.PhysxCollisionAPI)
            except ImportError:
                pass

            # Apply new PhysxSchema Collider
            add_rigid_body_collider(child_prim, approximation=approximation, kinematic=kinematic, attributes=attributes_dict, contact_offset=contact_offset, rest_offset=rest_offset, enable_ccd=enable_ccd)

    print("Collider update complete.")

def add_rigid_body_collider(prim: Usd.Prim,
                                        approximation: str = "convexDecomposition",
                                        kinematic: bool = False,
                                        attributes: dict= {
                                                            'physxConvexDecompositionCollision:maxConvexHulls': 256,
                                                            'physxConvexDecompositionCollision:shrinkWrap': 1,
                                                            'physxConvexDecompositionCollision:errorPercentage': 1,
                                                            },
                                        contact_offset: float = 0.02,
                                        rest_offset: float = 0.01,
                                        enable_ccd: bool=True,):
    from omni.physx.scripts import utils
    from pxr import PhysxSchema, Usd, UsdGeom

    # Add RigidBodyAPI to base prim
    if prim.IsA(UsdGeom.Xform):
        utils.setPhysics(prim, kinematic=kinematic)
        physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rigid_body.CreateEnableCCDAttr().Set(enable_ccd)  # Enable CCD - dynamically increases the contact offset based on the object's velocity
        print(f"\tAdding rigidBodyAPI to: {prim.GetPath()}")
        print(f"\tEnable CCD '{enable_ccd}' to: {prim.GetPath()}")
    else:
        print(f"\t[WARNING] {prim.GetPath()} is NOT a Xform ({prim.GetPrimTypeInfo().GetTypeName()}). Not adding rigid body physics.")

    # Traverse all prims in the stage starting at this path
    for prim in Usd.PrimRange(prim):
        # only process shapes and meshes
        if (
            prim.IsA(UsdGeom.Cylinder)
            or prim.IsA(UsdGeom.Capsule)
            or prim.IsA(UsdGeom.Cone)
            or prim.IsA(UsdGeom.Sphere)
            or prim.IsA(UsdGeom.Cube)
        ):
            # use a ConvexHull for regular prims
            utils.setCollider(prim, approximationShape="convexHull")
        elif prim.IsA(UsdGeom.Mesh):

            # "None" will use the base triangle mesh if available
            # Can also use "convexDecomposition", "convexHull", "boundingSphere", "boundingCube"
            print(f"\tAdding collider '{approximation}' to: {prim.GetPath()}")

            utils.setCollider(prim, approximationShape=approximation)

            if attributes is not None:
                for attribute, value in attributes.items():
                    prim.GetAttribute(attribute).Set(value)


                collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                if contact_offset:
                    collision_api.GetContactOffsetAttr().Set(contact_offset)
                if rest_offset:
                    collision_api.GetRestOffsetAttr().Set(rest_offset)

            if approximation == "sdf":
                # set sdf resolution to 256
                meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                meshCollision.CreateSdfResolutionAttr().Set(256)
                # Increase the contact offset
                physxCollisionApi = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physxCollisionApi.CreateContactOffsetAttr().Set(0.25)  # Increase the contact offset for better collision detection, especially for fast moving objects

                rigidBodyApi = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rigidBodyApi.CreateSolverPositionIterationCountAttr().Set(30)  # Increase position iterations for better solver convergence
                rigidBodyApi.CreateMaxDepenetrationVelocityAttr().Set(100)  # Reduce depenetration velocity to avoid violent collision responses
                rigidBodyApi.CreateEnableSpeculativeCCDAttr().Set(True)  # Enable speculative CCD - dynamically increases the contact offset based on the object's velocity
                rigidBodyApi.CreateEnableCCDAttr().Set(enable_ccd)  # Enable CCD - dynamically increases the contact offset based on the object's velocity


                # for attribute, value in attributes_dict.items():
                #     prim.GetAttribute(attribute).Set(value)
        else:
            continue

    return

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

def add_attributes_to_prim(attribute_dict: dict, root_prim_path: str=None) -> dict:
    """ Add attribute to the meshes attached to this partiular prim
    """
    data_dict = {}

    from pxr import UsdGeom

    root_prim, root_prim_path = usd_utils.get_root_prim_path(root_prim_path)

    print(f"Adding attributes to {root_prim_path}")
    # Add attribute to only the root prim
    if root_prim.IsA(UsdGeom.Xform):
        for name, value in attribute_dict.items():
            set_str_attribute_to_prim(root_prim, name, value)
            print(f"\tAttribute '{name}': '{value}'")
        data_dict[root_prim_path] = attribute_dict

    # Traverse all prims in the stage starting at this path
    # for prim in Usd.PrimRange(root_prim):
    #     if prim.IsA(UsdGeom.Mesh):
    #         for name, value in attribute_dict.items():
    #             set_str_attribute_to_prim(prim, name, value)
    #             print(f"Set {prim} {name} to: {value}")

    return data_dict

def add_semantics_to_prim(semantic_labels: dict,
                            root_prim_path=None) -> dict:

    from omni.isaac.core.utils.semantics import add_update_semantics, get_semantics
    from pxr import Sdf, UsdGeom

    data_dict={}

    root_prim, root_prim_path = usd_utils.get_root_prim_path(root_prim_path)

    # Apply semantic label to the root of the xform.
    if root_prim.IsA(UsdGeom.Xform):
        i = 0
        for type, label in semantic_labels.items():
            add_update_semantics(root_prim, semantic_label=label, type_label=type, suffix=f"{i}")
            i += 1

        semantics = get_semantics(root_prim)
        print(f"Added semantics to {root_prim}, semantics: {semantics}")
        data_dict[root_prim_path] = semantic_labels

    # for prim in Usd.PrimRange(root_prim):
    #     if prim.IsA(UsdGeom.Mesh):
    #         for type, label in semantic_labels.items():
    #             add_update_semantics(prim, semantic_label=label, type_label=type)
    #         semantics = get_semantics(prim)
    #         print(f"Added semantics to {prim}, semantics: {semantics}")
    return data_dict

def transform_mesh_prim(position_offset=None,
                        scale: float | np.ndarray | list=None,
                        root_prim_path: str=None) -> dict:
    from omni.physx.scripts import physicsUtils
    from pxr import Gf, Usd, UsdGeom
    data_dict = {}

    if position_offset is None and scale is None:
        return data_dict

    root_prim, root_prim_path = usd_utils.get_root_prim_path(root_prim_path)

    if root_prim.IsA(UsdGeom.Xform):
        xform = UsdGeom.Xformable(root_prim)
        if scale is not None:
            if isinstance(scale, float):
                scale = [scale, scale, scale]
            elif isinstance(scale, np.ndarray):
                scale = scale.tolist()
            elif isinstance(scale, list):
                pass
            physicsUtils.set_or_add_scale_op(xform, scale=scale)
        if position_offset is not None:
            physicsUtils.set_or_add_translate_op(xform, [-position_offset[0],-position_offset[1],-position_offset[2]])
            root_prim.GetAttribute('xformOp:translate').Set(-position_offset[0],-position_offset[1],-position_offset[2])
        data_dict[root_prim_path] = {
                                               'translate':position_offset}

    # for prim in Usd.PrimRange(root_prim):
    #    # Only process mesh
    #     if prim.IsA(UsdGeom.Mesh):
    #         xform = UsdGeom.Xformable(prim)
    #         if scale is not None:
    #             scaleOp = xform.AddScaleOp()
    #             scaleOp.Set(Gf.Vec3d(scale,scale,scale))
    #         # if position_offset is not None:
    #         #     print(f"\tTranslation: '{position_offset}'")
    #         #     translateOp = xform.AddTranslateOp()
    #         #     translateOp.Set(Gf.Vec3d(-position_offset[0],-position_offset[1],-position_offset[2]))
    #         data_dict[root_prim_path] = {'scale': scale,
    #                                           }
    return data_dict
