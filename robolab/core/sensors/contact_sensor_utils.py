# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg


def create_contact_sensor_cfg(entity_1, entity_2, update_period=0.0, history_length=6, debug_vis=False):
        return ContactSensorCfg(
            prim_path=entity_1,
            update_period=update_period,
            history_length=history_length,
            debug_vis=debug_vis,
            filter_prim_paths_expr=[entity_2],
        )


def create_batch_contact_sensor_cfg(entity_prim_path: str, filter_prim_paths: list[str],
                                     update_period=0.0, history_length=6, debug_vis=False):
    """
    Create a contact sensor that monitors contacts between one entity and multiple filter entities.

    This is more efficient than creating individual pairwise sensors when you need to check
    one body (e.g., gripper) against many objects at once.

    Args:
        entity_prim_path: The prim path of the body to monitor contacts for
        filter_prim_paths: List of prim paths to check contacts against
        update_period: Sensor update period
        history_length: Number of contact history frames to store
        debug_vis: Enable debug visualization

    Returns:
        ContactSensorCfg configured for batch contact detection
    """
    return ContactSensorCfg(
        prim_path=entity_prim_path,
        update_period=update_period,
        history_length=history_length,
        debug_vis=debug_vis,
        filter_prim_paths_expr=filter_prim_paths,
    )


def validate_contact_grippers(contact_gripper: dict) -> dict:
    """Validate a robot's contact_gripper declaration and return its concrete entries.

    Entries are either concrete (label -> prim path, gets sensors) or an alias
    group (label -> list of concrete labels, meaning "any member" at query
    time). Groups must be flat: members must be concrete labels declared in
    the same dict, so query-time resolution is a single lookup.

    Every robot must declare a "gripper" entry — concrete for single-gripper
    robots, or a group like ["left", "right"] — because benchmark task
    conditionals default to gripper_name="gripper".
    """
    concrete = {name: target for name, target in contact_gripper.items() if isinstance(target, str)}
    groups = {name: target for name, target in contact_gripper.items() if not isinstance(target, str)}
    for name, members in groups.items():
        if not isinstance(members, (list, tuple)) or not members:
            raise ValueError(
                f"contact_gripper entry '{name}' must be a prim path or a non-empty list of labels, got {members!r}."
            )
        unknown = [member for member in members if member not in concrete]
        if unknown:
            raise ValueError(
                f"contact_gripper group '{name}' references {unknown}, which are not concrete gripper labels. "
                f"Group members must be prim-path entries in the same dict (no nested groups); "
                f"declared concrete labels: {sorted(concrete)}."
            )
    if "gripper" not in contact_gripper:
        raise ValueError(
            "contact_gripper must declare a 'gripper' entry (a prim path, or a group such as "
            "['left', 'right'] on multi-gripper robots) — benchmark task conditionals assume it. "
            f"Declared labels: {sorted(contact_gripper)}."
        )
    return concrete


def create_contact_sensors(env_cfg):
    """
    Dynamically create contact sensors based on contact_gripper and contact_object_list.

    Creates:
    1. Batch sensors for each gripper (gripper vs all objects) - named "{gripper_name}__batch"
    2. Individual pairwise sensors for gripper-object pairs (for backwards compatibility)
    3. Individual pairwise sensors for object-object pairs

    Alias-group entries in contact_gripper (list values, e.g. "gripper":
    ["left", "right"]) get no sensors of their own; they resolve to their
    members' sensors at query time in WorldState.

    Args:
        env_cfg: Environment configuration containing scene, contact_gripper, and contact_object_list
    """

    scene = env_cfg.scene
    if env_cfg.contact_object_list is None or env_cfg.contact_gripper is None:
        return

    concrete_grippers = validate_contact_grippers(env_cfg.contact_gripper)

    # Objects to exclude from batch sensor (these are checked separately or not relevant for wrong-grab detection)
    batch_sensor_exclude = {"table"}

    for gripper_name, gripper_prim_path in concrete_grippers.items():
        # Create batch sensor for gripper vs all objects except excluded ones (for efficient batch queries)
        all_object_prim_paths = [getattr(scene, obj_name).prim_path for obj_name in env_cfg.contact_object_list if obj_name not in batch_sensor_exclude]
        batch_sensor_name = f"{gripper_name}__all_objs"
        batch_sensor = create_batch_contact_sensor_cfg(gripper_prim_path, all_object_prim_paths)
        setattr(scene, batch_sensor_name, batch_sensor)

        # Also create individual pairwise sensors (for backwards compatibility with in_contact)
        for obj_name in env_cfg.contact_object_list:
            contact_sensor_name = f"{gripper_name}__{obj_name}"
            contact_sensor = create_contact_sensor_cfg(gripper_prim_path, getattr(scene, obj_name).prim_path)
            setattr(scene, contact_sensor_name, contact_sensor)

    objects = env_cfg.contact_object_list
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj_name = objects[i]
            other_obj_name = objects[j]
            contact_sensor_name = f"{obj_name}__{other_obj_name}"
            contact_sensor = create_contact_sensor_cfg(
                getattr(scene, obj_name).prim_path,
                getattr(scene, other_obj_name).prim_path)
            setattr(scene, contact_sensor_name, contact_sensor)

def get_contact_sensors(scene):
    """
    Get all contact sensors.

    Example:
        env = create_env(...)
        contact_sensors = get_contact_sensors(env.scene)

    Args:
        scene (InteractiveScene): The scene to get the contact sensors from.
    """
    # Isaac Lab 3 instantiates sensors through a backend factory.  The runtime
    # object is consequently an isaaclab_physx ContactSensor rather than an
    # instance of the public isaaclab.sensors.ContactSensor facade.  The config
    # class remains backend-independent in both Isaac Lab 2.x and 3.x.
    contact_sensors = {
        name: sensor for name, sensor in scene.sensors.items()
        if isinstance(sensor, ContactSensor)
        or isinstance(getattr(sensor, "cfg", None), ContactSensorCfg)
    }
    return contact_sensors

def get_contact_sensor(scene, body1, body2) -> ContactSensor:
    contact_sensor_string = f"{body1}__{body2}"
    contact_sensor_string2 = f"{body2}__{body1}"

    sensors = get_contact_sensors(scene)
    if contact_sensor_string in sensors.keys():
        return sensors[contact_sensor_string]
    elif contact_sensor_string2 in sensors.keys():
        return sensors[contact_sensor_string2]
    else:
        raise ValueError(f"Contact sensor {contact_sensor_string} or {contact_sensor_string2} not found. available sensors: {sensors.keys()}")


def get_contact_sensor_with_order(scene, body1, body2) -> tuple[ContactSensor, bool]:
    """
    Get the contact sensor between two bodies and whether the order is reversed.

    Args:
        scene: The scene containing sensors
        body1: First body name (desired sensor body - forces on this body)
        body2: Second body name (desired filter body - forces from this body)

    Returns:
        tuple: (ContactSensor, is_reversed)
            - ContactSensor: The contact sensor
            - is_reversed: True if sensor was found as body2__body1 (order reversed),
                           meaning force_matrix_w gives forces on body2 from body1,
                           so the force needs to be negated to get forces on body1.
    """
    contact_sensor_string = f"{body1}__{body2}"
    contact_sensor_string2 = f"{body2}__{body1}"

    sensors = get_contact_sensors(scene)
    if contact_sensor_string in sensors.keys():
        return sensors[contact_sensor_string], False
    elif contact_sensor_string2 in sensors.keys():
        return sensors[contact_sensor_string2], True
    else:
        raise ValueError(f"Contact sensor {contact_sensor_string} or {contact_sensor_string2} not found. available sensors: {sensors.keys()}")


def get_batch_contact_sensor(scene, body: str) -> ContactSensor | None:
    """
    Get the batch contact sensor for a body (e.g., gripper).

    Args:
        scene: The scene containing sensors
        body: Name of the body (e.g., "gripper")

    Returns:
        The batch ContactSensor if it exists, None otherwise
    """
    batch_sensor_name = f"{body}__all_objs"
    sensors = get_contact_sensors(scene)
    return sensors.get(batch_sensor_name)
