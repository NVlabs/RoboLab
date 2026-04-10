# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from isaaclab.utils import configclass


def create_events_cfg_from_dict(events_dict: dict) -> object:
    """
    Create an events configclass from a dictionary of {event_name: EventTerm}.

    Args:
        events_dict: Dictionary mapping event names to EventTerm configurations.
            Example: {"reset_camera": EventTerm(func=reset_camera_pose_uniform, ...)}

    Returns:
        A configclass instance with the events as attributes.

    Example:
        from isaaclab.managers import EventTermCfg as EventTerm
        from robolab.core.events.reset_camera import reset_camera_pose_uniform

        events_cfg = create_events_cfg_from_dict({
            "reset_camera": EventTerm(
                func=reset_camera_pose_uniform,
                mode="reset",
                params={"camera_names": ["external_cam"], "pose_range": {"x": (-0.05, 0.05)}}
            )
        })
    """
    @configclass
    class DynamicEventsCfg:
        pass

    # Add each event as a class attribute
    for event_name, event_term in events_dict.items():
        setattr(DynamicEventsCfg, event_name, event_term)

    return DynamicEventsCfg()


def merge_events_cfg(existing_events_cfg: object, new_events: dict | object) -> object:
    """
    Merge new events into an existing events configclass, preserving existing events.

    This function copies all events from the existing config and adds/overrides with
    new events. This ensures that default events (like reset_scene_to_default) are
    preserved when adding custom events.

    Args:
        existing_events_cfg: The existing events configclass instance (e.g., BaseEventCfg()).
        new_events: Either a dictionary of {event_name: EventTerm} or a configclass instance
            with EventTerm attributes.

    Returns:
        A new configclass instance with merged events.

    Example:
        from isaaclab.managers import EventTermCfg as EventTerm
        from robolab.core.events.reset_camera import reset_camera_pose_uniform

        # existing_events has 'reset' event from BaseEventCfg
        merged = merge_events_cfg(existing_events, {
            "randomize_camera": EventTerm(
                func=reset_camera_pose_uniform,
                mode="reset",
                params={"camera_names": ["external_cam"], "pose_range": {"x": (-0.05, 0.05)}}
            )
        })
        # merged now has both 'reset' and 'randomize_camera' events
    """
    @configclass
    class MergedEventsCfg:
        pass

    # Copy existing events from the existing config
    if existing_events_cfg is not None:
        for attr_name in dir(existing_events_cfg):
            # Skip private/magic attributes and methods
            if attr_name.startswith('_'):
                continue
            attr_value = getattr(existing_events_cfg, attr_name)
            # Only copy EventTerm-like objects (have 'func' and 'mode' attributes)
            if hasattr(attr_value, 'func') and hasattr(attr_value, 'mode'):
                setattr(MergedEventsCfg, attr_name, attr_value)

    # Add/override with new events
    if isinstance(new_events, dict):
        for event_name, event_term in new_events.items():
            setattr(MergedEventsCfg, event_name, event_term)
    else:
        # new_events is a configclass instance
        for attr_name in dir(new_events):
            if attr_name.startswith('_'):
                continue
            attr_value = getattr(new_events, attr_name)
            if hasattr(attr_value, 'func') and hasattr(attr_value, 'mode'):
                setattr(MergedEventsCfg, attr_name, attr_value)

    return MergedEventsCfg()

