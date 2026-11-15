# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass


def _is_event_term(value: object) -> bool:
    return hasattr(value, "func") and hasattr(value, "mode")


def _event_items(events: dict | object | None) -> list[tuple[str, object]]:
    """``(name, EventTerm)`` pairs of a dict or events configclass, in declaration order."""
    if events is None:
        return []
    if isinstance(events, dict):
        items = events.items()
    elif vars(events):
        # Instance attributes keep declaration order; dir() would sort them.
        items = vars(events).items()
    else:
        items = ((name, getattr(events, name)) for name in dir(events))
    return [(name, value) for name, value in items if not name.startswith("_") and _is_event_term(value)]


def _ordered_events_cfg(class_name: str, items: list[tuple[str, object]]) -> object:
    """Events configclass whose terms run in the order of ``items``.

    The event manager runs terms in the instance's attribute order. Declaring the
    terms as annotated fields fixes that order; attributes set on an undecorated
    class would instead be copied in alphabetical order by ``configclass``.
    """
    namespace = {"__annotations__": {name: type(term) for name, term in items}, **dict(items)}
    return configclass(type(class_name, (), namespace))()


def create_events_cfg_from_dict(events_dict: dict) -> object:
    """
    Create an events configclass from a dictionary of {event_name: EventTerm}.

    Terms run in the dictionary's order.

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
                params={"camera_names": ["over_shoulder_left_camera"], "pose_range": {"x": (-0.05, 0.05)}}
            )
        })
    """
    return _ordered_events_cfg("DynamicEventsCfg", _event_items(events_dict))


def merge_events_cfg(existing_events_cfg: object, new_events: dict | object) -> object:
    """
    Merge new events into an existing events configclass, preserving existing events.

    This function copies all events from the existing config and adds/overrides with
    new events. This ensures that default events (like reset_scene_to_default) are
    preserved when adding custom events. Existing events keep their order and run
    first; new events run after them in their given order (an overridden event keeps
    its original position).

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
                params={"camera_names": ["over_shoulder_left_camera"], "pose_range": {"x": (-0.05, 0.05)}}
            )
        })
        # merged now has both 'reset' and 'randomize_camera' events, 'reset' first
    """
    merged = dict(_event_items(existing_events_cfg))
    merged.update(_event_items(new_events))
    return _ordered_events_cfg("MergedEventsCfg", list(merged.items()))
