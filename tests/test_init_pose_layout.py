# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry of initial-pose randomization: grouping, workspace limits, collision checks, seeding."""

import argparse
import math

import numpy as np
import pytest

from robolab.core.events.init_pose_layout import (
    Footprint,
    Group,
    GroupMotion,
    LayoutError,
    Workspace,
    build_groups,
    convex_hull,
    default_yaw_range_deg,
    episode_rng,
    polygons_overlap,
    sample_layout,
    transform_group,
    verify_layout,
)

WORKSPACE = Workspace(0.2, 0.9, -0.4, 0.4)


def box(x, y, half=(0.03, 0.03), yaw=0.0):
    """Rectangle outline centered at (x, y), turned by ``yaw``."""
    hx, hy = half
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    cos, sin = math.cos(yaw), math.sin(yaw)
    return local @ np.array([[cos, sin], [-sin, cos]]) + np.array([x, y])


def footprint(name, x, y, half=(0.03, 0.03), fixed_reason="", yaw=0.0):
    return Footprint(name, box(x, y, half, yaw), fixed_reason)


def groups_by_member(groups):
    return {tuple(sorted(group.members)): group for group in groups}


def test_convex_hull_drops_interior_points_and_handles_degenerate_input():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.2, 0.7]])
    hull = convex_hull(points)
    assert len(hull) == 4
    signed_area = 0.5 * np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - np.roll(hull[:, 0], -1) * hull[:, 1])
    assert signed_area == pytest.approx(1.0)  # counter-clockwise
    assert len(convex_hull(np.array([[0, 0], [1, 0]]))) == 4  # a line becomes a thin rectangle


def test_polygons_overlap_respects_rotation_and_clearance():
    bar = box(0.0, 0.0, (0.10, 0.01))
    dot = box(0.0, 0.05, (0.01, 0.01))
    # A long thin bar does not reach a point 5 cm away when lying along x ...
    assert not polygons_overlap(bar, dot)
    # ... but does once it is turned 90 degrees.
    assert polygons_overlap(box(0.0, 0.0, (0.10, 0.01), math.pi / 2), dot)
    # A 5 mm gap counts as overlap under a 1 cm clearance.
    near = box(0.205, 0.0, (0.10, 0.01))
    assert not polygons_overlap(bar, near)
    assert polygons_overlap(bar, near, clearance=0.01)


def test_outlines_do_not_group_objects_whose_boxes_would_touch():
    # A thin diagonal stick's world-axis box covers the cube; its outline does not.
    stick = Footprint("stick", box(0.5, 0.0, (0.15, 0.005), math.pi / 4))
    cube = footprint("cube", 0.58, -0.06, half=(0.02, 0.02))
    groups = groups_by_member(build_groups([stick, cube], WORKSPACE))
    assert ("stick",) in groups and ("cube",) in groups


def test_build_groups_links_touching_objects_and_keeps_isolated_ones_alone():
    footprints = [
        footprint("bowl", 0.5, 0.0, half=(0.08, 0.08)),
        footprint("banana_in_bowl", 0.5, 0.0, half=(0.05, 0.02)),
        footprint("cube", 0.5, 0.25),
    ]
    groups = groups_by_member(build_groups(footprints, WORKSPACE))
    assert groups[("banana_in_bowl", "bowl")].moves
    assert groups[("cube",)].moves


def test_build_groups_fixed_reasons():
    footprints = [
        footprint("shelf", 0.5, 0.3, half=(0.1, 0.05), fixed_reason="static"),
        footprint("mug_on_shelf_edge", 0.5, 0.36),  # touches the shelf
        footprint("riser", 0.3, -0.2, half=(0.05, 0.05), fixed_reason="static"),
        footprint("cube_on_riser", 0.3, -0.2),
        footprint("edge_cube", 0.88, 0.0),  # pokes 1 cm out of the workspace: may move
        footprint("off_table_cube", 1.0, 0.0),  # 13 cm out: fixed
        footprint("unselected", 0.6, -0.2, fixed_reason="not selected"),
    ]
    groups = groups_by_member(build_groups(footprints, WORKSPACE))
    assert groups[("mug_on_shelf_edge", "shelf")].fixed_reason == "linked to fixed object(s) shelf"
    assert groups[("cube_on_riser", "riser")].fixed_reason == "linked to fixed object(s) riser"
    assert groups[("edge_cube",)].moves
    assert groups[("edge_cube",)].workspace.x_max == pytest.approx(0.91)
    assert groups[("off_table_cube",)].fixed_reason == "starts 13.0 cm outside the workspace"
    assert groups[("unselected",)].fixed_reason == "not selected"


def test_declared_groups_add_links_and_reject_unknown_names():
    footprints = [footprint("fork", 0.4, -0.1), footprint("plate", 0.4, 0.0, half=(0.05, 0.05)),
                  footprint("cube", 0.7, 0.2)]
    groups = groups_by_member(build_groups(footprints, WORKSPACE, [["fork", "plate"]]))
    assert ("fork", "plate") in groups
    with pytest.raises(ValueError, match="unknown scene objects"):
        build_groups(footprints, WORKSPACE, [["fork", "spoon"]])


def test_declared_group_with_a_task_fixed_member_stays_fixed():
    # Task.fixed_objects = ["bowl_1"] and object_groups = [["bowl_1", "bowl_2"]]: neither bowl moves.
    footprints = [footprint("bowl_1", 0.4, -0.2, half=(0.06, 0.06), fixed_reason="fixed by the task"),
                  footprint("bowl_2", 0.4, 0.2, half=(0.06, 0.06)), footprint("cube", 0.7, 0.0)]
    groups = build_groups(footprints, WORKSPACE, [["bowl_1", "bowl_2"]])
    by_members = groups_by_member(groups)
    assert by_members[("bowl_1", "bowl_2")].fixed_reason == "linked to fixed object(s) bowl_1"
    assert by_members[("cube",)].moves
    rng = np.random.default_rng(0)
    for _ in range(20):
        motions = sample_layout(groups, footprints, WORKSPACE, 0.2, math.pi, rng).motions
        verify_layout(groups, footprints, motions)  # the cube never lands on a bowl
        assert [motion is None for motion in motions] == [not group.moves for group in groups]


def test_sampled_layout_stays_in_workspace_and_collision_free():
    footprints = [footprint(f"obj{i}", 0.3 + 0.12 * i, 0.0, half=(0.04, 0.02)) for i in range(5)]
    footprints.append(footprint("fixed_box", 0.55, 0.2, half=(0.06, 0.06), fixed_reason="static"))
    groups = build_groups(footprints, WORKSPACE)
    rng = np.random.default_rng(0)
    by_name = {fp.name: fp for fp in footprints}
    for _ in range(50):
        motions = sample_layout(groups, footprints, WORKSPACE, 0.2, math.pi, rng).motions
        verify_layout(groups, footprints, motions)  # raises on any overlap
        for group, motion in zip(groups, motions):
            if motion is None:
                assert not group.moves
                continue
            assert np.all(np.abs(motion.center_xy - group.center_xy) <= 0.2 + 1e-9)
            for polygon in transform_group(group, [by_name[name] for name in group.members], motion):
                assert WORKSPACE.contains(polygon)


def test_group_moves_rigidly():
    footprints = [footprint("bowl", 0.5, 0.0, half=(0.08, 0.08)), footprint("block", 0.5, 0.09)]
    groups = build_groups(footprints, WORKSPACE)
    assert len(groups) == 1
    group = groups[0]
    motion = GroupMotion(center_xy=group.center_xy + np.array([0.1, 0.0]), yaw_rad=math.pi / 2)
    moved = transform_group(group, footprints, motion)
    before = np.linalg.norm(footprints[0].polygon_xy.mean(axis=0) - footprints[1].polygon_xy.mean(axis=0))
    assert np.linalg.norm(moved[0].mean(axis=0) - moved[1].mean(axis=0)) == pytest.approx(before)


def test_snug_group_falls_back_to_its_default_pose():
    # The object exactly fills the workspace: only its default pose fits, which the last try uses.
    workspace = Workspace(0.0, 0.1, 0.0, 0.1)
    footprints = [footprint("snug", 0.05, 0.05, half=(0.05, 0.05))]
    groups = build_groups(footprints, workspace)
    assert groups[0].moves
    (motion,) = sample_layout(groups, footprints, workspace, 0.05, math.pi, np.random.default_rng(0)).motions
    assert np.allclose(motion.center_xy, groups[0].center_xy)
    assert motion.yaw_rad == 0.0


def test_overhanging_group_never_moves_further_out():
    footprints = [footprint("edge_cube", 0.88, 0.0)]  # x up to 0.91, workspace x_max 0.9
    groups = build_groups(footprints, WORKSPACE)
    rng = np.random.default_rng(0)
    for _ in range(50):
        (motion,) = sample_layout(groups, footprints, WORKSPACE, 0.1, math.pi, rng).motions
        (polygon,) = transform_group(groups[0], footprints, motion)
        assert polygon[:, 0].max() <= 0.91 + 1e-9
        assert WORKSPACE.y_min <= polygon[:, 1].min() and polygon[:, 1].max() <= WORKSPACE.y_max


def test_crowded_table_falls_back_to_smaller_noise():
    footprints = [footprint("a", 0.45, 0.0, half=(0.2, 0.3)), footprint("b", 0.88 - 0.015, 0.0, half=(0.015, 0.3))]
    groups = build_groups(footprints, WORKSPACE)
    assert all(group.moves for group in groups) and len(groups) == 2
    for seed in range(20):
        layout = sample_layout(groups, footprints, WORKSPACE, 0.2, math.pi, np.random.default_rng(seed))
        assert 0.0 <= layout.noise_scale <= 1.0
        verify_layout(groups, footprints, layout.motions)


def test_blocked_group_raises_layout_error():
    # Hand-built groups (build_groups would have linked these): the moving cube's every pose,
    # its default included, overlaps the fixed box.
    footprints = [footprint("cube", 0.5, 0.0), footprint("box", 0.5, 0.0, half=(0.3, 0.3), fixed_reason="static")]
    groups = [Group(members=["cube"], center_xy=np.array([0.5, 0.0])),
              Group(members=["box"], fixed_reason="static", center_xy=np.array([0.5, 0.0]))]
    with pytest.raises(LayoutError, match="not even the default"):
        sample_layout(groups, footprints, WORKSPACE, 0.05, 0.0, np.random.default_rng(0))


def test_episode_rng_is_reproducible_per_episode():
    a = episode_rng(0, "BananaInBowlTask", 7).uniform(size=3)
    b = episode_rng(0, "BananaInBowlTask", 7).uniform(size=3)
    c = episode_rng(0, "BananaInBowlTask", 8).uniform(size=3)
    d = episode_rng(1, "BananaInBowlTask", 7).uniform(size=3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)


def test_default_yaw_range():
    assert default_yaw_range_deg(0.0) == 0.0
    assert default_yaw_range_deg(0.1) == pytest.approx(60.0)
    assert default_yaw_range_deg(0.5) == 180.0


def test_workspace_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Workspace(0.5, 0.2, 0.0, 1.0)


def test_runner_flags_build_event_and_label():
    from robolab.eval.runner import add_common_eval_args, init_pose_event_from_args, init_pose_label

    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    args = parser.parse_args(["--init-pose-noise", "0.1", "45", "--init-pose-objects", "banana",
                              "--init-pose-workspace", "0.3", "0.8", "-0.3", "0.3"])
    event = init_pose_event_from_args(args)
    params = event["randomize_init_pose"].params
    assert params == {"noise_m": 0.1, "yaw_range_deg": 45.0, "objects": ["banana"], "workspace": (0.3, 0.8, -0.3, 0.3)}
    assert init_pose_label(args) == "initpose0.1_yaw45_banana_ws0.3,0.8,-0.3,0.3"

    assert init_pose_event_from_args(parser.parse_args([])) is None
    with pytest.raises(ValueError):
        init_pose_event_from_args(parser.parse_args(["--init-pose-objects", "banana"]))
    with pytest.raises(ValueError):
        init_pose_event_from_args(parser.parse_args(["--init-pose-noise", "0.1", "45", "3"]))


def test_merged_events_keep_order_so_randomization_runs_after_reset():
    from isaaclab.managers import EventTermCfg

    from robolab.core.environments.base import BaseEventCfg
    from robolab.core.events.reset_pose import INIT_POSE_EVENT_NAME, make_init_pose_event
    from robolab.core.events.utils import merge_events_cfg

    merged = merge_events_cfg(BaseEventCfg(), make_init_pose_event(0.1))
    # The event manager runs terms in instance attribute order.
    assert list(vars(merged)) == ["reset", INIT_POSE_EVENT_NAME]
    noop = EventTermCfg(func=lambda env, env_ids: None, mode="reset")
    merged = merge_events_cfg(merged, {"a_first_alphabetically": noop})
    assert list(vars(merged)) == ["reset", INIT_POSE_EVENT_NAME, "a_first_alphabetically"]


def test_height_ranges_let_short_objects_sit_under_a_hovering_link():
    gripper = Footprint("robot/gripper", box(0.5, 0.0, (0.05, 0.05)), "robot", z_range=(0.30, 0.45))
    mug = Footprint("mug", box(0.5, 0.0, (0.04, 0.04)), z_range=(0.0, 0.10))
    rack = Footprint("rack", box(0.5, 0.0, (0.04, 0.04)), z_range=(0.0, 0.35))
    groups = groups_by_member(build_groups([gripper, mug], WORKSPACE))
    assert groups[("mug",)].moves  # not linked to the gripper above it
    groups = groups_by_member(build_groups([gripper, rack], WORKSPACE))
    assert groups[("rack", "robot/gripper")].fixed_reason == "linked to fixed object(s) robot/gripper"
