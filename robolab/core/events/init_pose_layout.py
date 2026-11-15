# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry for initial-pose randomization: object groups and collision-free layouts.

Pure numpy so it can be unit-tested without IsaacSim. The event term in
``reset_pose.py`` gathers footprints from the stage and writes the sampled
poses back to the simulation.

Conventions: all coordinates are env-local, in meters. A footprint is an
object's convex outline seen from above (a convex polygon in the xy plane) at
its default pose, plus the height range it occupies; a group's yaw (radians,
about +z) rotates its members' outlines about the group's center. Two
footprints can only collide if their height ranges come within CLEARANCE_M.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

# Objects closer than this at their default poses are grouped; after sampling,
# every pair of footprints that can collide keeps at least this gap.
CLEARANCE_M = 0.01
# A group that already sticks out of the workspace by at most this much at its default pose
# may still move, but never further out than it started; groups sticking out more stay fixed.
EDGE_ALLOWANCE_M = 0.03
# Noise at which the coupled yaw range reaches a full turn.
FULL_YAW_AT_NOISE_M = 0.3
MAX_TRIES_PER_GROUP = 100
MAX_LAYOUT_RESTARTS = 10


class LayoutError(RuntimeError):
    """No collision-free layout was found inside the workspace."""


@dataclass(frozen=True)
class Workspace:
    """Rectangle in the env-local xy plane, meters. Every moved footprint stays inside it."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self):
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError(f"Workspace bounds must satisfy min < max, got {self}")

    def contains(self, points_xy: np.ndarray) -> bool:
        """True if every point (..., 2) lies inside the rectangle."""
        x, y = points_xy[..., 0], points_xy[..., 1]
        return bool(np.all((x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max)))


@dataclass(frozen=True)
class Footprint:
    """One scene object at its default pose, seen from above.

    Attributes:
        name: Scene object name.
        polygon_xy: Convex outline, (k, 2) meters, counter-clockwise.
        fixed_reason: Why the object cannot move (e.g. "static", "not
            selected"); empty if it may move. Fixed objects are obstacles.
        z_range: Lowest and highest point, meters. Unbounded by default.
    """

    name: str
    polygon_xy: np.ndarray
    fixed_reason: str = ""
    z_range: tuple[float, float] = (-math.inf, math.inf)

    @property
    def movable(self) -> bool:
        return not self.fixed_reason


@dataclass
class Group:
    """Objects that keep their relative poses. ``fixed_reason`` is empty when the group moves.

    ``center_xy`` is the center of the members' combined bounding rectangle; the group turns
    about it. ``workspace`` is the group's own limit: the workspace, widened on any side where
    the group already sticks out at its default pose (by at most EDGE_ALLOWANCE_M).
    """

    members: list[str]
    fixed_reason: str = ""
    center_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    workspace: Workspace | None = None

    @property
    def moves(self) -> bool:
        return not self.fixed_reason


@dataclass(frozen=True)
class GroupMotion:
    """Rigid xy motion of one group: rotate by ``yaw_rad`` about the group's default center, then move the center to ``center_xy``."""

    center_xy: np.ndarray
    yaw_rad: float


@dataclass(frozen=True)
class Layout:
    """One sampled layout: a motion per group (None for fixed groups), and the fraction of the
    requested noise and yaw ranges it was sampled with (1.0 unless crowding forced less)."""

    motions: list[GroupMotion | None]
    noise_scale: float


def default_yaw_range_deg(noise_m: float) -> float:
    """Yaw half-range coupled to the position noise: a full turn at FULL_YAW_AT_NOISE_M."""
    return 180.0 * min(1.0, noise_m / FULL_YAW_AT_NOISE_M)


def episode_rng(seed: int, task_name: str, episode_index: int) -> np.random.Generator:
    """Random generator for one episode, independent of how episodes are split across envs and runs."""
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(task_name.encode()), episode_index]))


def convex_hull(points_xy: np.ndarray, min_size: float = 1e-3) -> np.ndarray:
    """Counter-clockwise convex hull of (n, 2) points (monotone chain).

    Degenerate inputs (a point or a line) become a rectangle at least ``min_size`` wide.
    """
    points = np.unique(np.asarray(points_xy, dtype=float).reshape(-1, 2), axis=0)
    if len(points) >= 3:
        def half_hull(sequence):
            hull = []
            for p in sequence:
                while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                          - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) <= 0:
                    hull.pop()
                hull.append(p)
            return hull
        lower, upper = half_hull(points), half_hull(points[::-1])
        hull = np.array(lower[:-1] + upper[:-1])
        if len(hull) >= 3:
            return hull
    lo, hi = points.min(axis=0) - min_size / 2, points.max(axis=0) + min_size / 2
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def polygons_overlap(a: np.ndarray, b: np.ndarray, clearance: float = 0.0) -> bool:
    """True if two convex polygons overlap or are closer than ``clearance`` (separating-axis test).

    The clearance test is conservative: polygons reported as separated are at least
    ``clearance`` apart.
    """
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        proj_a, proj_b = a @ normals.T, b @ normals.T
        gap = np.maximum(proj_b.min(axis=0) - proj_a.max(axis=0), proj_a.min(axis=0) - proj_b.max(axis=0))
        if np.any(gap >= clearance):
            return False
    return True


def _bounding_circle(polygon: np.ndarray) -> tuple[np.ndarray, float]:
    center = (polygon.min(axis=0) + polygon.max(axis=0)) / 2.0
    return center, float(np.max(np.linalg.norm(polygon - center, axis=1)))


Z_UNBOUNDED = (-math.inf, math.inf)


class _Placed:
    """Footprints already in the layout, with height and bounding-circle pre-checks before the polygon test."""

    def __init__(self, entries: list[tuple[np.ndarray, tuple[float, float]]] = ()):
        self.polygons: list[np.ndarray] = []
        self.centers: list[np.ndarray] = []
        self.radii: list[float] = []
        self.z_low: list[float] = []
        self.z_high: list[float] = []
        for polygon, z_range in entries:
            self.add(polygon, z_range)

    def add(self, polygon: np.ndarray, z_range: tuple[float, float] = Z_UNBOUNDED) -> None:
        center, radius = _bounding_circle(polygon)
        self.polygons.append(polygon)
        self.centers.append(center)
        self.radii.append(radius)
        self.z_low.append(z_range[0])
        self.z_high.append(z_range[1])

    def collides(self, polygon: np.ndarray, z_range: tuple[float, float] = Z_UNBOUNDED) -> bool:
        if not self.polygons:
            return False
        center, radius = _bounding_circle(polygon)
        distance = np.linalg.norm(np.stack(self.centers) - center, axis=1)
        same_height = ((np.array(self.z_low) < z_range[1] + CLEARANCE_M)
                       & (np.array(self.z_high) > z_range[0] - CLEARANCE_M))
        near = np.nonzero(same_height & (distance < np.array(self.radii) + radius + CLEARANCE_M))[0]
        return any(polygons_overlap(polygon, self.polygons[i], CLEARANCE_M) for i in near)


def build_groups(
    footprints: list[Footprint],
    workspace: Workspace,
    declared_groups: list[list[str]] | None = None,
) -> list[Group]:
    """Group objects that must move together and decide which groups move.

    Two objects are linked if their default footprints are closer than
    CLEARANCE_M, or if ``declared_groups`` lists them together. Groups are the
    connected sets of linked objects, so an object on a riser, a shelf or a
    stack is in the same group as what it rests on. A group stays fixed if it
    contains a fixed object or starts more than EDGE_ALLOWANCE_M outside the
    workspace.

    Raises:
        ValueError: if ``declared_groups`` names an unknown object.
    """
    names = [fp.name for fp in footprints]
    index = {name: i for i, name in enumerate(names)}
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def link(i, j):
        parent[find(i)] = find(j)

    for i in range(len(footprints)):
        placed = _Placed([(footprints[i].polygon_xy, footprints[i].z_range)])
        for j in range(i + 1, len(footprints)):
            if placed.collides(footprints[j].polygon_xy, footprints[j].z_range):
                link(i, j)

    for declared in declared_groups or []:
        unknown = [name for name in declared if name not in index]
        if unknown:
            raise ValueError(f"object_groups names unknown scene objects {unknown}; scene objects: {names}")
        for name in declared[1:]:
            link(index[declared[0]], index[name])

    components: dict[int, list[int]] = {}
    for i in range(len(names)):
        components.setdefault(find(i), []).append(i)

    groups = []
    for member_ids in components.values():
        members = [footprints[i] for i in member_ids]
        vertices = np.concatenate([fp.polygon_xy for fp in members])
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        group = Group(members=[fp.name for fp in members], center_xy=(lo + hi) / 2.0)
        fixed = [fp for fp in members if not fp.movable]
        overhang = max(workspace.x_min - lo[0], hi[0] - workspace.x_max, workspace.y_min - lo[1], hi[1] - workspace.y_max)
        if fixed and len(fixed) == len(members):
            group.fixed_reason = ", ".join(sorted({fp.fixed_reason for fp in fixed}))
        elif fixed:
            group.fixed_reason = "linked to fixed object(s) " + ", ".join(fp.name for fp in fixed)
        elif overhang > EDGE_ALLOWANCE_M:
            group.fixed_reason = f"starts {overhang * 100:.1f} cm outside the workspace"
        else:
            group.workspace = Workspace(min(workspace.x_min, lo[0]), max(workspace.x_max, hi[0]),
                                        min(workspace.y_min, lo[1]), max(workspace.y_max, hi[1]))
        groups.append(group)
    return groups


def transform_group(group: Group, members: list[Footprint], motion: GroupMotion | None) -> list[np.ndarray]:
    """Member outlines after applying ``motion`` (unchanged if None)."""
    if motion is None:
        return [fp.polygon_xy for fp in members]
    cos, sin = math.cos(motion.yaw_rad), math.sin(motion.yaw_rad)
    rotation = np.array([[cos, -sin], [sin, cos]])
    return [(fp.polygon_xy - group.center_xy) @ rotation.T + motion.center_xy for fp in members]


def sample_layout(
    groups: list[Group],
    footprints: list[Footprint],
    workspace: Workspace,
    noise_m: float,
    yaw_range_rad: float,
    rng: np.random.Generator,
) -> Layout:
    """Sample one collision-free layout.

    Moving groups are placed largest first (in a random order on restarts). Each
    candidate is checked against every fixed object and every member of the
    groups already placed, and must stay inside the group's workspace (see
    :class:`Group`). The first half of a group's MAX_TRIES_PER_GROUP tries sample
    the full noise and yaw ranges; the second half shrink both toward the default
    pose (the last try is the default itself). If a group still cannot be placed,
    the whole layout is restarted in a random order with smaller ranges: restart r
    uses 1 - r / (MAX_LAYOUT_RESTARTS - 1) of them, so the last restart is the
    default layout, which is always legal for groups built by :func:`build_groups`.
    ``Layout.noise_scale`` reports the fraction used.

    Raises:
        LayoutError: if even the default layout is blocked (only possible for groups
            not built by :func:`build_groups`).
    """
    by_name = {fp.name: fp for fp in footprints}
    fixed = [(by_name[name].polygon_xy, by_name[name].z_range)
             for group in groups if not group.moves for name in group.members]

    def area(group_index):
        vertices = np.concatenate([by_name[name].polygon_xy for name in groups[group_index].members])
        return float(np.prod(vertices.max(axis=0) - vertices.min(axis=0)))

    moving = sorted((i for i, g in enumerate(groups) if g.moves), key=lambda i: -area(i))

    for restart in range(MAX_LAYOUT_RESTARTS):
        noise_scale = 1.0 - restart / (MAX_LAYOUT_RESTARTS - 1)
        order = moving if restart == 0 else list(rng.permutation(moving))
        placed = _Placed(fixed)
        motions: list[GroupMotion | None] = [None] * len(groups)
        for group_index in order:
            group = groups[group_index]
            members = [by_name[name] for name in group.members]
            limit = group.workspace or workspace
            low = np.array([limit.x_min, limit.y_min])
            high = np.array([limit.x_max, limit.y_max])
            for attempt in range(MAX_TRIES_PER_GROUP):
                scale = noise_scale * min(1.0, 2.0 * (MAX_TRIES_PER_GROUP - 1 - attempt) / (MAX_TRIES_PER_GROUP - 1))
                sample_low = np.maximum(group.center_xy - scale * noise_m, low)
                sample_high = np.minimum(group.center_xy + scale * noise_m, high)
                motion = GroupMotion(center_xy=rng.uniform(sample_low, sample_high),
                                     yaw_rad=float(rng.uniform(-scale * yaw_range_rad, scale * yaw_range_rad)))
                polygons = transform_group(group, members, motion)
                if all(limit.contains(p) for p in polygons) and not any(
                        placed.collides(p, fp.z_range) for p, fp in zip(polygons, members)):
                    break
            else:
                break
            motions[group_index] = motion
            for polygon, fp in zip(polygons, members):
                placed.add(polygon, fp.z_range)
        else:
            verify_layout(groups, footprints, motions)
            return Layout(motions=motions, noise_scale=noise_scale)
    names = [groups[i].members for i in moving]
    raise LayoutError(
        f"No collision-free layout, not even the default one (noise {noise_m} m, moving groups {names}). "
        f"A moving group overlaps a fixed object at its default pose."
    )


def verify_layout(groups: list[Group], footprints: list[Footprint], motions: list[GroupMotion | None]) -> None:
    """Check every pair of footprints that involves a moved group, at final poses.

    Pairs inside the same group, and pairs of two fixed objects, are skipped.

    Raises:
        LayoutError: if any such pair is closer than CLEARANCE_M.
    """
    by_name = {fp.name: fp for fp in footprints}
    entries = []  # (name, polygon, z range, group index, moved)
    for group_index, (group, motion) in enumerate(zip(groups, motions)):
        members = [by_name[name] for name in group.members]
        polygons = transform_group(group, members, motion)
        entries.extend((fp.name, polygon, fp.z_range, group_index, motion is not None)
                       for fp, polygon in zip(members, polygons))
    bad = []
    for i, (name_a, poly_a, z_a, group_a, moved_a) in enumerate(entries):
        for name_b, poly_b, z_b, group_b, moved_b in entries[i + 1:]:
            if group_a != group_b and (moved_a or moved_b) and _Placed([(poly_b, z_b)]).collides(poly_a, z_a):
                bad.append((name_a, name_b))
    if bad:
        raise LayoutError(f"Sampled layout has overlapping footprints: {bad}")
