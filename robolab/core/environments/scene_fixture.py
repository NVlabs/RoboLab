# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-robot table fixtures for task scenes.

Task scene USDs still author a legacy table-fixture prim (``franka_table``);
the env factory always deactivates it and instead spawns the fixture each robot
declares, so the fixture is owned by the robot, not the scene. Robot cfg
classes declare their needs via two labels, assigned *after* the class
definition and after any subclasses (``configclass`` converts every member it
sees at decoration time — class-body, ``ClassVar``-annotated, or inherited —
into a config field, and ``InteractiveScene`` rejects scene-cfg fields it does
not recognize):

- ``table_fixture``: a :class:`TableFixtureCfg` naming the fixture USD and its
  pose, or ``None`` for robots with their own base. Robots without the label
  default to :data:`FRANKA_TABLE_FIXTURE`.
- ``root_z_above_ground`` (float meters, default ``None``): rebases the robot
  root's init z to the scene's authored ``/GroundPlane`` height plus this
  offset — the reach of the robot's lowest colliders below its root, so the
  robot stands exactly on whatever floor the scene authors. Ground heights are
  per-scene (canonical -0.697, legacy -0.65; locked by
  ``tests/test_scene_ground.py``).
"""

import os
from dataclasses import dataclass
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from pxr import Usd, UsdGeom

from robolab.constants import ASSET_DIR

TABLE_FIXTURE_PRIM = "franka_table"


@dataclass(frozen=True)
class TableFixtureCfg:
    """Fixture USD a robot is mounted on, and its pose.

    The pose is expressed in the frame selected by ``frame``: ``"origin"`` for
    the env origin, ``"robot"`` for the robot root's init pose (the fixture
    then follows e.g. a ground-rebased root). Translation in meters, rotation
    as a ``(w, x, y, z)`` quaternion.
    """

    usd_path: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    frame: Literal["origin", "robot"] = "origin"


FRANKA_TABLE_FIXTURE = TableFixtureCfg(
    usd_path=os.path.join(ASSET_DIR, "fixtures", "franka_table.usd"),
    pos=(-0.087, 0.0, 0.0),
    # 180 degrees about +Z, matching the pose authored in every task scene.
    rot=(6.123233995736766e-17, 0.0, 0.0, 1.0),
)


def _quat_mul(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = q
    qv = _quat_mul(_quat_mul(q, (0.0, *v)), (w, -x, -y, -z))
    return qv[1:]


def table_fixture_asset(fixture: TableFixtureCfg | None, robot_cfg: type) -> AssetBaseCfg | None:
    """Resolve a robot's table-fixture spec into a spawnable scene asset."""
    if fixture is None:
        return None
    pos, rot = fixture.pos, fixture.rot
    if fixture.frame == "robot":
        robot_state = robot_cfg().robot.init_state
        pos = tuple(p + d for p, d in zip(robot_state.pos, _quat_rotate(robot_state.rot, fixture.pos)))
        rot = _quat_mul(robot_state.rot, fixture.rot)
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table_fixture",
        spawn=sim_utils.UsdFileCfg(usd_path=fixture.usd_path),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
    )


@clone
def spawn_scene_without_table_fixture(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a task scene, then deactivate its legacy table-fixture payload."""
    scene_prim = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    fixture = sim_utils.get_current_stage().GetPrimAtPath(f"{prim_path}/{TABLE_FIXTURE_PRIM}")
    if fixture.IsValid():
        # This removes the payload from rendering and physics; it is not a
        # visibility-only override.
        fixture.SetActive(False)
    return scene_prim


def scene_ground_height(scene_asset) -> float | None:
    """Return the scene's authored ``/GroundPlane`` z, or ``None`` if not authored."""
    stage = Usd.Stage.Open(scene_asset.spawn.usd_path)
    if stage is None:
        raise FileNotFoundError(f"Unable to open task scene: {scene_asset.spawn.usd_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        return None
    ground = stage.GetPrimAtPath(default_prim.GetPath().AppendChild("GroundPlane"))
    if not ground.IsValid():
        return None
    transform = UsdGeom.Xformable(ground).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return float(transform.ExtractTranslation()[2])


def scene_without_table_fixture(task_scene: type) -> tuple[type, float | None]:
    """Return a task-scene override without the legacy fixture, plus its ground height."""
    task_scene_cfg = task_scene()
    if not hasattr(task_scene_cfg, "scene"):
        return task_scene, None

    scene_asset = task_scene_cfg.scene.copy()
    ground_z = scene_ground_height(scene_asset)
    scene_asset.spawn = scene_asset.spawn.copy()
    scene_asset.spawn.func = spawn_scene_without_table_fixture

    scene_override = type(f"{task_scene.__name__}NoTableFixture", (task_scene,), {})
    scene_override.scene = scene_asset
    return scene_override, ground_z


def robot_cfg_above_ground(robot_cfg: type, ground_z: float | None, offset: float) -> type:
    """Return a robot cfg subclass with its root placed ``offset`` above the scene's ground."""
    if ground_z is None:
        raise ValueError(
            f"{robot_cfg.__name__} sets root_z_above_ground, but the task scene does not author /GroundPlane."
        )
    robot = robot_cfg().robot.copy()
    robot.init_state = robot.init_state.copy()
    robot.init_state.pos = (*robot.init_state.pos[:2], ground_z + offset)
    class_name = f"{robot_cfg.__name__}Ground{abs(round((ground_z + offset) * 1000))}mm"
    return configclass(type(class_name, (robot_cfg,), {"robot": robot}))
