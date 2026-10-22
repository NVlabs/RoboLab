# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for scene overrides affected by asset hierarchy changes."""

import os

from pxr import Sdf, Usd

from robolab.constants import SCENE_DIR
from robolab.core.scenes.utils import scrape_scene


def test_imported_scene_root_authors_wxyz_identity():
    """The compatibility boundary must receive a RoboLab-authored rotation.

    Isaac Lab 3's AssetBaseCfg default is already XYZW.  Allowing that default
    into prepare_env_cfg() double-converts it and rotates every task scene 180
    degrees about Z while resettable rigid objects return to authored positions.
    """
    scene_path = os.path.join(SCENE_DIR, "shelf_with_cleaning_products.usda")
    scene = scrape_scene(scene_path, ["table"])["scene_dict"]["scene"]
    assert scene.init_state.rot == (1.0, 0.0, 0.0, 0.0)


def test_wire_shelf_scene_has_no_unresolved_overrides():
    """Every override in the wire-shelf scene must resolve to a defined prim.

    Isaac Sim 6 rejects contact-filter paths backed only by stale ``over``
    specs.  The plate and fork assets were flattened during the port, so old
    nested override paths must not be reintroduced into this scene.
    """
    scene_path = os.path.join(SCENE_DIR, "wire_shelf_mugs_plate_spatula.usda")
    stage = Usd.Stage.Open(scene_path)
    assert stage is not None, f"failed to open {scene_path}"

    unresolved = [
        str(prim.GetPath())
        for prim in stage.TraverseAll()
        if prim.GetSpecifier() == Sdf.SpecifierOver and not prim.IsDefined()
    ]
    assert not unresolved, f"scene contains unresolved override prims: {unresolved}"
