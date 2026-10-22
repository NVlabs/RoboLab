# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify isaaclab is installed and importable.

Replaces the legacy scripts/check_isaaclab.py. The acceptable version is
pinned by pyproject.toml — no need to re-assert it here.
"""

from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest


def test_isaaclab_installed():
    try:
        version("isaaclab")
    except PackageNotFoundError as e:
        raise AssertionError("isaaclab is not installed") from e

    import isaaclab  # noqa: F401


def test_isaaclab3_usd_frame_view_accepts_float3_scales():
    """Regression test for Isaac Lab 3's float3-to-double3 scale crash."""
    from robolab.core.utils.isaaclab_compat import ISAACLAB_USES_XYZW

    if not ISAACLAB_USES_XYZW:
        pytest.skip("UsdFrameView is an Isaac Lab 3 compatibility path")

    # Importing WorldState installs RoboLab's precision-safe reader.
    from isaaclab.sim.views import UsdFrameView
    from pxr import Gf

    from robolab.core.world import world_state  # noqa: F401

    class _Attribute:
        def Get(self):
            return Gf.Vec3f(0.5, 1.0, 0.8)

    class _Prim:
        def GetAttribute(self, name):
            assert name == "xformOp:scale"
            return _Attribute()

    class _View:
        _device = "cpu"
        _prims = [_Prim()]

        @staticmethod
        def _resolve_indices(indices):
            assert indices is None
            return [0]

    scales = UsdFrameView.get_scales(_View()).numpy()
    np.testing.assert_allclose(scales, [[0.5, 1.0, 0.8]])
