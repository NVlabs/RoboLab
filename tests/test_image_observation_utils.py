# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording preserves camera pixels when sensor resolutions differ."""

import numpy as np
import pytest
import torch

from robolab.core.observations.observation_utils import unpack_image_obs


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_mixed_resolution_montage_preserves_independent_camera_images(scale):
    obs = {"image_obs": {
        "head_cam": torch.full((2, 12, 16, 3), 70, dtype=torch.uint8),
        "wrist_cam": torch.full((2, 8, 12, 3), 170, dtype=torch.uint8),
    }}
    obs["image_obs"]["head_cam"][0] = 0
    obs["image_obs"]["wrist_cam"][0] = 0
    images = unpack_image_obs(obs, env_id=1, scale=scale)
    h, w, sh, sw = (int(value * scale) for value in (12, 16, 8, 12))
    assert images["head_cam"].shape == (h, w, 3)
    assert images["wrist_cam"].shape == (sh, sw, 3)
    assert images["combined_image"].shape == (h, w + sw, 3)
    np.testing.assert_array_equal(images["combined_image"][:, :w], images["head_cam"])
    np.testing.assert_array_equal(images["combined_image"][:sh, w:], images["wrist_cam"])
    assert np.all(images["combined_image"][sh:, w:] == 0)
    assert np.all(images["head_cam"] == 70)
    assert np.all(images["wrist_cam"] == 170)
    assert obs["image_obs"]["wrist_cam"].shape == (2, 8, 12, 3)


def test_equal_resolution_montage_keeps_historical_pixels():
    obs = {"image_obs": {
        "a_camera": torch.full((1, 8, 12, 3), 10, dtype=torch.uint8),
        "b_camera": torch.full((1, 8, 12, 3), 20, dtype=torch.uint8),
    }}
    images = unpack_image_obs(obs)
    expected = np.concatenate((np.full((8, 12, 3), 10, dtype=np.uint8),
                               np.full((8, 12, 3), 20, dtype=np.uint8)), axis=1)
    np.testing.assert_array_equal(images["combined_image"], expected)
