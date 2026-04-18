# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Image preprocessing utilities for policy clients.

Vendored from openpi_client.image_tools (original at
``openpi/packages/openpi-client/src/openpi_client/image_tools.py``) so that
individual backends inside :mod:`robolab_policy_client` can share the same
resize behavior without forcing ``openpi_client`` as a runtime dep for
non-openpi backends. Kept semantically identical to the upstream.
"""

import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float image to uint8.

    Reduces the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(
    images: np.ndarray, height: int, width: int, method=Image.BILINEAR
) -> np.ndarray:
    """Replicates ``tf.image.resize_with_pad`` for a batch of images using PIL.

    Resizes while preserving aspect ratio and pads the remainder with zeros.

    Args:
        images: Batch of images in ``[..., height, width, channel]`` format.
        height: Target height.
        width: Target width.
        method: PIL interpolation method. Default is bilinear.

    Returns:
        Resized images in ``[..., height, width, channel]``.
    """
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Resize one PIL image to target size without distortion, padding with zeros.

    Note: PIL uses ``(width, height, channel)`` ordering.
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
