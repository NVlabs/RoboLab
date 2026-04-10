# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import cv2


def add_text_overlay(image, text, position=(10, 30), font_scale=0.8, color=(255, 255, 255), thickness=2):
    """
    Add text overlay to an image using OpenCV.

    Args:
        image: numpy array image (H, W, C) in RGB format
        text: string to overlay
        position: (x, y) position for text
        font_scale: font size scale
        color: RGB color tuple (default: white)
        thickness: text thickness

    Returns:
        Image with text overlay
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Add text
    cv2.putText(image_bgr, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

    # Convert back to RGB
    image_with_text = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_with_text

def add_multiline_text_overlay(image, text_lines, start_position=(10, 30), font_scale=0.8,
                              color=(255, 255, 255), thickness=2, line_spacing=35):
    """
    Add multiline text overlay to an image using OpenCV.

    Args:
        image: numpy array image (H, W, C) in RGB format
        text_lines: list of strings to overlay
        start_position: (x, y) position for first line
        font_scale: font size scale
        color: RGB color tuple (default: white)
        thickness: text thickness
        line_spacing: vertical spacing between lines

    Returns:
        Image with text overlay
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Add each line of text
    for i, line in enumerate(text_lines):
        y_position = start_position[1] + i * line_spacing
        position = (start_position[0], y_position)
        cv2.putText(image_bgr, line, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    # Convert back to RGB
    image_with_text = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_with_text

def add_multiline_text_overlay_with_background(image, text_lines, start_position=(10, 30),
                                             font_scale=0.8, color=(255, 255, 255), thickness=2,
                                             line_spacing=35, background_color=(0, 0, 0),
                                             background_alpha=0.7, padding=10):
    """
    Add multiline text overlay to an image with a semi-transparent background for better visibility.

    Args:
        image: numpy array image (H, W, C) in RGB format
        text_lines: list of strings to overlay
        start_position: (x, y) position for first line
        font_scale: font size scale
        color: RGB color tuple (default: white)
        thickness: text thickness
        line_spacing: vertical spacing between lines
        background_color: RGB color for background rectangle
        background_alpha: transparency of background (0.0 to 1.0)
        padding: padding around text for background rectangle

    Returns:
        Image with text overlay and background
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Calculate text dimensions to size the background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_width = 0
    total_height = len(text_lines) * line_spacing

    for line in text_lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_width)

    # Calculate background rectangle coordinates
    bg_x1 = start_position[0] - padding
    bg_y1 = start_position[1] - padding
    bg_x2 = start_position[0] + max_width + padding
    bg_y2 = start_position[1] + total_height + padding

    # Create background overlay
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), background_color, -1)

    # Blend background with original image
    image_with_bg = cv2.addWeighted(overlay, background_alpha, image_bgr, 1 - background_alpha, 0)

    # Add text on top of the background
    for i, line in enumerate(text_lines):
        y_position = start_position[1] + i * line_spacing
        position = (start_position[0], y_position)
        cv2.putText(image_with_bg, line, position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Convert back to RGB
    image_with_text = cv2.cvtColor(image_with_bg, cv2.COLOR_BGR2RGB)

    return image_with_text
