# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import inspect
import os
from typing import Optional


def get_caller_info(frame_depth: int = 1) -> str:
    """
    Get the current filename and function name as a string.

    Args:
        frame_depth: How many frames to go back (default 1 for immediate caller)

    Returns:
        String in format "filename/function_name" or "unknown" if frame cannot be determined
    """
    frame = inspect.currentframe()
    if frame is None:
        return "unknown"

    # Go up the specified number of frames to get the caller's info
    caller_frame = frame
    for _ in range(frame_depth):
        caller_frame = caller_frame.f_back
        if caller_frame is None:
            return "unknown"

    filename = os.path.basename(caller_frame.f_code.co_filename)
    funcname = caller_frame.f_code.co_name
    return f"{filename}/{funcname}"


def get_log_prefix(frame_depth: int = 1) -> str:
    """
    Get a log prefix with caller information.

    Args:
        frame_depth: How many frames to go back (default 1 for immediate caller)

    Returns:
        String in format "[filename/function_name]"
    """
    return f"[{get_caller_info(frame_depth)}]"


def log_with_caller(message: str, frame_depth: int = 1) -> str:
    """
    Create a log message with caller information prefix.

    Args:
        message: The log message
        frame_depth: How many frames to go back (default 1 for immediate caller)

    Returns:
        Formatted log message with caller prefix
    """
    return f"{get_log_prefix(frame_depth)} {message}"