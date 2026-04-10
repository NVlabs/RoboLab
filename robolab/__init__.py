# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
RoboLab - A robotics laboratory framework for simulation and experimentation.
"""

__version__ = "0.1.0"
__author__ = "RoboLab Team"

# Import constants module to make it available at package level
from . import constants

# Import main modules to make them available at package level
# Use try-except to handle missing dependencies gracefully
try:
    from . import core
except ImportError:
    core = None

try:
    from . import robots
except ImportError:
    robots = None

try:
    from . import observations
except ImportError:
    observations = None

try:
    from . import tasks
except ImportError:
    tasks = None

try:
    from . import inference
except ImportError:
    inference = None

__all__ = [
    "constants",
    "core",
    "robots",
    "observations",
    "tasks",
    "inference",
]