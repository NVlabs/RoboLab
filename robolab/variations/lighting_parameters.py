# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parameters shared by the built-in lighting selections and standalone previews."""

PRESETS = {
    "sphere_default": {"dome": 500, "sphere": 5000, "areas": []},
    "ceiling_soft": {"dome": 350, "areas": [
        {"name": "ceiling", "pos": [0.55, 0, 1.9], "target": [0.55, 0, 0],
         "size": [1.5, 1.5], "intensity": 4000, "kelvin": 5000}]},
    "ceiling_cool": {"dome": 150, "areas": [
        {"name": "ceiling", "pos": [0.55, 0, 1.9], "target": [0.55, 0, 0],
         "size": [1.5, 1.5], "intensity": 1800, "kelvin": 7000}]},
    "window_neutral": {"dome": 350, "areas": [
        {"name": "window", "pos": [0.55, 1.4, 1.35], "target": [0.55, 0, 0],
         "size": [1.2, 1.5], "intensity": 5500, "kelvin": 5600},
        {"name": "ceiling", "pos": [0.55, -0.2, 2.0], "target": [0.55, 0, 0],
         "size": [1.5, 1.5], "intensity": 900, "kelvin": 4500}]},
    "window_warm": {"dome": 350, "areas": [
        {"name": "window", "pos": [0.55, 1.4, 1.35], "target": [0.55, 0, 0],
         "size": [1.2, 1.5], "intensity": 5500, "kelvin": 4500},
        {"name": "ceiling", "pos": [0.55, -0.2, 2.0], "target": [0.55, 0, 0],
         "size": [1.5, 1.5], "intensity": 900, "kelvin": 4500}]},
    "dark": {"dome": 5, "sphere": 50, "areas": []},
    "harsh_shadows": {"dome": 20, "areas": [], "directional": {
        "direction": [-0.6, -0.8, -1], "intensity": 3000, "angle": 0.53, "exposure": 0}},
    "spotlight": {"dome": 200, "areas": [], "directional": {
        "direction": [0, 0, -1], "intensity": 3000, "angle": 0.53, "exposure": 0}},
    "soft_shadows": {"dome": 100, "areas": [
        {"name": "soft", "pos": [0.55, 1.2, 1.1], "target": [0.55, 0, 0],
         "size": [0.3, 0.3], "intensity": 250000, "kelvin": 5600}]},
}
