# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="function to check whether the env registration worked correctly.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli,
    renderer="PathTracing",
    carb_settings={
            "/rtx/post/dlss/enabled": False,
            "/rtx/post/denoiser/enabled": False,
            "/isaaclab/render/rtx_sensors": False,
            "/rtx/enabled": False,
            "/rtx/pathtracer/enabled": False,
        }
    )
simulation_app = app_launcher.app
simulation_app.close()