# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="function to check whether the env registration worked correctly.")
parser.add_argument("--task", nargs='+', default=None,
                       help="List of tasks")
parser.add_argument("--tag", nargs='+', default=None,
                       help="List of tags of tasks")
AppLauncher.add_app_launcher_args(parser)
args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Robolab components
from robolab.core.logging.recorder_manager import patch_recorder_manager
patch_recorder_manager()
from robolab.core.environments.factory import get_global_env_factory, get_envs, print_env_table

#############################
#  Call scene registration  #
#############################

# This is to check whether the env registration worked correctly.
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs
auto_register_droid_envs()

#############################
#  Print registered envs  #
#############################
factory = get_global_env_factory()
if args_cli.task:
    task_envs = get_envs(task=args_cli.task)
elif args_cli.tag:
    task_envs = get_envs(tag=args_cli.tag)
else:
    task_envs = get_envs()

# Print all registered envs
print("-"*60)
print(f"All registered envs:")

print_env_table()

# Print args

if args_cli.task or args_cli.tag:
    print("-"*60)
    print(f"tasks via args: --task {args_cli.task} --tag {args_cli.tag}")
    print("-"*60)
    print(f"Found {len(task_envs)} envs:")
    print(task_envs)
    print("-"*60)
    print()
