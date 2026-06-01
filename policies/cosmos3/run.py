# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the Cosmos3 policy backend across registered tasks."""

import argparse
import sys
import traceback

import cv2  # Must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "cosmos3"

parser = argparse.ArgumentParser(description="Evaluate the Cosmos3 policy backend.")
parser.add_argument(
    "--remote-host", default="localhost", help="Remote host for policy server (default: localhost)."
)
parser.add_argument(
    "--remote-port", default=8000, type=int, help="Remote port for policy server (default: 8000)."
)

from robolab.eval.runner import add_common_eval_args, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.cosmos3.client import Cosmos3Client
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

auto_register_droid_envs(task=args_cli.task, cameras=WRIST_LEFT_RIGHT_HEAD)


def make_client(args: argparse.Namespace) -> Cosmos3Client:
    """ """
    return Cosmos3Client(remote_host=args.remote_host, remote_port=args.remote_port)


def main() -> None:
    """ """
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
