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
parser.add_argument(
    "--embodiment", choices=["droid", "galbot"], default="droid", help="Robot embodiment to evaluate (default: droid)."
)
parser.add_argument(
    "--galbot-init-pose",
    default="default",
    help="Galbot only: key of GALBOT_GOLF_INIT_POSES for the initial joint pose (default: default).",
)
parser.add_argument(
    "--galbot-gain-profile",
    default="default",
    help="Galbot only: key of GALBOT_GOLF_ARM_GAIN_PROFILES for the arm PD gains (default: default).",
)
parser.add_argument(
    "--galbot-sim-hz",
    type=float,
    help="Galbot only: physics rate in Hz (dt = 1 / hz); default keeps the registration default (60.0).",
)
parser.add_argument(
    "--galbot-decimation",
    type=int,
    help="Galbot only: physics steps per control step; also sets render interval (default: registration default).",
)

from robolab.eval.runner import add_common_eval_args, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.cosmos3.client import Cosmos3Client, Cosmos3GalbotClient

if args_cli.embodiment == "droid":
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

    auto_register_droid_envs(task=args_cli.task, cameras=WRIST_LEFT_RIGHT_HEAD)
else:
    from robolab.registrations.galbot.auto_env_registrations_jointpos import auto_register_galbot_envs
    from robolab.robots.galbot_golf import (
        GalbotGolfLeftEgoCameraCfg,
        GalbotGolfLeftWristCameraCfg,
        GalbotGolfRightWristCameraCfg,
        make_galbot_golf_tabletop_cfg,
    )

    galbot_sim_kwargs = {}
    if args_cli.galbot_sim_hz is not None:
        galbot_sim_kwargs["dt"] = 1.0 / args_cli.galbot_sim_hz
    if args_cli.galbot_decimation is not None:
        galbot_sim_kwargs["decimation"] = args_cli.galbot_decimation
        galbot_sim_kwargs["render_interval"] = args_cli.galbot_decimation

    auto_register_galbot_envs(
        task=args_cli.task,
        cameras=[GalbotGolfLeftEgoCameraCfg, GalbotGolfLeftWristCameraCfg, GalbotGolfRightWristCameraCfg],
        robot_cfg=make_galbot_golf_tabletop_cfg(
            init_pose=args_cli.galbot_init_pose,
            gain_profile=args_cli.galbot_gain_profile,
        ),
        **galbot_sim_kwargs,
    )


def make_client(args: argparse.Namespace) -> Cosmos3Client:
    """ """
    client_cls = Cosmos3Client if args.embodiment == "droid" else Cosmos3GalbotClient
    return client_cls(remote_host=args.remote_host, remote_port=args.remote_port)


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
