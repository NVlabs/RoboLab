# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the DreamZero policy backend across registered tasks."""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "dreamzero"

parser = argparse.ArgumentParser(description="Evaluate the DreamZero policy backend.")
parser.add_argument("--remote-host", default="localhost",
                    help="Remote host for policy server (default: localhost).")
parser.add_argument("--remote-port", default=5000, type=int,
                    help="Remote port for policy server (default: 5000).")
parser.add_argument("--remote-uri", default=None,
                    help=("Full WebSocket URI for policy server, e.g. wss://host.lepton.run. "
                          "Overrides --remote-host and --remote-port when set."))
parser.add_argument("--remote-token", default=None,
                    help=("Bearer token for authenticated endpoints (e.g. Lepton). "
                          "Falls back to DREAMZERO_API_TOKEN env var."))
parser.add_argument("--open-loop-horizon", type=int,
                    help=("Number of actions to execute from each predicted chunk before "
                          "requesting a new one. If omitted, the client uses its own default. "
                          "Must match the model's action_horizon for best performance."))
parser.add_argument("--dz-binarize-gripper", action="store_true",
                    help="Re-enable gripper binarization at 0.5 threshold (ablation; default: off).")
parser.add_argument("--dz-resize", default="area",
                    choices=["area", "linear", "pad"],
                    help=("Image resize method: 'area' (default, INTER_AREA), 'linear' "
                          "(INTER_LINEAR), or 'pad' (aspect-preserving letterbox). Note: "
                          "'area'/'linear' change aspect ratio if source differs from 180x320 target."))
parser.add_argument("--cam2-source", default="black",
                    choices=["black", "right", "head", "duplicate"],
                    help=("Second exterior camera: 'black' (default, matches training dropout), "
                          "'right' (over-shoulder), 'head' (front overhead), 'duplicate' (copy of left)."))

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs as _register,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402

from policies.dreamzero.client import DreamZeroClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask

# 'right' and 'head' cam2 slots need the extra cameras attached at registration time;
# 'black' and 'duplicate' work with the default WRIST_LEFT preset.
if args_cli.cam2_source in ("right", "head"):
    _register(task_dirs=args_cli.task_dirs, task=args_cli.task, cameras=WRIST_LEFT_RIGHT_HEAD)
else:
    _register(task_dirs=args_cli.task_dirs, task=args_cli.task)


def make_client(args: argparse.Namespace) -> DreamZeroClient:
    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        api_token=args.remote_token,
        open_loop_horizon=args.open_loop_horizon,
        binarize_gripper=args.dz_binarize_gripper,
        resize=args.dz_resize,
        cam2_source=args.cam2_source,
    )
    return DreamZeroClient(**{k: v for k, v in kwargs.items() if v is not None})


def main() -> None:
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
