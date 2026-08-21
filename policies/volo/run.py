# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a policy backend through a VoLo inference proxy.

Selects an underlying backend with ``--policy`` (cosmos3 or a Pi0-family
variant) and registers environments with depth cameras enabled. Requests are
a strict superset of the backend's normal wire format, so the proxy can also
point at an unmodified policy server.
"""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

PI0_VARIANTS = ["pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"]
POLICY_CHOICES = ["cosmos3", *PI0_VARIANTS]

parser = argparse.ArgumentParser(description="Evaluate a policy backend through a VoLo inference proxy.")
parser.add_argument("--policy", choices=POLICY_CHOICES, default="cosmos3",
                    help="Backend policy behind the proxy (default: cosmos3).")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                    help="Remote host for the proxy/policy server (default: localhost).")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                    help="Remote port for the proxy/policy server (default: 8000).")
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None,
                    help=("Full WebSocket URI for the proxy/policy server, e.g. wss://host.lepton.run. "
                          "Overrides --remote-host and --remote-port when set."))
parser.add_argument("--remote-token", "--remote_token", type=str, default=None,
                    help=("Bearer token for an authenticated Cosmos3 proxy/policy server. "
                          "Falls back to COSMOS3_API_TOKEN. Cosmos3 backend only."))
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                    help=("Number of actions to execute from each predicted chunk before requesting a "
                          "new one. If omitted, the backend client uses its default. "
                          "Pi0-family backends only."))

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.volo.registration import register_volo_envs  # noqa: E402
from robolab.constants import DEFAULT_TASK_SUBFOLDERS  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT, WRIST_LEFT_RIGHT_HEAD  # noqa: E402

# Camera preset must match what the backend policy was trained on.
cameras = WRIST_LEFT_RIGHT_HEAD if args_cli.policy == "cosmos3" else WRIST_LEFT

# If --task-dirs was not given, argparse hands back the DEFAULT_TASK_SUBFOLDERS
# object itself; pass None so registration uses the VoLo task folders
# (benchmark + robovolo) instead.
task_dirs = None if args_cli.task_dirs is DEFAULT_TASK_SUBFOLDERS else args_cli.task_dirs

register_volo_envs(
    task_dirs=task_dirs,
    task=args_cli.task,
    cameras=cameras,
)


def make_client(args: argparse.Namespace):
    if args.policy == "cosmos3":
        from policies.volo.client import VoloCosmos3Client

        return VoloCosmos3Client(
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            remote_uri=args.remote_uri,
            api_token=args.remote_token,
        )

    from policies.volo.client import VoloPi0Client

    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        open_loop_horizon=args.open_loop_horizon,
        policy_variant=args.policy,
    )
    return VoloPi0Client(**{k: v for k, v in kwargs.items() if v is not None})


def main() -> None:
    run_evaluation(args_cli, policy=f"volo_{args_cli.policy}", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
