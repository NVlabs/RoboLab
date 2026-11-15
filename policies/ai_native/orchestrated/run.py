# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate Cosmos Action proposals reviewed by a persistent Astra agent."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def add_orchestrated_args(parser):
    """Simulator-independent policy flags, matching Agentic where semantics agree."""
    parser.add_argument("--model", choices=[f"gpt-6-astra-{effort}" for effort in
                        ("low", "medium", "high", "xhigh", "max")], default="gpt-6-astra-medium")
    parser.add_argument("--allow-model-requests", "--allow_model_requests", action="store_true")
    parser.add_argument("--codex-bin", "--codex_bin", "--codex", default="codex")
    parser.add_argument("--decision-timeout", "--decision_timeout", "--codex-timeout", type=float, default=120.0)
    parser.add_argument("--policy-uri", help="Required Cosmos Action websocket URL")
    parser.add_argument("--cosmos-api-key-env", default=None,
                        help="Optional environment variable for endpoints using Api-Key authentication")
    parser.add_argument("--student-action-horizon", type=int, default=15)
    parser.add_argument("--require-full-student-chunk", action="store_true")
    parser.add_argument("--max-decisions", type=int, default=3000,
                        help="Approved robot-decision cap, not LLM calls; hold until native termination at cap; 0 disables")


def parse_args(argv=None):
    import cv2  # noqa: F401 -- must precede IsaacLab
    from isaaclab.app import AppLauncher

    from robolab.eval.runner import add_common_eval_args

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_eval_args(parser)
    parser.set_defaults(seed=2040, instruction_type=["default"], video_mode="sensor")
    add_orchestrated_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not args.policy_uri:
        parser.error("--policy-uri is required")
    if not args.allow_model_requests:
        parser.error("Real model requests require --allow-model-requests")
    if args.num_envs != 1 or args.num_runs != 1 or args.num_episodes_adaptive is not None:
        parser.error("Orchestration currently requires one environment and one run per task")
    if args.enable_gt_state:
        parser.error("Orchestration does not expose simulator ground truth")
    from policies.ai_native.orchestrated.client import OrchestratedConfig
    try:
        config_from_args(args, OrchestratedConfig)
    except ValueError as exc:
        parser.error(str(exc))
    args.enable_cameras = True
    return args


def config_from_args(args, config_type=None):
    if config_type is None:
        from policies.ai_native.orchestrated.client import OrchestratedConfig
        config_type = OrchestratedConfig
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                  capture_output=True, text=True).stdout.strip() or "unknown"
    except FileNotFoundError:
        revision = "unknown"
    return config_type(policy_uri=args.policy_uri, codex=args.codex_bin,
                       model=args.model,
                       cosmos_api_key_env=args.cosmos_api_key_env,
                       student_action_horizon=args.student_action_horizon,
                       require_full_student_chunk=args.require_full_student_chunk,
                       max_decisions=args.max_decisions, timeout=args.decision_timeout,
                       seed=args.seed, robolab_revision=revision)


def make_client(args):
    from policies.ai_native.orchestrated.client import OrchestratedClient
    from robolab.constants import get_output_dir
    return OrchestratedClient(config_from_args(args), Path(get_output_dir()) / "orchestration")


def run(args):
    import cv2  # noqa: F401
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    constants = None
    try:
        from policies.ai_native.orchestrated.episode import run_orchestrated_episode
        from robolab import constants
        from robolab.eval.runner import run_evaluation
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT
        constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args.enable_subtask
        auto_register_droid_envs(task_dirs=args.task_dirs, task=args.task, cameras=WRIST_LEFT_RIGHT)
        run_evaluation(args, policy="cosmos3_astra_orchestrated", client_factory=make_client,
                       episode_runner=run_orchestrated_episode)
    except BaseException as exc:
        # IsaacSim shutdown can terminate Python before an outer handler runs.
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        print(failure["traceback"], file=sys.stderr, flush=True)
        try:
            if constants is not None:
                path = Path(constants.get_output_dir()) / "evaluation_error.json"
                path.write_text(json.dumps(failure, indent=2) + "\n")
        except OSError:
            pass
        raise
    finally:
        app.close()


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
