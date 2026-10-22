# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialize and briefly step RoboLab tasks without a policy server.

This is intentionally stronger than config/import validation: every selected
environment is constructed, reset, and stepped so first-step predicates,
Fabric synchronization, contacts, sensors, and termination terms are exercised.
Results are appended to a JSONL report, making a long preflight resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
import torch
from isaaclab.app import AppLauncher

from robolab.constants import DEFAULT_TASK_SUBFOLDERS

parser = argparse.ArgumentParser(
    description="Construct, reset, and briefly step RoboLab tasks without a policy server."
)
parser.add_argument("--task", nargs="+", default=None, help="Task class names to test (default: all).")
parser.add_argument(
    "--task-dirs", "--task_dirs", nargs="+", default=DEFAULT_TASK_SUBFOLDERS,
    help="Task directories to discover (default: benchmark).",
)
parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=2, help="Simulation steps per task (default: 2).")
parser.add_argument("--resets", type=int, default=2, help="Resets before stepping (default: 2, matching eval).")
parser.add_argument(
    "--action-mode", choices=("hold", "random"), default="random",
    help="Hold the reset pose or add small random joint-position perturbations (default: random).",
)
parser.add_argument(
    "--random-action-scale", type=float, default=0.02, metavar="RADIANS",
    help="Uniform arm-joint perturbation around the current pose (default: 0.02 rad).",
)
parser.add_argument("--preflight-seed", type=int, default=0)
parser.add_argument(
    "--skip-completed-from", type=Path, default=None, metavar="RESULTS_DIR_OR_JSONL",
    help="Skip envs having at least --required-episodes entries in an evaluation results file.",
)
parser.add_argument(
    "--required-episodes", type=int, default=1,
    help="Completed episode rows required before an env is skipped (default: 1).",
)
parser.add_argument(
    "--report", type=Path, default=None,
    help="Append-only JSONL report. Passing an existing report resumes its successful tasks.",
)
parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed task.")
parser.add_argument("--enable-verbose", action="store_true")
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.logging.results import load_episode_results  # noqa: E402
from robolab.core.utils.isaaclab_compat import as_torch  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)


def _load_completed_counts(path: Path | None) -> Counter:
    if path is None:
        return Counter()
    results_dir = path if path.is_dir() else path.parent
    if path.is_file() and path.name not in {"episode_results.jsonl", "episode_results.json"}:
        raise ValueError(f"Unsupported results file: {path}")
    return Counter(row.get("env_name") for row in load_episode_results(str(results_dir)) if row.get("env_name"))


def _load_passed_report_envs(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    passed = set()
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "passed" and row.get("env_name"):
                passed.add(row["env_name"])
    return passed


def _make_actions(env, *, randomize: bool, scale: float, generator: torch.Generator) -> torch.Tensor:
    """Make safe joint-position actions centered on the current reset pose."""
    action_dim = env.action_manager.total_action_dim
    actions = torch.zeros((env.num_envs, action_dim), dtype=torch.float32, device=env.device)

    robot = env.scene.articulations["robot"]
    arm_ids = [index for index, name in enumerate(robot.data.joint_names) if name.startswith("panda_joint")]
    arm_pos = as_torch(robot.data.joint_pos)[:, arm_ids]
    num_arm_actions = min(arm_pos.shape[1], action_dim)
    actions[:, :num_arm_actions] = arm_pos[:, :num_arm_actions]

    if randomize and num_arm_actions:
        noise = torch.empty(
            (env.num_envs, num_arm_actions), dtype=torch.float32, device=env.device
        ).uniform_(-scale, scale, generator=generator)
        actions[:, :num_arm_actions] += noise
    return actions


def _append_report(report_path: Path, row: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def run_preflight() -> int:
    if args_cli.steps < 1 or args_cli.resets < 1 or args_cli.num_envs < 1:
        parser.error("--steps, --resets, and --num-envs must all be positive")
    if args_cli.random_action_scale < 0:
        parser.error("--random-action-scale must be non-negative")
    if args_cli.required_episodes < 1:
        parser.error("--required-episodes must be positive")

    robolab.constants.VERBOSE = args_cli.enable_verbose
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True

    auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)
    task_envs = get_envs(task=args_cli.task) if args_cli.task else get_envs()

    completed_counts = _load_completed_counts(args_cli.skip_completed_from)
    report_path = args_cli.report
    if report_path is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = Path(robolab.constants.PACKAGE_DIR) / "output" / f"{stamp}_preflight.jsonl"
    report_path = report_path.expanduser().resolve()
    already_passed = _load_passed_report_envs(report_path)

    selected = [
        env_name for env_name in task_envs
        if completed_counts[env_name] < args_cli.required_episodes and env_name not in already_passed
    ]
    print(
        f"[RoboLab preflight] selected {len(selected)}/{len(task_envs)} envs; "
        f"report={report_path}",
        flush=True,
    )

    failures = 0
    for task_index, env_name in enumerate(selected):
        started = time.monotonic()
        env = None
        try:
            env, _ = create_env(
                env_name,
                device=args_cli.device,
                seed=args_cli.preflight_seed,
                num_envs=args_cli.num_envs,
                use_fabric=True,
            )
            for _ in range(args_cli.resets):
                observations, _ = env.reset()
                if observations is None:
                    raise RuntimeError("env.reset() returned no observations")

            generator = torch.Generator(device=env.device)
            generator.manual_seed(args_cli.preflight_seed + task_index)
            for _ in range(args_cli.steps):
                actions = _make_actions(
                    env,
                    randomize=args_cli.action_mode == "random",
                    scale=args_cli.random_action_scale,
                    generator=generator,
                )
                observations, _, terminated, truncated, _ = env.step(actions)
                if observations is None or terminated is None or truncated is None:
                    raise RuntimeError("env.step() returned an incomplete result")

            duration = time.monotonic() - started
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "env_name": env_name,
                "status": "passed",
                "duration_s": round(duration, 3),
                "steps": args_cli.steps,
                "resets": args_cli.resets,
                "num_envs": args_cli.num_envs,
                "action_mode": args_cli.action_mode,
                "seed": args_cli.preflight_seed,
            }
            print(f"[PASS] {env_name} ({duration:.1f}s)", flush=True)
        except Exception as exc:
            failures += 1
            duration = time.monotonic() - started
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "env_name": env_name,
                "status": "failed",
                "duration_s": round(duration, 3),
                "steps": args_cli.steps,
                "resets": args_cli.resets,
                "num_envs": args_cli.num_envs,
                "action_mode": args_cli.action_mode,
                "seed": args_cli.preflight_seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"[FAIL] {env_name}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    traceback.print_exc()

        _append_report(report_path, row)
        if failures and args_cli.stop_on_error:
            break

    print(
        f"[RoboLab preflight] complete: {len(selected) - failures} passed, "
        f"{failures} failed; report={report_path}",
        flush=True,
    )
    return 1 if failures else 0


def main() -> None:
    exit_code = 1
    try:
        exit_code = run_preflight()
    finally:
        try:
            simulation_app.close()
        except SystemExit:
            pass
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
