# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evaluation driver: argparse helpers + per-task run loop.

Each per-policy script under ``policies/<policy>/run.py``:

1. Builds an :class:`argparse.ArgumentParser` with backend-specific flags.
2. Calls :func:`add_common_eval_args` to inject the shared flags.
3. Launches IsaacSim and registers envs (policy-specific camera presets etc.).
4. Defines a ``client_factory(args) -> InferenceClient`` closure.
5. Calls :func:`run_evaluation` with the policy name and the factory.

This module stays policy-agnostic — no backend names appear in it.

Import order constraint: :func:`add_common_eval_args` must be callable
*before* ``AppLauncher`` launches (so the parser knows the flags at parse
time). The heavy isaaclab/episode/summarize imports are therefore deferred
into :func:`run_evaluation`, which only runs *after* AppLauncher has set up
the simulation app.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Callable

from robolab.constants import DEFAULT_TASK_SUBFOLDERS

if TYPE_CHECKING:
    from robolab.eval.base_client import InferenceClient

    ClientFactory = Callable[[argparse.Namespace], InferenceClient]
else:
    ClientFactory = Callable[[argparse.Namespace], object]


def _unit_interval(s: str) -> float:
    v = float(s)
    if not (0.0 < v <= 1.0):
        raise argparse.ArgumentTypeError(f"--ci-pp-width must be in (0, 1], got {v}")
    return v


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared eval flags. Call this once per runner script."""
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1,
                        help="Number of environments to spawn.")
    parser.add_argument("--task", nargs="+", default=None,
                        help="List of tasks to evaluate on.")
    parser.add_argument("--tag", nargs="+", default=None,
                        help="List of tags of tasks to evaluate on.")
    parser.add_argument("--task-dirs", "--task_dirs", nargs="+", default=DEFAULT_TASK_SUBFOLDERS,
                        help="List of task directories to evaluate on.")
    parser.add_argument("--num-runs", "--num_runs", type=int, default=1,
                        help=("Number of sequential runs per task (default: 1). "
                              "Total episodes = num_runs * num_envs. "
                              "Prefer increasing --num-envs for more episodes. "
                              "Only increase --num-runs if you run out of GPU memory "
                              "with the desired num-envs."))
    parser.add_argument("--num-episodes-adaptive", "--num_episodes_adaptive",
                        type=int, default=None, metavar="MAX_N",
                        help=("Enable adaptive sampling per task. Overrides --num-runs. "
                              "Keeps adding batches of num_envs episodes until the 95%% "
                              "Beta credible interval on success rate is <= --ci-pp-width "
                              "wide, or until MAX_N episodes are reached. Recommended "
                              "MAX_N=200 (matches TRI LBM sim protocol, arXiv:2507.05331)."))
    parser.add_argument("--ci-pp-width", "--ci_pp_width", type=_unit_interval,
                        default=0.14, metavar="W",
                        help=("Target 95%% Beta credible interval width (as a fraction in "
                              "(0, 1]) for adaptive sampling. Default 0.14 = worst-case CI "
                              "width at n=200 (TRI LBM sim protocol, arXiv:2507.05331). "
                              "Only used when --num-episodes-adaptive is set."))
    parser.add_argument("--enable-subtask", "--enable_subtask", dest="enable_subtask",
                        action="store_true", default=True,
                        help="Enable subtask progress checking (default: on; kept for "
                             "backward compatibility).")
    parser.add_argument("--disable-subtask", "--disable_subtask", dest="enable_subtask",
                        action="store_false",
                        help="Disable subtask progress checking (episode results will "
                             "have no score/reason and an empty events log).")
    parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                        help=("Output folder name under <repo>/output. Default is "
                              "<timestamp>_<policy>. If you provide the output folder name "
                              "for a previous run, the script will skip the tasks and "
                              "episodes that have already been run."))
    parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default",
                        help=("Which instruction variant to use when a task defines multiple "
                              "(default, vague, specific, etc.)."))
    parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                        choices=["all", "viewport", "sensor", "none"],
                        help=("Which videos to save: 'all' (sensor + viewport), "
                              "'viewport' only, 'sensor' only, or 'none' (default: all)."))
    parser.add_argument("--renderer", type=str, default="realtime",
                        choices=["realtime", "pathtracing"],
                        help=("RTX renderer mode (default: realtime). 'realtime' uses "
                              "the RaytracedLighting interactive renderer; 'pathtracing' "
                              "uses the offline path tracer (higher fidelity but much "
                              "slower — intended for beauty/demo renders, not large-N "
                              "eval). Sets carb '/rtx/rendermode'."))
    # NOTE: named --rendering-type (not --rendering-mode) on purpose. Isaac Lab's
    # AppLauncher.add_app_launcher_args() reserves the dest `rendering_mode` on some
    # versions (e.g. 2.2.0 in the OSMO eval image) and raises if the parser already
    # defines it. Using a distinct dest avoids that collision regardless of arg order.
    parser.add_argument("--rendering-type", "--rendering_type", type=str, default=None,
                        choices=["performance", "balanced", "quality"],
                        help=("Realtime renderer quality preset (maps to IsaacLab "
                              "RenderCfg.rendering_mode). Default: unset, which lets "
                              "IsaacLab fall back to 'balanced'. No effect under "
                              "--renderer pathtracing."))


def run_evaluation(
    args: argparse.Namespace,
    *,
    policy: str,
    client_factory: ClientFactory,
) -> None:
    """Drive the per-task evaluation loop.

    Must be called *after* ``AppLauncher`` has launched, since this is when
    the isaaclab-dependent modules below become safe to import.

    Args:
        args: Parsed argparse namespace. Must include the flags from
            :func:`add_common_eval_args` plus ``device`` and ``headless`` from
            ``AppLauncher.add_app_launcher_args``.
        policy: Backend label stamped into the output folder name and
            ``env_cfg.policy``. No behavioral branching.
        client_factory: Callable that builds the :class:`InferenceClient` given
            ``args``. Called once per task; the client is reused across runs
            within a task.
    """
    import os

    import robolab.constants
    from robolab.constants import PACKAGE_DIR, get_timestamp, set_output_dir
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.logging.results import (
        check_all_episodes_complete,
        check_run_complete,
        init_experiment,
        summarize_experiment_results,
    )
    from robolab.core.utils.adaptive_sampling import count_task_episodes, should_continue_sampling
    from robolab.core.utils.print_utils import print_experiment_summary
    from robolab.eval.episode import run_episode
    from robolab.eval.summarize import summarize_run

    if args.output_folder_name is not None:
        output_folder_name = args.output_folder_name
    else:
        output_folder_name = get_timestamp() + f"_{policy}"
        if args.instruction_type != "default":
            output_folder_name += f"_{args.instruction_type}"

    output_dir = os.path.join(PACKAGE_DIR, "output", output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.task:
        task_envs = get_envs(task=args.task)
        filter_str = f"tasks: {', '.join(args.task)}"
    elif getattr(args, "tag", None):
        task_envs = get_envs(tag=args.tag)
        filter_str = f"tags: {', '.join(args.tag)}"
    else:
        task_envs = get_envs()
        filter_str = "all"

    num_envs = args.num_envs
    num_runs = args.num_runs
    adaptive_max = args.num_episodes_adaptive
    is_adaptive = adaptive_max is not None
    total_episodes = adaptive_max if is_adaptive else num_runs * num_envs

    print_experiment_summary(
        task_envs=task_envs,
        filter_str=filter_str,
        num_envs=num_envs,
        num_episodes=total_episodes,
        policy=policy,
        instruction_type=args.instruction_type,
        output_dir=output_dir,
    )

    episode_results_file, episode_results = init_experiment(output_dir)

    save_videos = args.video_mode != "none"

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        if check_all_episodes_complete(
            episode_results=episode_results, env_name=task_env, num_episodes=total_episodes
        ):
            print(f"\033[96m[RoboLab] Task `{task_env}` already done. Skipping.\033[0m")
            continue

        env, env_cfg = create_env(
            task_env,
            device=args.device,
            num_envs=num_envs,
            instruction_type=args.instruction_type,
            policy=policy,
            renderer=args.renderer,
            rendering_mode=args.rendering_type,
        )

        if robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING and getattr(env_cfg, "subtasks", None) is None:
            print(
                f"\033[93m[RoboLab] WARNING: Subtask tracking is enabled but task `{task_env}` "
                f"has no subtask specification — subtask tracking is skipped for this task, so "
                f"its episode results will have no score/reason and an empty events log.\033[0m"
            )

        client = client_factory(args)

        run_idx = 0
        while True:
            if is_adaptive:
                k_so_far, n_so_far = count_task_episodes(episode_results, task_env)
                if not should_continue_sampling(
                    k=k_so_far, n=n_so_far, target_width=args.ci_pp_width, n_max=adaptive_max
                ):
                    print(
                        f"\033[96m[RoboLab] Task `{task_env}` adaptive stop at {n_so_far} "
                        f"episodes ({k_so_far}/{n_so_far} success).\033[0m"
                    )
                    break
            else:
                if run_idx >= num_runs:
                    break

            run_episode_ids = [run_idx * num_envs + eid for eid in range(num_envs)]
            if all(
                check_run_complete(episode_results=episode_results, env_name=task_env, episode=ep_id)
                for ep_id in run_episode_ids
            ):
                print(f"\033[96m[RoboLab] Task `{task_env}` run `{run_idx}` already done. Skipping.\033[0m")
                run_idx += 1
                continue

            if args.instruction_type != "default":
                run_name = f"{task_env}_{args.instruction_type}_{run_idx}"
            else:
                run_name = f"{task_env}_{run_idx}"
            print(
                f"\033[96m[RoboLab] Running {run_name}: '{env_cfg.instruction}' "
                f"(run {run_idx}, {num_envs} envs)\033[0m"
            )

            env_results, msgs, timing = run_episode(
                env=env,
                env_cfg=env_cfg,
                episode=run_idx,
                client=client,
                save_videos=save_videos,
                video_mode=args.video_mode,
                headless=args.headless,
            )

            episode_results = summarize_run(
                env_results=env_results,
                msgs=msgs,
                env=env,
                env_cfg=env_cfg,
                num_envs=num_envs,
                run_idx=run_idx,
                run_name=run_name,
                task_env=task_env,
                scene_output_dir=scene_output_dir,
                policy=policy,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
                enable_subtask_progress=robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING,
                timing=timing,
                instruction_type=args.instruction_type,
            )

            env.reset_eval_state()
            run_idx += 1

        env.close()

    summarize_experiment_results(episode_results, show_timing=True)
