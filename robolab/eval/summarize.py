# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Per-run summarization: turns the raw outputs of :func:`run_episode` into
persisted per-env log files, HDF5-derived trajectory metrics, and aggregated
``run_summary`` dicts folded into the experiment's ``episode_results``.

Lives in ``robolab.eval`` because it's tightly coupled to the eval loop: it
consumes exactly what ``run_episode`` emits and writes results to the same
``scene_output_dir`` layout that ``run_eval.py`` and its variants share.
"""

import os
import re
from collections import Counter

from robolab.core.logging.results import (
    dump_results_to_file,
    extract_subtask_status_changes,
    get_final_subtask_info,
    update_experiment_results,
)
from robolab.core.metrics import compute_episode_metrics, load_demo_data
from robolab.core.task.status import EVENT_STATUS_CODES, StatusCode, get_status_name
from robolab.core.utils.file_utils import load_file


def split_msgs_per_env(
    msgs: list[list[dict] | None], num_envs: int
) -> dict[int, list[dict | None]]:
    """Transpose step-major msgs (``list[list[dict] | None]``) into per-env
    log streams (``{env_id: list[dict | None]}``)."""
    per_env: dict[int, list[dict | None]] = {eid: [] for eid in range(num_envs)}
    for step_infos in msgs:
        if step_infos is None:
            for eid in range(num_envs):
                per_env[eid].append(None)
        else:
            for eid in range(num_envs):
                per_env[eid].append(step_infos[eid] if eid < len(step_infos) else None)
    return per_env


def extract_events_from_log(log_file: str) -> dict:
    """Read a per-env JSON log and tally occurrences of each status code in
    :data:`EVENT_STATUS_CODES`. For ``WRONG_OBJECT_GRABBED_FAILURE``, also
    collects the name of each wrong object.
    """
    if not os.path.exists(log_file):
        return {}

    log_data = load_file(log_file)
    if log_data is None:
        return {}

    status_changes = extract_subtask_status_changes(log_data)
    if not status_changes:
        return {}

    event_counts: Counter = Counter()
    wrong_objects_grabbed: list[str] = []

    for change in status_changes:
        status_code = change.get("status", 0)
        if status_code not in EVENT_STATUS_CODES:
            continue

        event_name = get_status_name(status_code)
        if event_name.endswith("_FAILURE"):
            event_name = event_name[:-8]

        event_counts[event_name] += 1

        if status_code == StatusCode.WRONG_OBJECT_GRABBED_FAILURE:
            info = change.get("info", "")
            match = re.search(r"Wrong object grabbed: '([^']+)'", info)
            if match:
                wrong_objects_grabbed.append(match.group(1))

    events: dict = dict(event_counts)
    if wrong_objects_grabbed:
        events["wrong_objects_grabbed"] = wrong_objects_grabbed

    return events


def build_run_summary(
    *,
    env_result: dict,
    env_id: int,
    run_idx: int,
    num_envs: int,
    run_name: str,
    task_env: str,
    env_cfg,
    policy: str,
    dt: float,
    traj_metrics: dict | None,
    events: dict,
    env_msgs: list[dict | None],
    final_info: dict | None,
    enable_subtask_progress: bool,
    instruction_type: str | None = None,
    timing: dict | None = None,
    task_name: str | None = None,
    extra_fields: dict | None = None,
) -> dict:
    """Construct one per-env ``run_summary`` dict. Pure: no IO.

    ``instruction_type`` and ``timing`` are included only when provided
    (not-None). ``task_name`` overrides ``env_cfg._task_name`` when set.
    ``extra_fields`` merges arbitrary caller-specific keys at the end
    (used by variant-eval scripts to attach ``background``/``lighting``/etc.).
    """
    episode_id = run_idx * num_envs + env_id

    summary: dict = {
        "env_name": task_env,
        "task_name": task_name if task_name is not None else env_cfg._task_name,
        "run_name": run_name,
        "run": run_idx,
        "episode": episode_id,
        "env_id": env_id,
        "policy": policy,
        "instruction": env_cfg.instruction,
        "attributes": env_cfg._task_attributes,
        "success": env_result["success"],
        "episode_step": env_result["step"],
        "duration": env_result["step"] * dt if env_result["step"] else 0,
        "dt": dt,
        "metrics": traj_metrics or {},
        "events": events or {},
    }
    if instruction_type is not None:
        summary["instruction_type"] = instruction_type
    if timing is not None:
        summary["timing"] = timing

    if enable_subtask_progress:
        last_msg = next((m for m in reversed(env_msgs) if m is not None), None)
        if last_msg is not None:
            summary["score"] = last_msg.get("score", None)
            summary["reason"] = last_msg.get("info", None)
        else:
            summary["score"] = None
            summary["reason"] = None

        # For failed episodes, prefer the per-env final_info reason if available.
        if not env_result["success"] and final_info is not None:
            summary["reason"] = final_info.get("info", summary.get("reason"))

    if extra_fields:
        summary.update(extra_fields)

    return summary


def summarize_run(
    *,
    env_results: list[dict],
    msgs: list[list[dict] | None],
    env,
    env_cfg,
    num_envs: int,
    run_idx: int,
    run_name: str,
    task_env: str,
    scene_output_dir: str,
    policy: str,
    episode_results: dict,
    episode_results_file: str,
    enable_subtask_progress: bool = False,
    timing: dict | None = None,
    instruction_type: str | None = None,
    task_name: str | None = None,
    extra_fields: dict | None = None,
) -> dict:
    """Fold the outputs of :func:`run_episode` into ``episode_results``.

    For each env: writes its per-step log to ``log_{run_idx}_env{eid}.json``,
    extracts event counts, loads its trajectory from the run's HDF5, computes
    episode metrics, builds a ``run_summary``, and updates the aggregate.

    Optional ``timing`` / ``instruction_type`` are included in each summary
    only when provided. ``task_name`` overrides ``env_cfg._task_name``.
    ``extra_fields`` is merged into each summary (for variant-eval scripts
    that attach ``background``/``lighting``/etc.).

    Returns the updated ``episode_results`` dict.
    """
    per_env_msgs = split_msgs_per_env(msgs, num_envs)
    final_infos = get_final_subtask_info(env, env_id=None)

    # Write per-env logs + tally events.
    per_env_events: dict[int, dict] = {}
    for eid in range(num_envs):
        log_file = os.path.join(scene_output_dir, f"log_{run_idx}_env{eid}.json")
        dump_results_to_file(log_file, per_env_msgs[eid], append=False)
        per_env_events[eid] = extract_events_from_log(log_file)

    dt = env_cfg.sim.dt * env_cfg.decimation
    hdf5_path = os.path.join(scene_output_dir, f"run_{run_idx}.hdf5")

    for r in env_results:
        env_id = r["env_id"]
        traj_data = load_demo_data(hdf5_path, f"demo_{env_id}")
        traj_metrics = compute_episode_metrics(traj_data, dt=dt) if traj_data else None
        final_info = final_infos[env_id] if final_infos else None

        run_summary = build_run_summary(
            env_result=r,
            env_id=env_id,
            run_idx=run_idx,
            num_envs=num_envs,
            run_name=run_name,
            task_env=task_env,
            env_cfg=env_cfg,
            policy=policy,
            dt=dt,
            traj_metrics=traj_metrics,
            events=per_env_events.get(env_id, {}),
            env_msgs=per_env_msgs.get(env_id, []),
            final_info=final_info,
            enable_subtask_progress=enable_subtask_progress,
            instruction_type=instruction_type,
            timing=timing,
            task_name=task_name,
            extra_fields=extra_fields,
        )

        episode_results = update_experiment_results(
            run_summary=run_summary,
            episode_results=episode_results,
            episode_results_file=episode_results_file,
        )

    return episode_results
