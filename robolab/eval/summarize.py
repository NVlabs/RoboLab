# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

import h5py

from robolab.core.logging.results import (
    dump_results_to_file,
    extract_subtask_status_changes,
    get_all_env_events,
    get_final_subtask_info,
    update_experiment_results,
)
from robolab.core.metrics import compute_episode_metrics, load_demo_data
from robolab.core.task.status import EVENT_STATUS_CODES, StatusCode, get_status_name
from robolab.core.utils.file_utils import load_file


def _read_final_score_from_hdf5(hdf5_path: str, env_id: int) -> float | None:
    """Return the canonical final-step SM score for an env from
    ``demo_<env_id>/subtask/score``. Returns ``None`` if the file or dataset
    is missing or empty.

    This is the source of truth for ``episode_results.score`` and matches
    the per-event ``score`` field in the v2 events log (both denormalize
    from ``subtask/score[step]``).
    """
    if not os.path.exists(hdf5_path):
        return None
    try:
        with h5py.File(hdf5_path, "r") as f:
            ds = f.get(f"data/demo_{env_id}/subtask/score")
            if ds is None or ds.shape[0] == 0:
                return None
            return float(ds[-1])
    except Exception:
        return None


def _tally_events(events: list[dict]) -> dict:
    """Tally a v2 events list (each entry has ``code``, ``info``, ...) and
    return ``{event_name: count, ..., wrong_objects_grabbed: [...]}``.

    Pure function: works on in-memory event dicts, used by both ``summarize_run``
    (post-episode) and ``extract_events_from_log`` (file-based, v1 or v2).
    """
    event_counts: Counter = Counter()
    wrong_objects_grabbed: list[str] = []

    for change in events:
        status_code = change.get("code", change.get("status", 0))
        if status_code not in EVENT_STATUS_CODES:
            continue

        event_name = change.get("name") or get_status_name(status_code)
        if event_name.endswith("_FAILURE"):
            event_name = event_name[:-8]

        event_counts[event_name] += 1

        if status_code == StatusCode.WRONG_OBJECT_GRABBED_FAILURE:
            info = change.get("info", "")
            match = re.search(r"Wrong object grabbed: '([^']+)'", info)
            if match:
                wrong_objects_grabbed.append(match.group(1))

    out: dict = dict(event_counts)
    if wrong_objects_grabbed:
        out["wrong_objects_grabbed"] = wrong_objects_grabbed
    return out


def extract_events_from_log(log_file: str) -> dict:
    """Read a per-env JSON log (v1 or v2) and tally occurrences of each status
    code in :data:`EVENT_STATUS_CODES`. For ``WRONG_OBJECT_GRABBED_FAILURE``,
    also collects the name of each wrong object.

    v1 logs are dense per-step lists; v2 logs are sparse event arrays under a
    ``schema_version: 2`` envelope.
    """
    if not os.path.exists(log_file):
        return {}

    log_data = load_file(log_file)
    if log_data is None:
        return {}

    if isinstance(log_data, dict) and log_data.get("schema_version") == 2:
        return _tally_events(log_data.get("events", []))

    if isinstance(log_data, list):
        # v1: dense per-step list, expand all_status_codes via legacy helper
        status_changes = extract_subtask_status_changes(log_data)
        # Map v1 'status' field to v2 'code' for the shared tally helper
        v2_shaped = [
            {"code": c.get("status", 0), "info": c.get("info", "")}
            for c in status_changes
        ]
        return _tally_events(v2_shaped)

    return {}


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
    events_list: list[dict],
    final_info: dict | None,
    enable_subtask_progress: bool,
    instruction_type: str | None = None,
    timing: dict | None = None,
    task_name: str | None = None,
    extra_fields: dict | None = None,
    final_score: float | None = None,
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
        # Score: HDF5 subtask/score[final_step] is canonical (matches
        # per-event score in the v2 events log).
        if final_score is not None:
            summary["score"] = final_score
        elif events_list:
            summary["score"] = events_list[-1].get("score")
        else:
            summary["score"] = None

        # Reason: derived from the v2 events stream. For success, walk back
        # past post-success drift (failure-range codes that fire after the
        # success transition) to the last success-range event.
        if events_list:
            last_event = events_list[-1]
            if env_result["success"]:
                success_event = next(
                    (e for e in reversed(events_list)
                     if e.get("info") and e.get("code", 0) < 200),
                    last_event,
                )
                summary["reason"] = success_event.get("info")
            else:
                summary["reason"] = last_event.get("info")
        else:
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
    # ``msgs`` is retained in the signature for backward compatibility with
    # existing eval callers but is no longer consumed: events come from the
    # recorder and score/reason derive from HDF5 + the v2 events stream.
    final_infos = get_final_subtask_info(env, env_id=None)

    dt = env_cfg.sim.dt * env_cfg.decimation
    hdf5_path = os.path.join(scene_output_dir, f"run_{run_idx}.hdf5")

    # v2 event log: pull per-env events from the recorder and persist as
    # {schema_version: 2, ..., events: [...]}. Tally counts in memory; no
    # disk round-trip needed.
    per_env_events_list = get_all_env_events(env) or [[] for _ in range(num_envs)]
    env_results_by_id = {r["env_id"]: r for r in env_results}
    resolved_task_name = task_name if task_name is not None else env_cfg._task_name

    per_env_events: dict[int, dict] = {}
    for eid in range(num_envs):
        events = per_env_events_list[eid] if eid < len(per_env_events_list) else []
        r = env_results_by_id.get(eid, {})
        log_obj = {
            "schema_version": 2,
            "dt": dt,
            "task": resolved_task_name,
            "env_id": eid,
            "run": run_idx,
            "success": r.get("success"),
            "final_step": r.get("step"),
            "events": events,
        }
        log_file = os.path.join(scene_output_dir, f"log_{run_idx}_env{eid}.json")
        dump_results_to_file(log_file, log_obj, append=False)
        per_env_events[eid] = _tally_events(events)

    for r in env_results:
        env_id = r["env_id"]
        traj_data = load_demo_data(hdf5_path, f"demo_{env_id}")
        traj_metrics = compute_episode_metrics(traj_data, dt=dt) if traj_data else None
        final_info = final_infos[env_id] if final_infos else None
        final_score = _read_final_score_from_hdf5(hdf5_path, env_id)

        events_list_for_env = (
            per_env_events_list[env_id] if env_id < len(per_env_events_list) else []
        )
        per_env_extra = dict(extra_fields or {})
        # Reset-time sequential drop is not part of the policy trajectory, but
        # when its frames were prepended to an MP4 the dashboard needs the
        # video offset to align policy events and plots with that MP4.
        setup_sim = getattr(env, "_dynamic_setup_sim_duration_s", {})
        setup_video = getattr(env, "_dynamic_setup_video_duration_s", {})
        if isinstance(setup_sim, dict) and env_id in setup_sim:
            per_env_extra["setup_sim_duration_s"] = float(setup_sim[env_id])
        if isinstance(setup_video, dict) and env_id in setup_video:
            per_env_extra["setup_video_duration_s"] = float(setup_video[env_id])

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
            events_list=events_list_for_env,
            final_info=final_info,
            enable_subtask_progress=enable_subtask_progress,
            instruction_type=instruction_type,
            timing=timing,
            task_name=task_name,
            extra_fields=per_env_extra,
            final_score=final_score,
        )

        episode_results = update_experiment_results(
            run_summary=run_summary,
            episode_results=episode_results,
            episode_results_file=episode_results_file,
        )

    return episode_results
