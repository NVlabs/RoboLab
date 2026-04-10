# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import copy
import fnmatch
import json
import math
import os
import re
import statistics
from collections import Counter

import h5py

from robolab.constants import BENCHMARK_TASK_CATEGORIES, TASK_DIR
from robolab.core.task.status import StatusCode, get_status_name
from robolab.core.utils.file_utils import load_file  # noqa


def _get_metric(ep: dict, key: str):
    """Get a metric value from an episode result, checking both flat and nested 'metrics' dict."""
    val = ep.get(key)
    if val is None:
        val = ep.get('metrics', {}).get(key)
    return val


def _get_timing(ep: dict, key: str):
    """Get a timing value from an episode result's 'timing' dict."""
    return ep.get('timing', {}).get(key)


# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'

# ============================================================================
# Processing Info Functions
# ============================================================================
def filter_episodes_by_task(episode_results: list[dict],
                            tasks: str | list[str] | None) -> list[dict]:
    """
    Filter episode results by task name(s).

    Args:
        episode_results: List of episode result dictionaries
        tasks: Single task name, list of task names, or None to return all episodes

    Returns:
        Filtered list of episodes matching the specified task(s)
    """
    if tasks is None:
        return episode_results

    # Convert single task to list for uniform handling
    if isinstance(tasks, str):
        tasks = [tasks]

    filtered = [ep for ep in episode_results if ep.get("env_name") in tasks]

    if len(filtered) == 0:
        available = set(ep.get('env_name') for ep in episode_results)
        raise ValueError(f"No episodes found for task(s): {tasks}. Available env_names are: {available}")
    return filtered

def filter_episodes_by_pattern(episode_results: list[dict],
                               pattern: str | None,
                                field: str = "env_name") -> list[dict]:
    """
    Filter episode results by a glob-style pattern matching against a field in the episode results.

    Args:
        episode_results: List of episode result dictionaries
        pattern: Glob-style pattern to filter by (e.g., '*cube*', 'pick_*').
                 If None, returns the original list unchanged.
        field: Field to filter by (e.g., 'env_name', 'task_name', 'scene', 'attributes')
    Returns:
        Filtered list of episodes whose field values match the pattern
    """
    if pattern is None:
        return episode_results

    # Check if field is valid
    if field not in episode_results[0]:
        raise ValueError(f"Field '{field}' not found in episode results. Available fields are: {episode_results[0].keys()}")

    filtered_episodes = [
        ep for ep in episode_results
        if ep.get(field) and fnmatch.fnmatch(ep.get(field), pattern)
    ]
    if len(filtered_episodes) == 0:
        raise ValueError(f"No episodes found matching pattern: '{pattern}' in field: '{field}'. Available episodes are: {set([ep.get(field) for ep in episode_results])}")
    return filtered_episodes

def summarize_error_reasons(episode_results: list[dict], indent: str = "  ") -> None:
    """
    Summarize error reasons from failed episodes.

    Prints a breakdown of error types and their counts. For "Wrong object grabbed"
    errors, additionally shows which objects were incorrectly grabbed.

    Args:
        episode_results: List of episode result dictionaries
        indent: Indentation string for output formatting
    """
    # Filter to failed episodes only
    failed_episodes = [ep for ep in episode_results if not ep.get("success", True)]

    if not failed_episodes:
        print(f"\n{indent}{GREEN}No failures found!{RESET}")
        return

    # Extract and categorize error reasons
    error_reasons = []
    wrong_object_details = []  # Track specific objects for "Wrong object grabbed"

    for ep in failed_episodes:
        reason = ep.get("reason", ep.get("info", ""))
        if not reason:
            reason = "(no reason provided)"
        error_reasons.append(reason)

        # Extract wrong object name if applicable
        if "Wrong object grabbed" in reason:
            # Parse: "Wrong object grabbed: 'object_name' (target objects: [...])"
            match = re.search(r"Wrong object grabbed: '([^']+)'", reason)
            if match:
                wrong_object_details.append(match.group(1))

    # Count error types
    # Normalize errors by extracting the main error type (before specific details)
    def normalize_error(reason: str) -> str:
        """Extract the main error type from a reason string."""
        # Handle "Wrong object grabbed: 'x' (...)" -> "Wrong object grabbed"
        if "Wrong object grabbed" in reason:
            return "Wrong object grabbed"
        # Handle "Wrong object detached: 'x'" -> "Wrong object detached"
        if "Wrong object detached" in reason:
            return "Wrong object detached"
        # Handle object displacement events
        if "Object bumped:" in reason:
            return "Object bumped"
        if "Object moved:" in reason:
            return "Object moved"
        if "Object out of scene:" in reason:
            return "Object out of scene"
        if "Object started moving:" in reason:
            return "Object started moving"
        if "Object tipped over:" in reason:
            return "Object tipped over"
        # Handle gripper/grasp events
        if "Target object dropped" in reason:
            return "Target object dropped"
        if "Gripper hit object:" in reason:
            return "Gripper hit object"
        if "Multiple objects grabbed:" in reason:
            return "Multiple objects grabbed"
        # For other errors, use as-is (they're usually short)
        return reason

    normalized_errors = [normalize_error(r) for r in error_reasons]
    error_counts = Counter(normalized_errors)

    # Print error summary
    # Sort by count (descending)
    for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"{indent}{error_type}")

    # If there were wrong object grabs, show the object breakdown
    if wrong_object_details:
        wrong_object_counts = Counter(wrong_object_details)
        print(f"{indent}{BOLD}Wrong Objects Grabbed:{RESET}")
        for obj_name, count in sorted(wrong_object_counts.items(), key=lambda x: -x[1]):
            print(f"{indent}  {count}x - '{obj_name}'")


# Container object patterns to exclude from wrong object grabbed counts
CONTAINER_PATTERNS = [
    'grey_bin', 'grey_container', 'bin', 'crate', 'box', 'container',
    'basket', 'tray', 'bucket', 'shelf', 'drawer', 'cabinet'
]

def is_container_object(obj_name: str) -> bool:
    """Check if an object name matches a container pattern."""
    obj_lower = obj_name.lower()
    return any(pattern in obj_lower for pattern in CONTAINER_PATTERNS)


def _resolve_log_file(ep: dict) -> str | None:
    """Resolve the log file path for an episode result, supporting per-env and legacy formats."""
    data_dir = ep.get('data_dir')
    if not data_dir:
        return None
    run_idx = ep.get('run', ep.get('episode'))
    env_id = ep.get('env_id')
    if run_idx is None:
        return None
    # Per-env log file (multi-env)
    if env_id is not None:
        log_file = os.path.join(data_dir, f"log_{run_idx}_env{env_id}.json")
        if os.path.exists(log_file):
            return log_file
    # Fallback: legacy single log file
    log_file = os.path.join(data_dir, f"log_{run_idx}.json")
    if os.path.exists(log_file):
        return log_file
    return None


def get_wrong_object_stats(episode_results: list[dict], exclude_containers: bool = False) -> dict:
    """
    Get wrong object grabbed statistics from log files for the given episodes.

    Reads the log files for each episode and extracts WRONG_OBJECT_GRABBED events,
    counting each grab event (after consecutive deduplication in extract_subtask_status_changes).

    Each grab event is counted separately - if the robot grabs the same object multiple
    times (grab -> release -> grab again), each grab is counted as a separate event.

    Args:
        episode_results: List of episode result dictionaries (must have 'data_dir' and 'episode' keys)
        exclude_containers: If True, exclude container objects (bin, crate, box, etc.) from counts

    Returns:
        Dictionary with:
            - 'count': Total number of wrong object grabbed events
            - 'objects': Counter of object names that were grabbed (object_name -> count)
            - 'count_success': Count of events in successful episodes
            - 'count_failure': Count of events in failed episodes
            - 'objects_success': Counter of objects grabbed in successful episodes
            - 'objects_failure': Counter of objects grabbed in failed episodes
    """
    wrong_object_counts = Counter()
    wrong_object_counts_success = Counter()
    wrong_object_counts_failure = Counter()

    for ep in episode_results:
        log_file = _resolve_log_file(ep)
        if log_file is None:
            continue

        log_data = load_file(log_file)

        if log_data is None:
            continue

        # Extract status changes from log (with consecutive deduplication applied)
        status_changes = extract_subtask_status_changes(log_data)

        if not status_changes:
            continue

        # Determine if this episode was successful
        is_success = ep.get("success", False)

        # Count each grab event (not unique per episode)
        for change in status_changes:
            status_code = change.get("status", 0)

            # Check for wrong object grabbed by status code
            if status_code == StatusCode.WRONG_OBJECT_GRABBED_FAILURE:
                info = change.get("info", "")
                # Extract object name from info: "Wrong object grabbed: 'object_name' (...)"
                match = re.search(r"Wrong object grabbed: '([^']+)'", info)
                if match:
                    obj_name = match.group(1)

                    # Skip container objects if exclude_containers is True
                    if exclude_containers and is_container_object(obj_name):
                        continue

                    wrong_object_counts[obj_name] += 1  # Count every grab event
                    if is_success:
                        wrong_object_counts_success[obj_name] += 1
                    else:
                        wrong_object_counts_failure[obj_name] += 1

    return {
        'count': sum(wrong_object_counts.values()),
        'objects': wrong_object_counts,
        'count_success': sum(wrong_object_counts_success.values()),
        'count_failure': sum(wrong_object_counts_failure.values()),
        'objects_success': wrong_object_counts_success,
        'objects_failure': wrong_object_counts_failure,
    }


def format_wrong_object_str(stats: dict, num_episodes: int, csv: bool = False, show_objects: bool = True, split_by_success: bool = False) -> str:
    """
    Format wrong object grabbed statistics as a string.

    Args:
        stats: Dictionary from get_wrong_object_stats()
        num_episodes: Number of episodes to calculate average
        csv: If True, format for CSV output
        show_objects: If True, include the list of objects grabbed
        split_by_success: If True, show counts split by total/success/failure

    Returns:
        Formatted string like "4/1 (obj1 x2, obj2 x1)" showing total/avg or "-" if no wrong objects
        If split_by_success, returns "T/S/F" format (e.g., "5/2/3")
    """
    if stats['count'] == 0:
        return "-"

    total = stats['count']
    # Calculate average per episode
    avg_per_episode = total / num_episodes if num_episodes > 0 else 0
    avg_rounded = round(avg_per_episode)

    if split_by_success:
        count_success = stats.get('count_success', 0)
        count_failure = stats.get('count_failure', 0)
        return f"{total}/{count_success}/{count_failure}"

    if show_objects:
        obj_counts = stats['objects']
        # Format object list with counts
        obj_list = ", ".join(
            f"{obj} x{cnt}" for obj, cnt in sorted(obj_counts.items(), key=lambda x: -x[1])
        )
        return f"{total}/{avg_rounded} ({obj_list})"
    else:
        return f"{total}/{avg_rounded}"


def format_wrong_object_names_str(stats: dict) -> str:
    """
    Format wrong object names as a string (just the object names with counts).

    Args:
        stats: Dictionary from get_wrong_object_stats()

    Returns:
        Formatted string like "obj1(2), obj2(1)" or "-" if no wrong objects
    """
    if stats['count'] == 0 or not stats.get('objects'):
        return "-"

    obj_counts = stats['objects']
    # Format object list with counts as name(count)
    obj_list = ", ".join(
        f"{obj}({cnt})" for obj, cnt in sorted(obj_counts.items(), key=lambda x: -x[1])
    )
    return obj_list


def summarize_timestep_errors(episode_results: list[dict], indent: str = "  ") -> None:
    """
    Summarize timestep-level errors from log files for the given episodes.

    Reads the log files for each episode and aggregates unique error events
    (not counting every timestep, but distinct error occurrences).
    For "Wrong object grabbed" errors, shows which objects were grabbed.

    Args:
        episode_results: List of episode result dictionaries (must have 'data_dir' and 'episode' keys)
        indent: Indentation string for output formatting
    """
    # Aggregate errors across all episodes
    error_counts = Counter()  # Normalized error type -> count of unique occurrences
    wrong_object_details = []  # List of wrong object names

    total_episodes = len(episode_results)
    episodes_with_errors = 0

    for ep in episode_results:
        log_file = _resolve_log_file(ep)
        if log_file is None:
            continue

        log_data = load_file(log_file)

        if log_data is None:
            continue

        # Extract status changes from log
        status_changes = extract_subtask_status_changes(log_data)

        if not status_changes:
            continue

        # Track unique errors in this episode (to avoid counting repeated timesteps)
        episode_errors = set()
        episode_wrong_objects = set()

        for change in status_changes:
            status_code = change.get("status", 0)
            info = change.get("info", "")

            # Only count failure/error status codes (200+)
            if status_code < 200:
                continue

            # Get error type name from StatusCode enum
            error_type = get_status_name(status_code)

            # For WRONG_OBJECT_GRABBED, extract object name from info
            if status_code == StatusCode.WRONG_OBJECT_GRABBED_FAILURE:
                match = re.search(r"Wrong object grabbed: '([^']+)'", info)
                if match:
                    obj_name = match.group(1)
                    if obj_name not in episode_wrong_objects:
                        episode_wrong_objects.add(obj_name)
                        wrong_object_details.append(obj_name)

            # Count unique error types per episode
            if error_type not in episode_errors:
                episode_errors.add(error_type)
                error_counts[error_type] += 1

        if episode_errors:
            episodes_with_errors += 1

    if not error_counts:
        return

    # Print error summary
    # Get wrong object counts for inline display
    wrong_object_counts = Counter(wrong_object_details) if wrong_object_details else {}

    # Sort by count (descending)
    for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        # For WRONG_OBJECT_GRABBED, show objects inline
        if error_type == "WRONG_OBJECT_GRABBED" and wrong_object_counts:
            # Format objects: ('obj1', 'obj2') or ('obj1' x3, 'obj2' x2) for multiple episodes
            if total_episodes == 1:
                # Single episode: just list object names
                obj_list = ", ".join(f"'{obj}'" for obj in sorted(wrong_object_counts.keys()))
            else:
                # Multiple episodes: show counts
                obj_list = ", ".join(
                    f"'{obj}' {cnt}x" for obj, cnt in sorted(wrong_object_counts.items(), key=lambda x: -x[1])
                )
            print(f"{indent} {error_type} ({obj_list})")
        else:
            print(f"{indent} {error_type} ({count}x)")


def extract_subtask_info(info):
    """
    Extract the subtask completion information from the info dictionary.
    This function does the following: copy.deepcopy(info["log"].get("SubtaskCompletionRecorderTerm")) and returns None otherwise.
    """
    if info is None:
        return None
    log = info.get("log")
    if log is None:
        return None
    subtask_data = copy.deepcopy(log.get("SubtaskCompletionRecorderTerm", {}))
    return subtask_data


def get_current_subtask_info(env, env_id: int = 0) -> dict | None:
    """
    Get the current subtask info directly from the recorder term for a specific env.

    Args:
        env: The environment instance with a recorder_manager
        env_id: Which env to get info for (default: 0)

    Returns:
        Current subtask info dict with 'status', 'info', 'completed', 'total', 'score' keys,
        or None if no recorder manager available.
    """
    if env.recorder_manager is None:
        return None

    if not hasattr(env.recorder_manager, '_terms'):
        return None

    for term in env.recorder_manager._terms.values():
        if hasattr(term, 'infos') and hasattr(term, 'subtask_state_machines'):
            return copy.deepcopy(term.infos[env_id])

    return None


def get_all_env_subtask_infos(env) -> list[dict] | None:
    """
    Get the current subtask info for all envs from the recorder term.

    Args:
        env: The environment instance with a recorder_manager

    Returns:
        List of per-env subtask info dicts, or None if no recorder manager available.
    """
    if env.recorder_manager is None:
        return None

    if not hasattr(env.recorder_manager, '_terms'):
        return None

    for term in env.recorder_manager._terms.values():
        if hasattr(term, 'infos') and hasattr(term, 'subtask_state_machines'):
            return copy.deepcopy(term.infos)

    return None


def get_final_subtask_info(env, env_id: int | None = None) -> dict | list[dict | None] | None:
    """
    Get the final subtask info for incomplete episodes.

    Args:
        env: The environment instance with a recorder_manager
        env_id: If None, return list of all envs' final infos.
               If int, return that env's final info.

    Returns:
        Final info dict (or list of dicts), or None if not available.
    """
    if env.recorder_manager is None:
        return None

    if not hasattr(env.recorder_manager, '_terms'):
        return None

    for term in env.recorder_manager._terms.values():
        if hasattr(term, 'get_final_info'):
            return term.get_final_info(env_id=env_id)

    return None

def extract_initial_state_info(info, type="rigid_object"):
    """
    Extract the initial state of the environment from the info dictionary.
    This function does the following: copy.deepcopy(info["log"].get("InitialStateRecorder").get("rigid_object")) and returns None otherwise.
    """
    if info is None:
        return None
    log = info.get("log")
    if log is None:
        return None
    init_state_recorder = log.get("InitialStateRecorder")
    if init_state_recorder is None:
        return None
    if type not in init_state_recorder:
        return None
    return copy.deepcopy(init_state_recorder.get(type))

# ============================================================================
# Experiment Functions
# ============================================================================

def init_experiment(output_dir: str) -> tuple[str, list[dict]]:
    """Initialize or load existing experiment results."""
    episode_results_file = os.path.join(output_dir, "episode_results.jsonl")

    # Load existing results (supports both .jsonl and legacy .json)
    episode_results = load_episode_results(output_dir)
    if episode_results:
        print(f"Loaded {len(episode_results)} existing results from {output_dir}.")

    return episode_results_file, episode_results


def update_experiment_results(run_summary: dict, episode_results_file: str, episode_results: list[dict] = None):
    if run_summary is None:
        return episode_results

    if episode_results is None:
        episode_results = []

    # Print result
    succ = run_summary.get('success', None)
    env_name = run_summary.get('env_name', None)
    episode = run_summary.get('episode', None)

    run_name = f"{env_name}_{episode}"
    if succ:
        print(f"{GREEN}{run_name} complete:{RESET}", ", ".join([f"{k}: {v}" for k, v in run_summary.items() if k not in ("env_name", "episode")]))
    else:
        print(f"{RED}{run_name} complete:{RESET}", ", ".join([f"{k}: {v}" for k, v in run_summary.items() if k not in ("env_name", "episode")]))

    # Record results by episode to file
    episode_results.append(run_summary)
    append_episode_to_jsonl(episode_results_file, run_summary)

    return episode_results

def dump_results_to_file(result_file: str, data: dict | list, append: bool = False):
    """Save results data to a JSON file, optionally appending to existing data."""
    if append:
        # Load existing data
        if os.path.exists(result_file):
            existing_data = load_file(result_file)
        else:
            existing_data = []

        # Ensure existing_data is a list
        if not isinstance(existing_data, list):
            existing_data = [existing_data]

        # Append or extend
        if isinstance(data, list):
            existing_data.extend(data)
        else:
            existing_data.append(data)

        data = existing_data

    # Write data to file
    with open(result_file, "w") as f:
        json.dump(data, f)
    print(f"Wrote to {result_file}.")


def append_episode_to_jsonl(file_path: str, episode: dict):
    """Append a single episode result as one JSON line (append-only, no read-modify-write)."""
    with open(file_path, "a") as f:
        f.write(json.dumps(episode) + "\n")


def load_jsonl(file_path: str) -> list[dict]:
    """Load a JSONL file, returning a list of dicts. Skips blank/malformed lines."""
    results = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line {i} in {file_path}")
    return results


def load_episode_results(folder_path: str) -> list[dict]:
    """Load episode results from either JSONL (new) or JSON (legacy) format.

    Checks for episode_results.jsonl first, falls back to episode_results.json.
    """
    jsonl_path = os.path.join(folder_path, "episode_results.jsonl")
    json_path = os.path.join(folder_path, "episode_results.json")

    if os.path.exists(jsonl_path):
        return load_jsonl(jsonl_path)
    elif os.path.exists(json_path):
        data = load_file(json_path)
        return data if data is not None else []
    else:
        return []


def save_episode_results_jsonl(file_path: str, episodes: list[dict]):
    """Write a list of episode dicts as JSONL (one JSON object per line)."""
    with open(file_path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")


# ============================================================================
# Result Processing Functions
# ============================================================================

def get_avg_score(episode_results: list[dict], task_name=None, fail_only=False):
    """
    Get the average score of the episode results.
    If task_name is provided, only return the average score for that task.
    If fail_only is True, only return the average score for failed episodes.
    Returns None if there are no valid scores (all None).
    """
    if len(episode_results) == 0:
        return None

    if task_name is not None:
        episode_results = [episode_result for episode_result in episode_results if episode_result.get("env_name") == task_name]

    if fail_only:
        episode_results = [episode_result for episode_result in episode_results if episode_result.get("success") == False]

    # Filter out None scores
    valid_scores = []
    for episode_result in episode_results:
        score = episode_result.get("score")
        if score is not None:
            valid_scores.append(float(score))

    if len(valid_scores) == 0:
        return None

    return sum(valid_scores) / len(valid_scores)

def get_task_based_results(episode_data: list[dict]) -> dict:
    """Get task-based results from episode data."""
    results = {}
    results["success"] = []
    results["failure"] = []
    for episode in episode_data:
        env_name = episode.get("env_name")
        if env_name not in results:
            results[env_name] = {}
            results[env_name]["success"] = []
            results[env_name]["failure"] = []
        if episode.get("success"):
            results["success"].append(f"{env_name}_{episode.get('episode')}")
            results[env_name]["success"].append(episode)
        else:
            results["failure"].append(f"{env_name}_{episode.get('episode')}")
            results[env_name]["failure"].append(episode)
    return results

def get_success_stats(results: dict):
    """
    Assumes results is a dictionary with keys "success" and "failure"
    """
    success_runs = results.get("success", None)
    failure_runs = results.get("failure", None)
    if success_runs is None and failure_runs is None:
        return 0, 0, 0, 0, 0, [], []
    elif len(success_runs) == 0 and len(failure_runs) == 0:
        return 0, 0, 0, 0, 0, [], []
    num_success = len(success_runs)
    num_failure = len(failure_runs)
    num_total = num_success + num_failure
    success_rate = num_success / num_total
    failure_rate = num_failure / num_total
    return num_success, num_failure, num_total, success_rate, failure_rate, success_runs, failure_runs

def get_run_data(episode_results: list[dict], env_name: str, episode: int) -> dict:
    """Get experiment data for a specific environment and episode."""
    for episode_result in episode_results:
        if episode_result.get("env_name") == env_name and episode_result.get("episode") == episode:
            return episode_result
    return None

def check_run_complete(episode_results: list[dict], env_name: str, episode: int) -> bool:
    """Check if a run is complete."""
    return get_run_data(episode_results=episode_results, env_name=env_name, episode=episode) is not None

def check_all_episodes_complete(episode_results: list[dict], env_name: str, num_episodes: int) -> bool:
    """Check if all episodes for an environment are complete."""
    for episode in range(num_episodes):
        if get_run_data(episode_results=episode_results, env_name=env_name, episode=episode) is None:
            return False
    return True

def extract_subtask_status_changes(log_data: list[dict]) -> list[dict]:
    """
    Extract only the steps where subtask status changed.

    When multiple conditions are satisfied in a single timestep (e.g., object_grabbed
    AND object_above_bottom), this function expands them into separate entries so each
    condition gets its own row in the output. This handles cases like grasp-based
    conditions (object_grabbed, object_dropped) that were previously skipped.

    For certain error types (WRONG_OBJECT_GRABBED, GRIPPER_HIT_TABLE, WRONG_OBJECT_DETACHED),
    consecutive duplicate entries are deduplicated - only the first in a consecutive run
    is kept. Non-consecutive occurrences (with gaps) are preserved as separate events.

    Args:
        log_data: List of subtask status dictionaries at each timestep

    Returns:
        List of dictionaries containing step, status, completed, total, info, and score
    """
    if not log_data or len(log_data) == 0:
        return []

    # Status codes that should be deduplicated across consecutive steps
    DEDUP_CODES = {250, 255, 257}  # WRONG_OBJECT_GRABBED, GRIPPER_HIT_TABLE, WRONG_OBJECT_DETACHED

    status_changes = []
    prev_status = 0
    prev_score = 0.0
    seen_status_codes = set()  # Track which status codes we've already recorded for this step

    # Track last step where each (info, code) was seen for consecutive deduplication
    last_seen_step: dict[tuple[str, int], int] = {}

    for step, entry in enumerate(log_data):
        if entry is None:
            continue
        status = entry.get("status", 0)
        score = entry.get("score", 0.0)
        all_status_codes = entry.get("all_status_codes", [])

        # Check if there are multiple conditions satisfied in this step
        if all_status_codes and len(all_status_codes) > 0:
            # Expand all status codes into separate entries
            num_conditions = len(all_status_codes)
            score_per_condition = score / num_conditions if num_conditions > 0 else score
            cumulative_score = prev_score

            for i, (cond_info, cond_status) in enumerate(all_status_codes):
                # Skip if we've already seen this status code in the current batch
                if cond_status in seen_status_codes:
                    continue
                seen_status_codes.add(cond_status)

                cumulative_score += score_per_condition

                # Only add if status is non-zero and different from what we've seen
                if cond_status != 0:
                    # Check for consecutive duplicate for target error codes
                    if cond_status in DEDUP_CODES:
                        key = (cond_info, cond_status)
                        if key in last_seen_step and step == last_seen_step[key] + 1:
                            # This is a consecutive duplicate - skip it but update tracking
                            last_seen_step[key] = step
                            continue
                        last_seen_step[key] = step

                    status_changes.append({
                        "step": step,
                        "status": cond_status,
                        "completed": entry.get("completed", 0),
                        "total": entry.get("total", 0),
                        "info": cond_info,
                        "score": cumulative_score
                    })

            prev_status = status
            prev_score = score
            seen_status_codes.clear()

        elif status != 0 and (status != prev_status or score != prev_score):
            # Original behavior for single status changes (backward compatibility)
            status_changes.append({
                "step": step,
                "status": status,
                "completed": entry.get("completed", 0),
                "total": entry.get("total", 0),
                "info": entry.get("info", ""),
                "score": score
            })
            prev_status = status
            prev_score = score

    return status_changes

# ============================================================================
# Printing Functions for episode_results.json
# ============================================================================

def get_attribute_grouped_results(episode_results: list[dict]) -> dict:
    """
    Group episodes by their attributes. Episodes can appear in multiple groups.

    Args:
        episode_results: List of episode result dictionaries

    Returns:
        Dictionary with attributes as keys and {"success": [...], "failure": [...]} as values
    """
    results = {}
    for episode in episode_results:
        attributes = episode.get("attributes", [])
        for attr in attributes:
            if attr not in results:
                results[attr] = {"success": [], "failure": []}

            if episode.get("success"):
                results[attr]["success"].append(episode)
            else:
                results[attr]["failure"].append(episode)

    return results

def get_scene_grouped_results(episode_results: list[dict]) -> dict:
    """
    Group episodes by their scene. Episodes must have a "scene" field.

    Args:
        episode_results: List of episode result dictionaries with "scene" field

    Returns:
        Dictionary with scene names as keys and {"success": [...], "failure": [...]} as values
    """
    results = {}
    for episode in episode_results:
        scene = episode.get("scene", "(unknown scene)")
        if scene not in results:
            results[scene] = {"success": [], "failure": []}

        if episode.get("success"):
            results[scene]["success"].append(episode)
        else:
            results[scene]["failure"].append(episode)

    # Add overall success/failure lists
    results["success"] = [ep for ep in episode_results if ep.get("success")]
    results["failure"] = [ep for ep in episode_results if not ep.get("success")]
    return results

def print_result_table(episode_results: list[dict],
                      group_by: str = "task",
                      title: str = None,
                      show_scores: bool = True,
                      show_eps: bool = False,
                      show_header: bool = True,
                      show_total: bool = True,
                      show_duration: bool = True,
                      show_wrong_objects: bool = False,
                      exclude_containers: bool = False,
                      show_metrics: bool = True,
                      show_metric_stddev: bool = False,
                      show_timing: bool = False,
                      dt: float = None,
                      csv: bool = False,
                      csv_compact: bool = False):
    """
    Helper function to print a complete result table with optional title and formatting.

    Args:
        episode_results: List of episode result dictionaries
        group_by: Field to group by ("task" or "attributes")
        title: Optional title to display above the table
        show_scores: If True, show score columns
        show_eps: If True, show episode ID columns
        show_header: If True, show the column header row
        show_total: If True, show the total summary row
        show_wrong_objects: If True, show wrong object grabbed column
        exclude_containers: If True, exclude container objects from wrong object counts
        show_metrics: If True, show trajectory metrics columns (EE SPARC, Path Length, Avg Speed)
        show_metric_stddev: If True, show stddev columns for duration and metrics (verbose mode)
        show_timing: If True, show wall-clock timing columns (it/s, Walltime(m))
        csv: If True, output in CSV format for easy pasting into spreadsheets
        csv_compact: If True, combine value and stddev into single column like '-9.14 (± 4.72)'
    """
    header, total_line, table, total_width = get_grouped_result_table_str(
        episode_results, group_by=group_by, show_scores=show_scores, show_eps=show_eps,
        show_duration=show_duration, show_wrong_objects=show_wrong_objects,
        exclude_containers=exclude_containers, show_metrics=show_metrics,
        show_metric_stddev=show_metric_stddev, show_timing=show_timing,
        dt=dt, csv=csv, csv_compact=csv_compact
    )

    if csv:
        # CSV format: just print header, total, and table rows
        if show_header:
            print(header)
        if show_total:
            print(total_line)
        print(table)
    else:
        if title:
            print(format_centered_header(title, total_width))

        if show_header:
            print(header)
            print("-" * total_width)

        if show_total:
            print(total_line)
            print("-" * total_width)

        print(table)
        print("-" * total_width)


def get_attribute_sort_key(attr_name: str) -> tuple:
    """
    Generate a sort key for attributes that prioritizes difficulty levels.

    Args:
        attr_name: Attribute name to sort

    Returns:
        Tuple for sorting: (priority_order, name) where lower priority_order comes first
    """
    priority_attrs = ['simple', 'moderate', 'complex']
    if attr_name in priority_attrs:
        return (0, priority_attrs.index(attr_name), attr_name)
    else:
        return (1, 0, attr_name)

def get_grouped_result_table_str(episode_results: list[dict],
                               group_by: str = "task",
                               show_scores: bool = True,
                               show_eps: bool = False,
                               show_duration: bool = True,
                               show_wrong_objects: bool = False,
                               exclude_containers: bool = False,
                               show_metrics: bool = True,
                               show_metric_stddev: bool = False,
                               show_timing: bool = False,
                               dt: float = None,
                               csv: bool = False,
                               csv_compact: bool = False):
    """
    Generate a formatted table string of results grouped by a specified field.

    Args:
        episode_results: List of episode result dictionaries
        group_by: Field to group by. Can be "task", "attributes", or "scene"
        show_scores: If True, show score columns (Score and Fail Score)
        show_eps: If True, show episode ID columns (Success Eps and Failure Eps)
        show_wrong_objects: If True, show wrong object grabbed column
        exclude_containers: If True, exclude container objects from wrong object counts
        show_metrics: If True, show trajectory metrics columns (EE SPARC, Path Length, Avg Speed)
        show_metric_stddev: If True, show stddev columns for duration and metrics (verbose mode)
        show_timing: If True, show wall-clock timing columns (it/s, Walltime(m))
        csv_compact: If True, combine value and stddev into single column like '-9.14 (± 4.72)'

    Returns:
        Tuple of (header, total_line, table, total_width) for flexible formatting
    """
    # Group episodes by the specified field
    if group_by == "task":
        results = get_task_based_results(episode_results)
        group_name_header = "Task Name"
    elif group_by == "attributes":
        results = get_attribute_grouped_results(episode_results)
        group_name_header = "Attribute"
    elif group_by == "scene":
        results = get_scene_grouped_results(episode_results)
        group_name_header = "Scene"
    else:
        raise ValueError(f"Invalid group_by value: {group_by}. Must be 'task', 'attributes', or 'scene'.")

    group_name_width = len(group_name_header)
    success_count_width = len("000/000 ")
    success_pct_width = len("100.0% ")
    failure_count_width = len("000/000 ")
    failure_pct_width = len("100.0% ")

    # Optional column widths
    avg_score_width = len("Avg Score") if show_scores else 0
    avg_score_fail_width = len("Score(fail)") if show_scores else 0
    success_eps_width = len("Eps(succ)") if show_eps else 0
    duration_width = len("Time(s)") if show_duration else 0
    wrong_objects_width = len("WrongObj(T/S/F)") if show_wrong_objects else 0
    wrong_objects_names_width = len("WrongObjNames") if show_wrong_objects and show_metric_stddev and not csv else 0

    # Metrics column widths
    ee_sparc_width = len("EE SPARC") if show_metrics else 0
    ee_sparc_stddev_width = len("SPARC σ") if show_metrics and show_metric_stddev else 0
    path_length_width = len("PathLen(m)") if show_metrics else 0
    path_length_stddev_width = len("Path σ") if show_metrics and show_metric_stddev else 0
    avg_speed_width = len("Speed(cm/s)") if show_metrics else 0
    avg_speed_stddev_width = len("Speed σ") if show_metrics and show_metric_stddev else 0
    duration_stddev_width = len("Time σ") if show_duration and show_metric_stddev else 0

    # Timing column widths
    it_per_sec_width = len("it/s") if show_timing else 0
    wall_total_width = len("Walltime(m)") if show_timing else 0

    grouped_data = []
    # Use custom sorting for attributes to prioritize difficulty levels
    if group_by == "attributes":
        sorted_items = sorted(results.items(), key=lambda x: get_attribute_sort_key(x[0]))
    else:
        sorted_items = sorted(results.items())

    for group_name, group_results in sorted_items:
        if group_name not in ("success", "failure"):
            num_success, num_failure, num_total, success_rate, failure_rate, success_runs, failure_runs = get_success_stats(group_results)
            success_count_str = f"{num_success}/{num_total}"
            success_pct_str = f"{success_rate*100:.1f}%"
            success_pct_csv_str = f"{success_rate*100:.1f}"  # No % for CSV
            failure_count_str = f"{num_failure}/{num_total}"
            failure_pct_str = f"{failure_rate*100:.1f}%"

            # Optional: Get scores
            if show_scores:
                # For tasks, we can filter by env_name; for attributes, use all episodes in this group
                if group_by == "task":
                    group_episodes = [ep for ep in episode_results if ep.get("env_name") == group_name]
                else:
                    group_episodes = success_runs + failure_runs

                avg_score = get_avg_score(group_episodes)
                avg_score_fail = get_avg_score([ep for ep in group_episodes if not ep.get("success")], fail_only=False)
                avg_score_str = format_score(avg_score)
                avg_score_fail_str = format_score(avg_score_fail)
                avg_score_width = max(avg_score_width, len(avg_score_str))
                avg_score_fail_width = max(avg_score_fail_width, len(avg_score_fail_str))
            else:
                avg_score_str = ""
                avg_score_fail_str = ""

            # Optional: Get episode IDs
            if show_eps:
                success_episodes = [run.get('episode') for run in success_runs]
                success_eps_str = str(success_episodes) if success_episodes else "-"
                success_eps_width = max(success_eps_width, len(success_eps_str))
            else:
                success_eps_str = ""

            # Optional: Calculate average duration (only for successful episodes)
            if show_duration:
                # For tasks, we can filter by env_name; for attributes, use all episodes in this group
                if group_by == "task":
                    group_episodes = [ep for ep in episode_results if ep.get("env_name") == group_name]
                else:
                    group_episodes = success_runs + failure_runs

                # Calculate average duration (only for successful episodes)
                valid_durations = []
                for ep in group_episodes:
                    # Only count successful episodes
                    if not ep.get("success"):
                        continue

                    # First check if duration is available directly
                    duration = ep.get("duration")
                    if duration is not None:
                        valid_durations.append(duration)
                    else:
                        # Fall back to calculating from episode_step * dt
                        episode_step = ep.get("episode_step")
                        if episode_step is not None:
                            # Get dt for this episode's task
                            ep_dt = ep.get("dt")
                            if ep_dt is None:
                                ep_dt = dt

                            if ep_dt is not None:
                                valid_durations.append(episode_step * ep_dt)

                if len(valid_durations) > 0:
                    avg_duration = sum(valid_durations) / len(valid_durations)
                    duration_str = f"{avg_duration:.2f}"
                    # Calculate stddev
                    if len(valid_durations) > 1:
                        stddev_duration = statistics.stdev(valid_durations)
                        stddev_str = f"{stddev_duration:.2f}"
                    else:
                        stddev_str = "-"
                else:
                    duration_str = "-"
                    stddev_str = "-"
                duration_width = max(duration_width, len(duration_str))
            else:
                duration_str = ""
                stddev_str = ""

            # Optional: Get wrong object grabbed stats
            if show_wrong_objects:
                # For tasks, we can filter by env_name; for attributes, use all episodes in this group
                if group_by == "task":
                    group_episodes = [ep for ep in episode_results if ep.get("env_name") == group_name]
                else:
                    group_episodes = success_runs + failure_runs

                wrong_obj_stats = get_wrong_object_stats(group_episodes, exclude_containers=exclude_containers)
                wrong_objects_str = format_wrong_object_str(wrong_obj_stats, num_episodes=num_total, csv=csv, show_objects=False, split_by_success=True)
                wrong_objects_width = max(wrong_objects_width, len(wrong_objects_str))
                # Extract individual counts for CSV
                wrong_obj_total = wrong_obj_stats.get('count', 0)
                wrong_obj_success = wrong_obj_stats.get('count_success', 0)
                wrong_obj_fail = wrong_obj_stats.get('count_failure', 0)
                # Get object names
                wrong_objects_names_str = format_wrong_object_names_str(wrong_obj_stats)
                if not csv and show_metric_stddev:
                    wrong_objects_names_width = max(wrong_objects_names_width, len(wrong_objects_names_str))
            else:
                wrong_objects_str = ""
                wrong_objects_names_str = ""
                wrong_obj_total = 0
                wrong_obj_success = 0
                wrong_obj_fail = 0

            # Optional: Calculate trajectory metrics averages
            if show_metrics:
                if group_by == "task":
                    group_episodes = [ep for ep in episode_results if ep.get("env_name") == group_name]
                else:
                    group_episodes = success_runs + failure_runs

                # EE SPARC (average across all episodes)
                sparc_values = [float(_get_metric(ep, 'ee_sparc')) for ep in group_episodes if _get_metric(ep, 'ee_sparc') is not None and math.isfinite(float(_get_metric(ep, 'ee_sparc')))]
                if sparc_values:
                    avg_sparc = sum(sparc_values) / len(sparc_values)
                    ee_sparc_str = f"{avg_sparc:.2f}"
                    if len(sparc_values) > 1:
                        sparc_stddev = statistics.stdev(sparc_values)
                        ee_sparc_stddev_str = f"{sparc_stddev:.2f}"
                    else:
                        ee_sparc_stddev_str = "-"
                else:
                    ee_sparc_str = "-"
                    ee_sparc_stddev_str = "-"

                # Path Length (average across all episodes)
                path_values = [float(_get_metric(ep, 'ee_path_length')) for ep in group_episodes if _get_metric(ep, 'ee_path_length') is not None and math.isfinite(float(_get_metric(ep, 'ee_path_length')))]
                if path_values:
                    avg_path = sum(path_values) / len(path_values)
                    path_length_str = f"{avg_path:.2f}"
                    if len(path_values) > 1:
                        path_stddev = statistics.stdev(path_values)
                        path_length_stddev_str = f"{path_stddev:.2f}"
                    else:
                        path_length_stddev_str = "-"
                else:
                    path_length_str = "-"
                    path_length_stddev_str = "-"

                # Average Speed (average across all episodes, convert m/s to cm/s)
                speed_values = [float(_get_metric(ep, 'ee_speed_mean')) * 100 for ep in group_episodes if _get_metric(ep, 'ee_speed_mean') is not None and math.isfinite(float(_get_metric(ep, 'ee_speed_mean')))]
                if speed_values:
                    avg_speed = sum(speed_values) / len(speed_values)
                    avg_speed_str = f"{avg_speed:.1f}"
                    if len(speed_values) > 1:
                        speed_stddev = statistics.stdev(speed_values)
                        avg_speed_stddev_str = f"{speed_stddev:.1f}"
                    else:
                        avg_speed_stddev_str = "-"
                else:
                    avg_speed_str = "-"
                    avg_speed_stddev_str = "-"

                ee_sparc_width = max(ee_sparc_width, len(ee_sparc_str))
                path_length_width = max(path_length_width, len(path_length_str))
                avg_speed_width = max(avg_speed_width, len(avg_speed_str))
                if show_metric_stddev:
                    ee_sparc_stddev_width = max(ee_sparc_stddev_width, len(ee_sparc_stddev_str))
                    path_length_stddev_width = max(path_length_stddev_width, len(path_length_stddev_str))
                    avg_speed_stddev_width = max(avg_speed_stddev_width, len(avg_speed_stddev_str))
            else:
                ee_sparc_str = ""
                path_length_str = ""
                avg_speed_str = ""
                ee_sparc_stddev_str = ""
                path_length_stddev_str = ""
                avg_speed_stddev_str = ""

            # Optional: Calculate timing averages
            if show_timing:
                if group_by == "task":
                    group_episodes = [ep for ep in episode_results if ep.get("env_name") == group_name]
                else:
                    group_episodes = success_runs + failure_runs

                it_values = [float(_get_timing(ep, 'it_per_sec')) for ep in group_episodes if _get_timing(ep, 'it_per_sec') is not None]
                wall_values = [float(_get_timing(ep, 'wall_total_s')) for ep in group_episodes if _get_timing(ep, 'wall_total_s') is not None]
                it_per_sec_str = f"{sum(it_values)/len(it_values):.2f}" if it_values else "-"
                wall_total_str = f"{sum(wall_values)/len(wall_values)/60:.1f}" if wall_values else "-"
                it_per_sec_width = max(it_per_sec_width, len(it_per_sec_str))
                wall_total_width = max(wall_total_width, len(wall_total_str))
            else:
                it_per_sec_str = ""
                wall_total_str = ""

            group_name_width = max(group_name_width, len(group_name))
            success_count_width = max(success_count_width, len(success_count_str))
            success_pct_width = max(success_pct_width, len(success_pct_str))
            failure_count_width = max(failure_count_width, len(failure_count_str))
            failure_pct_width = max(failure_pct_width, len(failure_pct_str))

            grouped_data.append((group_name, success_count_str, success_pct_str, success_pct_csv_str, failure_count_str, failure_pct_str, avg_score_str, avg_score_fail_str, success_eps_str, duration_str, stddev_str, wrong_objects_str, wrong_obj_total, wrong_obj_success, wrong_obj_fail, wrong_objects_names_str, ee_sparc_str, ee_sparc_stddev_str, path_length_str, path_length_stddev_str, avg_speed_str, avg_speed_stddev_str, it_per_sec_str, wall_total_str, num_success, num_failure, num_total))

    # Calculate total width based on what columns are shown
    total_width = group_name_width + success_count_width + success_pct_width + failure_count_width + failure_pct_width + 5  # base spacing
    if show_scores:
        total_width += avg_score_width + avg_score_fail_width + 2
    if show_duration:
        total_width += duration_width + 1
        if show_metric_stddev:
            total_width += duration_stddev_width + 1
    if show_wrong_objects:
        total_width += wrong_objects_width + 1
        if show_metric_stddev and not csv:
            total_width += wrong_objects_names_width + 1
    if show_metrics:
        total_width += ee_sparc_width + path_length_width + avg_speed_width + 3
        if show_metric_stddev:
            total_width += ee_sparc_stddev_width + path_length_stddev_width + avg_speed_stddev_width + 3
    if show_timing:
        total_width += it_per_sec_width + wall_total_width + 2
    if show_eps:
        total_width += success_eps_width + 1

    # Separator for CSV or space-aligned output
    sep = "," if csv else " "

    # Build header row dynamically
    if csv:
        header_parts = [
            group_name_header,
            "Success",
            "Success %",
            "Total"
        ]
    else:
        header_parts = [
            f"{group_name_header:<{group_name_width}}",
            f"{'Success':<{success_count_width}}",
            f"{'  %':<{success_pct_width}}"
        ]
    if show_scores:
        if csv:
            header_parts.append("Score(fail)")
        else:
            header_parts.append(f"{'Score(fail)':<{avg_score_fail_width}}")
    if show_duration:
        if csv:
            if csv_compact:
                header_parts.append("Time(s)")  # Compact: value (± stddev) in one column
            else:
                header_parts.append("Time(s)")
                header_parts.append("Time σ")  # Separate stddev column
        else:
            header_parts.append(f"{'Time(s)':<{duration_width}}")
            if show_metric_stddev:
                header_parts.append(f"{'Time σ':<{duration_stddev_width}}")
    if show_wrong_objects:
        if csv:
            header_parts.append("WrongObj Total")
            header_parts.append("WrongObj Succ")
            header_parts.append("WrongObj Fail")
        else:
            header_parts.append(f"{'WrongObj(T/S/F)':<{wrong_objects_width}}")
    if show_metrics:
        if csv:
            if csv_compact:
                # Compact: value (± stddev) in one column
                header_parts.append("EE SPARC")
                header_parts.append("PathLen(m)")
                header_parts.append("Speed(cm/s)")
            else:
                header_parts.append("EE SPARC")
                header_parts.append("SPARC σ")  # Separate stddev column
                header_parts.append("PathLen(m)")
                header_parts.append("Path σ")  # Separate stddev column
                header_parts.append("Speed(cm/s)")
                header_parts.append("Speed σ")  # Separate stddev column
        else:
            header_parts.append(f"{'EE SPARC':<{ee_sparc_width}}")
            if show_metric_stddev:
                header_parts.append(f"{'SPARC σ':<{ee_sparc_stddev_width}}")
            header_parts.append(f"{'PathLen(m)':<{path_length_width}}")
            if show_metric_stddev:
                header_parts.append(f"{'Path σ':<{path_length_stddev_width}}")
            header_parts.append(f"{'Speed(cm/s)':<{avg_speed_width}}")
            if show_metric_stddev:
                header_parts.append(f"{'Speed σ':<{avg_speed_stddev_width}}")
    if show_timing:
        if csv:
            header_parts.append("it/s")
            header_parts.append("Walltime(m)")
        else:
            header_parts.append(f"{'it/s':<{it_per_sec_width}}")
            header_parts.append(f"{'Walltime(m)':<{wall_total_width}}")
    if show_eps and not csv:
        header_parts.append(f"{'Eps(succ)':<{success_eps_width}}")
    # WrongObjNames at the end
    if show_wrong_objects:
        if csv:
            header_parts.append("WrongObjNames")
        elif show_metric_stddev:
            header_parts.append(f"{'WrongObjNames':<{wrong_objects_names_width}}")

    header = sep.join(header_parts)

    # Print total summary row
    total_num_success, total_num_failure, total_num_total, total_success_rate, total_failure_rate, total_success_runs, total_failure_runs = get_success_stats(results)
    total_success_count_str = f"{total_num_success}/{total_num_total}"
    total_success_pct_str = f"{total_success_rate*100:.1f}%"
    total_success_pct_csv_str = f"{total_success_rate*100:.1f}"  # No % for CSV
    total_failure_count_str = f"{total_num_failure}/{total_num_total}"
    total_failure_pct_str = f"{total_failure_rate*100:.1f}%"

    # Determine the count label
    count_label = "tasks" if group_by == "task" else "attrs"
    num_groups = len(grouped_data)

    if csv:
        total_row_parts = [
            f"TOTAL ({num_groups} {count_label})",
            str(total_num_success),
            total_success_pct_csv_str,
            str(total_num_total)
        ]
    else:
        total_row_parts = [
            f"{BOLD}{f'TOTAL ({num_groups} {count_label})':<{group_name_width}}{RESET}",
            f"{GREEN}{total_success_count_str:<{success_count_width}}{RESET}",
            f"{GREEN}{total_success_pct_str:<{success_pct_width}}{RESET}"
        ]

    if show_scores:
        total_avg_score_fail = get_avg_score(episode_results, fail_only=True)
        total_avg_score_fail_str = format_score(total_avg_score_fail)
        if csv:
            total_row_parts.append(total_avg_score_fail_str)
        else:
            total_row_parts.append(f"{total_avg_score_fail_str:<{avg_score_fail_width}}")

    if show_duration:
        # Calculate average duration across all successful episodes
        valid_durations = []
        for ep in episode_results:
            # Only count successful episodes
            if not ep.get("success"):
                continue

            # First check if duration is available directly
            duration = ep.get("duration")
            if duration is not None:
                valid_durations.append(duration)
            else:
                # Fall back to calculating from episode_step * dt
                episode_step = ep.get("episode_step")
                if episode_step is not None:
                    # Get dt for this episode's task
                    ep_dt = ep.get("dt")
                    if ep_dt is None:
                        ep_dt = dt

                    if ep_dt is not None:
                        valid_durations.append(episode_step * ep_dt)

        if len(valid_durations) > 0:
            total_avg_duration = sum(valid_durations) / len(valid_durations)
            total_duration_str = f"{total_avg_duration:.2f}"
            # Calculate stddev
            if len(valid_durations) > 1:
                total_stddev_duration = statistics.stdev(valid_durations)
                total_stddev_str = f"{total_stddev_duration:.2f}"
            else:
                total_stddev_str = "-"
        else:
            total_duration_str = "-"
            total_stddev_str = "-"
        if csv:
            if csv_compact:
                total_row_parts.append(format_compact_value(total_duration_str, total_stddev_str))
            else:
                total_row_parts.append(total_duration_str)
                total_row_parts.append(total_stddev_str)  # Separate stddev column
        else:
            total_row_parts.append(f"{total_duration_str:<{duration_width}}")
            if show_metric_stddev:
                total_row_parts.append(f"{total_stddev_str:<{duration_stddev_width}}")

    if show_wrong_objects:
        # Calculate total wrong object stats across all episodes (split by success/failure)
        total_wrong_obj_stats = get_wrong_object_stats(episode_results, exclude_containers=exclude_containers)
        total_wrong_objects_str = format_wrong_object_str(total_wrong_obj_stats, num_episodes=total_num_total, csv=csv, show_objects=False, split_by_success=True)
        if csv:
            total_row_parts.append(str(total_wrong_obj_stats.get('count', 0)))
            total_row_parts.append(str(total_wrong_obj_stats.get('count_success', 0)))
            total_row_parts.append(str(total_wrong_obj_stats.get('count_failure', 0)))
        else:
            total_row_parts.append(f"{total_wrong_objects_str:<{wrong_objects_width}}")

    if show_metrics:
        # Calculate total metrics averages across all episodes
        sparc_values = [float(_get_metric(ep, 'ee_sparc')) for ep in episode_results if _get_metric(ep, 'ee_sparc') is not None and math.isfinite(float(_get_metric(ep, 'ee_sparc')))]
        if sparc_values:
            total_avg_sparc = sum(sparc_values) / len(sparc_values)
            total_ee_sparc_str = f"{total_avg_sparc:.2f}"
            if len(sparc_values) > 1:
                total_sparc_stddev = statistics.stdev(sparc_values)
                total_ee_sparc_stddev_str = f"{total_sparc_stddev:.2f}"
            else:
                total_ee_sparc_stddev_str = "-"
        else:
            total_ee_sparc_str = "-"
            total_ee_sparc_stddev_str = "-"

        path_values = [float(_get_metric(ep, 'ee_path_length')) for ep in episode_results if _get_metric(ep, 'ee_path_length') is not None and math.isfinite(float(_get_metric(ep, 'ee_path_length')))]
        if path_values:
            total_avg_path = sum(path_values) / len(path_values)
            total_path_length_str = f"{total_avg_path:.2f}"
            if len(path_values) > 1:
                total_path_stddev = statistics.stdev(path_values)
                total_path_length_stddev_str = f"{total_path_stddev:.2f}"
            else:
                total_path_length_stddev_str = "-"
        else:
            total_path_length_str = "-"
            total_path_length_stddev_str = "-"

        # Convert m/s to cm/s
        speed_values = [float(_get_metric(ep, 'ee_speed_mean')) * 100 for ep in episode_results if _get_metric(ep, 'ee_speed_mean') is not None and math.isfinite(float(_get_metric(ep, 'ee_speed_mean')))]
        if speed_values:
            total_avg_speed = sum(speed_values) / len(speed_values)
            total_avg_speed_str = f"{total_avg_speed:.1f}"
            if len(speed_values) > 1:
                total_speed_stddev = statistics.stdev(speed_values)
                total_avg_speed_stddev_str = f"{total_speed_stddev:.1f}"
            else:
                total_avg_speed_stddev_str = "-"
        else:
            total_avg_speed_str = "-"
            total_avg_speed_stddev_str = "-"

        if csv:
            if csv_compact:
                total_row_parts.append(format_compact_value(total_ee_sparc_str, total_ee_sparc_stddev_str))
                total_row_parts.append(format_compact_value(total_path_length_str, total_path_length_stddev_str))
                total_row_parts.append(format_compact_value(total_avg_speed_str, total_avg_speed_stddev_str))
            else:
                total_row_parts.append(total_ee_sparc_str)
                total_row_parts.append(total_ee_sparc_stddev_str)  # Separate stddev column
                total_row_parts.append(total_path_length_str)
                total_row_parts.append(total_path_length_stddev_str)  # Separate stddev column
                total_row_parts.append(total_avg_speed_str)
                total_row_parts.append(total_avg_speed_stddev_str)  # Separate stddev column
        else:
            total_row_parts.append(f"{total_ee_sparc_str:<{ee_sparc_width}}")
            if show_metric_stddev:
                total_row_parts.append(f"{total_ee_sparc_stddev_str:<{ee_sparc_stddev_width}}")
            total_row_parts.append(f"{total_path_length_str:<{path_length_width}}")
            if show_metric_stddev:
                total_row_parts.append(f"{total_path_length_stddev_str:<{path_length_stddev_width}}")
            total_row_parts.append(f"{total_avg_speed_str:<{avg_speed_width}}")
            if show_metric_stddev:
                total_row_parts.append(f"{total_avg_speed_stddev_str:<{avg_speed_stddev_width}}")

    if show_timing:
        # Calculate total timing averages
        all_it = [float(_get_timing(ep, 'it_per_sec')) for ep in episode_results if _get_timing(ep, 'it_per_sec') is not None]
        all_wall = [float(_get_timing(ep, 'wall_total_s')) for ep in episode_results if _get_timing(ep, 'wall_total_s') is not None]
        total_it_str = f"{sum(all_it)/len(all_it):.2f}" if all_it else "-"
        total_wall_str = f"{sum(all_wall)/len(all_wall)/60:.1f}" if all_wall else "-"
        if csv:
            total_row_parts.append(total_it_str)
            total_row_parts.append(total_wall_str)
        else:
            total_row_parts.append(f"{total_it_str:<{it_per_sec_width}}")
            total_row_parts.append(f"{total_wall_str:<{wall_total_width}}")

    if show_eps and not csv:
        total_row_parts.append(f"{'-':<{success_eps_width}}")

    # WrongObjNames at the end - empty for TOTAL row
    if show_wrong_objects:
        if csv:
            total_row_parts.append('""')  # Empty quoted string for CSV
        elif show_metric_stddev:
            total_row_parts.append(f"{'':<{wrong_objects_names_width}}")

    total_line = sep.join(total_row_parts)

    # Print data rows
    table_lines = []
    for (group_name, success_count_str, success_pct_str, success_pct_csv_str, failure_count_str, failure_pct_str,
         avg_score_str, avg_score_fail_str, success_eps_str, duration_str, duration_stddev_str,
         wrong_objects_str, wrong_obj_total, wrong_obj_success, wrong_obj_fail, wrong_objects_names_str,
         ee_sparc_str, ee_sparc_stddev_str, path_length_str, path_length_stddev_str,
         avg_speed_str, avg_speed_stddev_str, it_per_sec_str, wall_total_str, num_success, num_failure, num_total) in grouped_data:
        if csv:
            row_parts = [
                group_name,
                str(num_success),
                success_pct_csv_str,
                str(num_total)
            ]
        else:
            row_parts = [
                f"{group_name:<{group_name_width}}",
                f"{GREEN}{success_count_str:<{success_count_width}}{RESET}",
                f"{GREEN}{success_pct_str:<{success_pct_width}}{RESET}"
            ]

        if show_scores:
            if csv:
                row_parts.append(avg_score_fail_str)
            else:
                row_parts.append(f"{avg_score_fail_str:<{avg_score_fail_width}}")

        if show_duration:
            if csv:
                if csv_compact:
                    row_parts.append(format_compact_value(duration_str, duration_stddev_str))
                else:
                    row_parts.append(duration_str)
                    row_parts.append(duration_stddev_str)  # Separate stddev column
            else:
                row_parts.append(f"{duration_str:<{duration_width}}")
                if show_metric_stddev:
                    row_parts.append(f"{duration_stddev_str:<{duration_stddev_width}}")

        if show_wrong_objects:
            if csv:
                row_parts.append(str(wrong_obj_total))
                row_parts.append(str(wrong_obj_success))
                row_parts.append(str(wrong_obj_fail))
            else:
                row_parts.append(f"{wrong_objects_str:<{wrong_objects_width}}")

        if show_metrics:
            if csv:
                if csv_compact:
                    row_parts.append(format_compact_value(ee_sparc_str, ee_sparc_stddev_str))
                    row_parts.append(format_compact_value(path_length_str, path_length_stddev_str))
                    row_parts.append(format_compact_value(avg_speed_str, avg_speed_stddev_str))
                else:
                    row_parts.append(ee_sparc_str)
                    row_parts.append(ee_sparc_stddev_str)  # Separate stddev column
                    row_parts.append(path_length_str)
                    row_parts.append(path_length_stddev_str)  # Separate stddev column
                    row_parts.append(avg_speed_str)
                    row_parts.append(avg_speed_stddev_str)  # Separate stddev column
            else:
                row_parts.append(f"{ee_sparc_str:<{ee_sparc_width}}")
                if show_metric_stddev:
                    row_parts.append(f"{ee_sparc_stddev_str:<{ee_sparc_stddev_width}}")
                row_parts.append(f"{path_length_str:<{path_length_width}}")
                if show_metric_stddev:
                    row_parts.append(f"{path_length_stddev_str:<{path_length_stddev_width}}")
                row_parts.append(f"{avg_speed_str:<{avg_speed_width}}")
                if show_metric_stddev:
                    row_parts.append(f"{avg_speed_stddev_str:<{avg_speed_stddev_width}}")

        if show_timing:
            if csv:
                row_parts.append(it_per_sec_str)
                row_parts.append(wall_total_str)
            else:
                row_parts.append(f"{it_per_sec_str:<{it_per_sec_width}}")
                row_parts.append(f"{wall_total_str:<{wall_total_width}}")

        if show_eps and not csv:
            row_parts.append(f"{success_eps_str:<{success_eps_width}}")

        # WrongObjNames at the end
        if show_wrong_objects:
            if csv:
                # Wrap in quotes since value contains commas
                row_parts.append(f'"{wrong_objects_names_str}"')
            elif show_metric_stddev:
                row_parts.append(f"{wrong_objects_names_str:<{wrong_objects_names_width}}")

        table_line = sep.join(row_parts)
        table_lines.append(table_line)

    table = "\n".join(table_lines)

    return header, total_line, table, total_width

def summarize_experiments_by_attributes(episode_results: list[dict] | str, VERBOSE=False, csv=False):
    """
    Summarize results for all tasks in an experiment by attributes.

    Groups episodes by their attributes and calculates success/failure statistics for each attribute.
    Note: Since episodes can have multiple attributes, the same episode may be counted in multiple groups.

    Args:
        episode_results: List of episode results or path to episode_results.json
        VERBOSE: If True, print detailed information for each attribute
    """
    # Load data from files or dicts
    if not isinstance(episode_results, list):
        episode_results = load_file(episode_results)

    # Check if episode_results is None or empty
    if episode_results is None or len(episode_results) == 0:
        print(f"No episode results found or file is empty.")
        return

    # Group episodes by attributes
    attribute_results = {}
    all_attributes = set()

    for episode in episode_results:
        attributes = episode.get("attributes", [])
        if attributes is None:
            print(f"Env {episode.get('env_name')} Episode {episode.get('episode')} has no attributes.")
            continue
        for attr in attributes:
            all_attributes.add(attr)
            if attr not in attribute_results:
                attribute_results[attr] = {"success": [], "failure": []}

            if episode.get("success"):
                attribute_results[attr]["success"].append(episode)
            else:
                attribute_results[attr]["failure"].append(episode)

    # Sort attributes with priority for difficulty levels
    priority_attrs = ['simple', 'moderate', 'complex']

    # Separate into priority and other attributes
    priority_list = [attr for attr in priority_attrs if attr in all_attributes]
    other_list = sorted([attr for attr in all_attributes if attr not in priority_attrs])

    # Combine them with priority attributes first
    sorted_attributes = priority_list + other_list

    if sorted_attributes is None or len(sorted_attributes) == 0:
        print(f"No attributes found in episode results.")
        return

    if VERBOSE:
        # Calculate overall stats
        num_success = sum(1 for ep in episode_results if ep.get("success"))
        num_failure = sum(1 for ep in episode_results if not ep.get("success"))
        num_total = len(episode_results)
        success_rate = num_success / num_total if num_total > 0 else 0
        failure_rate = num_failure / num_total if num_total > 0 else 0
        avg_score = get_avg_score(episode_results)
        avg_score_str = format_score(avg_score)

        print(f"-------------EXPERIMENT SUMMARY BY ATTRIBUTES-------------------")
        print(f"Total episodes: {num_total}, Unique attributes: {len(sorted_attributes)}")
        print(f"{GREEN}Success {num_success}/{num_total} ({success_rate*100:.2f}%){RESET}, {RED}Failure {num_failure}/{num_total} ({failure_rate*100:.2f}%){RESET} avg score: {avg_score_str}")
        print(f"----------------------------------------------------------------")

        for attr in sorted_attributes:
            attr_data = attribute_results[attr]
            num_success, num_failure, num_total, success_rate, failure_rate, success_runs, failure_runs = get_success_stats(attr_data)

            # Calculate average scores for episodes with this attribute
            episodes_with_attr = success_runs + failure_runs
            avg_score = get_avg_score(episodes_with_attr)
            avg_score_fail = get_avg_score([ep for ep in episodes_with_attr if not ep.get("success")], fail_only=False)
            avg_score_str = format_score(avg_score)
            avg_score_fail_str = format_score(avg_score_fail)

            # Use get_grouped_result_table_str to display tasks under this attribute
            header, total_line, table, total_width = get_grouped_result_table_str(episodes_with_attr, group_by="task", show_scores=True, show_eps=True, show_duration=True)

            print(format_centered_header(attr, total_width))
            print(header)
            print("-" * total_width)

            print(total_line)
            print(table)
        print("-" * total_width)
    else:
        print_result_table(
            episode_results,
            group_by="attributes",
            title="EXPERIMENT SUMMARY BY ATTRIBUTES",
            show_scores=True,
            show_eps=False,
            show_duration=True,
            csv=csv,
        )

def summarize_experiments_by_category_with_attributes(episode_results: list[dict] | str,
                                                       remap: dict[str, str] = None,
                                                       VERBOSE=False,
                                                       csv=False,
                                                       show_metrics=True,
                                                       csv_compact=False):
    """
    Summarize results in one giant table with categories as section headers
    and attributes listed under each category.

    Output format:
        Category
        VISUAL                  X/Y   Z%   ...
          color                 X/Y   Z%   ...
          size                  X/Y   Z%   ...
          semantics             X/Y   Z%   ...
        RELATIONAL              X/Y   Z%   ...
          spatial               X/Y   Z%   ...
          ...

    Args:
        episode_results: List of episode results or path to episode_results.json
        remap: Dictionary mapping attribute names to category names.
               If None, uses BENCHMARK_TASK_CATEGORIES from paths.py
        VERBOSE: If True, print detailed information (stddev columns, wrong objects)
        csv: If True, output in CSV format
        show_metrics: If True, show trajectory metrics columns (EE SPARC, PathLen, Speed)
        csv_compact: If True, combine value and stddev into single column like '-9.14 (± 4.72)'
    """
    if remap is None:
        remap = BENCHMARK_TASK_CATEGORIES

    # Load data from files or dicts
    if not isinstance(episode_results, list):
        episode_results = load_file(episode_results)

    # Check if episode_results is None or empty
    if episode_results is None or len(episode_results) == 0:
        print("No episode results found or file is empty.")
        return

    # Build category -> attributes mapping and gather stats
    category_to_attrs = {}
    for attr, cat in remap.items():
        if cat not in category_to_attrs:
            category_to_attrs[cat] = []
        category_to_attrs[cat].append(attr)

    # Group episodes by attributes
    attribute_results = get_attribute_grouped_results(episode_results)

    # Group episodes by remapped categories
    category_results = {}
    for episode in episode_results:
        attributes = episode.get("attributes", [])
        if attributes is None:
            continue

        episode_categories = set()
        for attr in attributes:
            if attr in remap:
                episode_categories.add(remap[attr])

        for category in episode_categories:
            if category not in category_results:
                category_results[category] = {"success": [], "failure": []}

            if episode.get("success"):
                category_results[category]["success"].append(episode)
            else:
                category_results[category]["failure"].append(episode)

    # Settings based on VERBOSE
    show_metric_stddev = VERBOSE
    show_wrong_objects = VERBOSE

    # Calculate column widths
    name_width = max(len("Attribute"), max(len(cat) for cat in category_to_attrs.keys()), max(len(attr) for attr in remap.keys())) + 2
    success_count_width = len("000/000 ")
    success_pct_width = len("100.0% ")
    score_fail_width = len("Score(fail) ")
    duration_width = len("Time(s) ")
    duration_stddev_width = len("Time σ ") if show_metric_stddev else 0
    wrong_objects_width = len("WrongObj(T/S/F) ") if show_wrong_objects else 0

    # Metrics column widths
    ee_sparc_width = len("EE SPARC ") if show_metrics else 0
    ee_sparc_stddev_width = len("SPARC σ ") if show_metrics and show_metric_stddev else 0
    path_length_width = len("PathLen(m) ") if show_metrics else 0
    path_length_stddev_width = len("Path σ ") if show_metrics and show_metric_stddev else 0
    avg_speed_width = len("Speed(cm/s) ") if show_metrics else 0
    avg_speed_stddev_width = len("Speed σ ") if show_metrics and show_metric_stddev else 0

    sep = "," if csv else " "

    # Build header - matching the style of get_grouped_result_table_str
    if csv:
        if csv_compact:
            # Compact: value (± stddev) in one column
            header_parts = ["Category/Attribute", "Success", "Success %", "Total", "Score(fail)", "Time(s)"]
            if show_wrong_objects:
                header_parts.extend(["WrongObj Total", "WrongObj Succ", "WrongObj Fail"])
            if show_metrics:
                header_parts.extend(["EE SPARC", "PathLen(m)", "Speed(cm/s)"])
        else:
            # Separate stddev columns
            header_parts = ["Category/Attribute", "Success", "Success %", "Total", "Score(fail)", "Time(s)", "Time σ"]
            if show_wrong_objects:
                header_parts.extend(["WrongObj Total", "WrongObj Succ", "WrongObj Fail"])
            if show_metrics:
                header_parts.extend(["EE SPARC", "SPARC σ", "PathLen(m)", "Path σ", "Speed(cm/s)", "Speed σ"])
    else:
        header_parts = [
            f"{'Category/Attribute':<{name_width}}",
            f"{'Success':<{success_count_width}}",
            f"{'  %':<{success_pct_width}}",
            f"{'Score(fail)':<{score_fail_width}}",
            f"{'Time(s)':<{duration_width}}"
        ]
        if show_metric_stddev:
            header_parts.append(f"{'Time σ':<{duration_stddev_width}}")
        if show_wrong_objects:
            header_parts.append(f"{'WrongObj(T/S/F)':<{wrong_objects_width}}")
        if show_metrics:
            header_parts.append(f"{'EE SPARC':<{ee_sparc_width}}")
            if show_metric_stddev:
                header_parts.append(f"{'SPARC σ':<{ee_sparc_stddev_width}}")
            header_parts.append(f"{'PathLen(m)':<{path_length_width}}")
            if show_metric_stddev:
                header_parts.append(f"{'Path σ':<{path_length_stddev_width}}")
            header_parts.append(f"{'Speed(cm/s)':<{avg_speed_width}}")
            if show_metric_stddev:
                header_parts.append(f"{'Speed σ':<{avg_speed_stddev_width}}")
    header = sep.join(header_parts)

    # Calculate total width
    total_width = name_width + success_count_width + success_pct_width + score_fail_width + duration_width + 5
    if show_metric_stddev:
        total_width += duration_stddev_width
    if show_wrong_objects:
        total_width += wrong_objects_width
    if show_metrics:
        total_width += ee_sparc_width + path_length_width + avg_speed_width + 3
        if show_metric_stddev:
            total_width += ee_sparc_stddev_width + path_length_stddev_width + avg_speed_stddev_width + 3

    # Calculate overall totals
    total_num_success = sum(1 for ep in episode_results if ep.get("success"))
    total_num_total = len(episode_results)
    total_success_rate = total_num_success / total_num_total if total_num_total > 0 else 0
    total_avg_score_fail = get_avg_score(episode_results, fail_only=True)
    total_avg_score_fail_str = format_score(total_avg_score_fail)

    # Calculate total duration
    total_valid_durations = []
    for ep in episode_results:
        if not ep.get("success"):
            continue
        duration = ep.get("duration")
        if duration is not None:
            total_valid_durations.append(duration)
        else:
            episode_step = ep.get("episode_step")
            ep_dt = ep.get("dt")
            if episode_step is not None and ep_dt is not None:
                total_valid_durations.append(episode_step * ep_dt)

    if len(total_valid_durations) > 0:
        total_avg_duration = sum(total_valid_durations) / len(total_valid_durations)
        total_duration_str = f"{total_avg_duration:.2f}"
        if len(total_valid_durations) > 1:
            total_stddev_str = f"{statistics.stdev(total_valid_durations):.2f}"
        else:
            total_stddev_str = "-"
    else:
        total_duration_str = "-"
        total_stddev_str = "-"

    # Calculate total wrong object stats
    if show_wrong_objects:
        total_wrong_obj_stats = get_wrong_object_stats(episode_results)
        total_wrong_objects_str = format_wrong_object_str(total_wrong_obj_stats, num_episodes=total_num_total, csv=csv, show_objects=False, split_by_success=True)

    # Calculate total metrics
    if show_metrics:
        # EE SPARC
        sparc_values = [float(_get_metric(ep, 'ee_sparc')) for ep in episode_results if _get_metric(ep, 'ee_sparc') is not None and math.isfinite(float(_get_metric(ep, 'ee_sparc')))]
        if sparc_values:
            total_ee_sparc_str = f"{sum(sparc_values) / len(sparc_values):.2f}"
            total_ee_sparc_stddev_str = f"{statistics.stdev(sparc_values):.2f}" if len(sparc_values) > 1 else "-"
        else:
            total_ee_sparc_str = "-"
            total_ee_sparc_stddev_str = "-"

        # Path Length
        path_values = [float(_get_metric(ep, 'ee_path_length')) for ep in episode_results if _get_metric(ep, 'ee_path_length') is not None and math.isfinite(float(_get_metric(ep, 'ee_path_length')))]
        if path_values:
            total_path_length_str = f"{sum(path_values) / len(path_values):.2f}"
            total_path_length_stddev_str = f"{statistics.stdev(path_values):.2f}" if len(path_values) > 1 else "-"
        else:
            total_path_length_str = "-"
            total_path_length_stddev_str = "-"

        # Average Speed (convert m/s to cm/s)
        speed_values = [float(_get_metric(ep, 'ee_speed_mean')) * 100 for ep in episode_results if _get_metric(ep, 'ee_speed_mean') is not None and math.isfinite(float(_get_metric(ep, 'ee_speed_mean')))]
        if speed_values:
            total_avg_speed_str = f"{sum(speed_values) / len(speed_values):.1f}"
            total_avg_speed_stddev_str = f"{statistics.stdev(speed_values):.1f}" if len(speed_values) > 1 else "-"
        else:
            total_avg_speed_str = "-"
            total_avg_speed_stddev_str = "-"

    # Build total row
    if csv:
        if csv_compact:
            # Compact: value (± stddev) in one column
            total_row_parts = [
                "TOTAL",
                str(total_num_success),
                f"{total_success_rate*100:.1f}",
                str(total_num_total),
                total_avg_score_fail_str,
                format_compact_value(total_duration_str, total_stddev_str)
            ]
            if show_wrong_objects:
                total_row_parts.append(str(total_wrong_obj_stats.get('count', 0)))
                total_row_parts.append(str(total_wrong_obj_stats.get('count_success', 0)))
                total_row_parts.append(str(total_wrong_obj_stats.get('count_failure', 0)))
            if show_metrics:
                total_row_parts.extend([
                    format_compact_value(total_ee_sparc_str, total_ee_sparc_stddev_str),
                    format_compact_value(total_path_length_str, total_path_length_stddev_str),
                    format_compact_value(total_avg_speed_str, total_avg_speed_stddev_str)
                ])
        else:
            # Separate stddev columns
            total_row_parts = [
                "TOTAL",
                str(total_num_success),
                f"{total_success_rate*100:.1f}",
                str(total_num_total),
                total_avg_score_fail_str,
                total_duration_str,
                total_stddev_str
            ]
            if show_wrong_objects:
                total_row_parts.append(str(total_wrong_obj_stats.get('count', 0)))
                total_row_parts.append(str(total_wrong_obj_stats.get('count_success', 0)))
                total_row_parts.append(str(total_wrong_obj_stats.get('count_failure', 0)))
            if show_metrics:
                total_row_parts.extend([
                    total_ee_sparc_str,
                    total_ee_sparc_stddev_str,
                    total_path_length_str,
                    total_path_length_stddev_str,
                    total_avg_speed_str,
                    total_avg_speed_stddev_str
                ])
    else:
        total_row_parts = [
            f"{BOLD}{'TOTAL':<{name_width}}{RESET}",
            f"{GREEN}{total_num_success}/{total_num_total:<{success_count_width-4}}{RESET}",
            f"{GREEN}{total_success_rate*100:.1f}%{'':<{success_pct_width-6}}{RESET}",
            f"{total_avg_score_fail_str:<{score_fail_width}}",
            f"{total_duration_str:<{duration_width}}"
        ]
        if show_metric_stddev:
            total_row_parts.append(f"{total_stddev_str:<{duration_stddev_width}}")
        if show_wrong_objects:
            total_row_parts.append(f"{total_wrong_objects_str:<{wrong_objects_width}}")
        if show_metrics:
            total_row_parts.append(f"{total_ee_sparc_str:<{ee_sparc_width}}")
            if show_metric_stddev:
                total_row_parts.append(f"{total_ee_sparc_stddev_str:<{ee_sparc_stddev_width}}")
            total_row_parts.append(f"{total_path_length_str:<{path_length_width}}")
            if show_metric_stddev:
                total_row_parts.append(f"{total_path_length_stddev_str:<{path_length_stddev_width}}")
            total_row_parts.append(f"{total_avg_speed_str:<{avg_speed_width}}")
            if show_metric_stddev:
                total_row_parts.append(f"{total_avg_speed_stddev_str:<{avg_speed_stddev_width}}")
    total_line = sep.join(total_row_parts)

    # Print header and total
    if not csv:
        print(format_centered_header("EXPERIMENT SUMMARY BY CATEGORY WITH ATTRIBUTES", total_width))
    print(header)
    if not csv:
        print("-" * total_width)
    print(total_line)
    if not csv:
        print("-" * total_width)

    # Helper function to build a row
    def build_row(name: str, episodes: list[dict], indent: str = "", is_category: bool = False) -> str:
        num_success = sum(1 for ep in episodes if ep.get("success"))
        num_total = len(episodes)
        success_rate = num_success / num_total if num_total > 0 else 0

        avg_score_fail = get_avg_score([ep for ep in episodes if not ep.get("success")], fail_only=False)
        avg_score_fail_str = format_score(avg_score_fail)

        # Calculate duration
        valid_durations = []
        for ep in episodes:
            if not ep.get("success"):
                continue
            duration = ep.get("duration")
            if duration is not None:
                valid_durations.append(duration)
            else:
                episode_step = ep.get("episode_step")
                ep_dt = ep.get("dt")
                if episode_step is not None and ep_dt is not None:
                    valid_durations.append(episode_step * ep_dt)

        if len(valid_durations) > 0:
            avg_duration = sum(valid_durations) / len(valid_durations)
            duration_str = f"{avg_duration:.2f}"
            if len(valid_durations) > 1:
                stddev_str = f"{statistics.stdev(valid_durations):.2f}"
            else:
                stddev_str = "-"
        else:
            duration_str = "-"
            stddev_str = "-"

        # Calculate wrong object stats
        if show_wrong_objects:
            wrong_obj_stats = get_wrong_object_stats(episodes)
            wrong_objects_str = format_wrong_object_str(wrong_obj_stats, num_episodes=num_total, csv=csv, show_objects=False, split_by_success=True)

        # Calculate metrics
        if show_metrics:
            # EE SPARC
            sparc_values = [float(_get_metric(ep, 'ee_sparc')) for ep in episodes if _get_metric(ep, 'ee_sparc') is not None and math.isfinite(float(_get_metric(ep, 'ee_sparc')))]
            if sparc_values:
                ee_sparc_str = f"{sum(sparc_values) / len(sparc_values):.2f}"
                ee_sparc_stddev_str = f"{statistics.stdev(sparc_values):.2f}" if len(sparc_values) > 1 else "-"
            else:
                ee_sparc_str = "-"
                ee_sparc_stddev_str = "-"

            # Path Length
            path_values = [float(_get_metric(ep, 'ee_path_length')) for ep in episodes if _get_metric(ep, 'ee_path_length') is not None and math.isfinite(float(_get_metric(ep, 'ee_path_length')))]
            if path_values:
                path_length_str = f"{sum(path_values) / len(path_values):.2f}"
                path_length_stddev_str = f"{statistics.stdev(path_values):.2f}" if len(path_values) > 1 else "-"
            else:
                path_length_str = "-"
                path_length_stddev_str = "-"

            # Average Speed (convert m/s to cm/s)
            speed_values = [float(_get_metric(ep, 'ee_speed_mean')) * 100 for ep in episodes if _get_metric(ep, 'ee_speed_mean') is not None and math.isfinite(float(_get_metric(ep, 'ee_speed_mean')))]
            if speed_values:
                avg_speed_str = f"{sum(speed_values) / len(speed_values):.1f}"
                avg_speed_stddev_str = f"{statistics.stdev(speed_values):.1f}" if len(speed_values) > 1 else "-"
            else:
                avg_speed_str = "-"
                avg_speed_stddev_str = "-"

        display_name = f"{indent}{name}"

        if csv:
            if csv_compact:
                # Compact: value (± stddev) in one column
                row_parts = [
                    display_name,
                    str(num_success),
                    f"{success_rate*100:.1f}",
                    str(num_total),
                    avg_score_fail_str,
                    format_compact_value(duration_str, stddev_str)
                ]
                if show_wrong_objects:
                    row_parts.append(str(wrong_obj_stats.get('count', 0)))
                    row_parts.append(str(wrong_obj_stats.get('count_success', 0)))
                    row_parts.append(str(wrong_obj_stats.get('count_failure', 0)))
                if show_metrics:
                    row_parts.extend([
                        format_compact_value(ee_sparc_str, ee_sparc_stddev_str),
                        format_compact_value(path_length_str, path_length_stddev_str),
                        format_compact_value(avg_speed_str, avg_speed_stddev_str)
                    ])
            else:
                # Separate stddev columns
                row_parts = [
                    display_name,
                    str(num_success),
                    f"{success_rate*100:.1f}",
                    str(num_total),
                    avg_score_fail_str,
                    duration_str,
                    stddev_str
                ]
                if show_wrong_objects:
                    row_parts.append(str(wrong_obj_stats.get('count', 0)))
                    row_parts.append(str(wrong_obj_stats.get('count_success', 0)))
                    row_parts.append(str(wrong_obj_stats.get('count_failure', 0)))
                if show_metrics:
                    row_parts.extend([
                        ee_sparc_str,
                        ee_sparc_stddev_str,
                        path_length_str,
                        path_length_stddev_str,
                        avg_speed_str,
                        avg_speed_stddev_str
                    ])
        else:
            if is_category:
                row_parts = [
                    f"{BOLD}{display_name:<{name_width}}{RESET}",
                    f"{GREEN}{num_success}/{num_total:<{success_count_width-4}}{RESET}",
                    f"{GREEN}{success_rate*100:.1f}%{'':<{success_pct_width-6}}{RESET}",
                    f"{avg_score_fail_str:<{score_fail_width}}",
                    f"{duration_str:<{duration_width}}"
                ]
            else:
                row_parts = [
                    f"{display_name:<{name_width}}",
                    f"{GREEN}{num_success}/{num_total:<{success_count_width-4}}{RESET}",
                    f"{GREEN}{success_rate*100:.1f}%{'':<{success_pct_width-6}}{RESET}",
                    f"{avg_score_fail_str:<{score_fail_width}}",
                    f"{duration_str:<{duration_width}}"
                ]
            if show_metric_stddev:
                row_parts.append(f"{stddev_str:<{duration_stddev_width}}")
            if show_wrong_objects:
                row_parts.append(f"{wrong_objects_str:<{wrong_objects_width}}")
            if show_metrics:
                row_parts.append(f"{ee_sparc_str:<{ee_sparc_width}}")
                if show_metric_stddev:
                    row_parts.append(f"{ee_sparc_stddev_str:<{ee_sparc_stddev_width}}")
                row_parts.append(f"{path_length_str:<{path_length_width}}")
                if show_metric_stddev:
                    row_parts.append(f"{path_length_stddev_str:<{path_length_stddev_width}}")
                row_parts.append(f"{avg_speed_str:<{avg_speed_width}}")
                if show_metric_stddev:
                    row_parts.append(f"{avg_speed_stddev_str:<{avg_speed_stddev_width}}")
        return sep.join(row_parts)

    # Sort categories alphabetically
    sorted_categories = sorted(category_to_attrs.keys())

    # Print each category with its attributes
    for category in sorted_categories:
        if category not in category_results:
            continue

        cat_episodes = category_results[category]["success"] + category_results[category]["failure"]

        # Print category row (bold, uppercase)
        print(build_row(category.upper(), cat_episodes, indent="", is_category=True))

        # Print each attribute under this category
        attrs = sorted(category_to_attrs[category])
        for attr in attrs:
            if attr in attribute_results:
                attr_episodes = attribute_results[attr]["success"] + attribute_results[attr]["failure"]
                print(build_row(attr, attr_episodes, indent="  ", is_category=False))

    if not csv:
        print("-" * total_width)


def summarize_experiments_by_remapped_attributes(episode_results: list[dict] | str,
                                                  remap: dict[str, str] = None,
                                                  VERBOSE=False,
                                                  csv=False,
                                                  csv_compact=False):
    """
    Summarize results for all tasks in an experiment by remapped attribute categories.

    Maps fine-grained attributes to higher-level categories using the provided remap dict,
    then groups episodes by these categories and calculates success/failure statistics.

    Args:
        episode_results: List of episode results or path to episode_results.json
        remap: Dictionary mapping attribute names to category names.
               If None, uses BENCHMARK_TASK_CATEGORIES from paths.py
        VERBOSE: If True, print detailed information for each category
        csv: If True, output in CSV format
        csv_compact: If True, combine value and stddev into single column like '-9.14 (± 4.72)'
    """
    if remap is None:
        remap = BENCHMARK_TASK_CATEGORIES

    # Load data from files or dicts
    if not isinstance(episode_results, list):
        episode_results = load_file(episode_results)

    # Check if episode_results is None or empty
    if episode_results is None or len(episode_results) == 0:
        print(f"No episode results found or file is empty.")
        return

    # Group episodes by remapped categories
    category_results = {}
    all_categories = set()
    unmapped_attributes = set()

    for episode in episode_results:
        attributes = episode.get("attributes", [])
        if attributes is None:
            continue

        # Map each attribute to its category
        episode_categories = set()
        for attr in attributes:
            if attr in remap:
                episode_categories.add(remap[attr])
            else:
                unmapped_attributes.add(attr)

        # Add episode to each category it belongs to
        for category in episode_categories:
            all_categories.add(category)
            if category not in category_results:
                category_results[category] = {"success": [], "failure": []}

            if episode.get("success"):
                category_results[category]["success"].append(episode)
            else:
                category_results[category]["failure"].append(episode)

    # Sort categories alphabetically
    sorted_categories = sorted(all_categories)

    if len(sorted_categories) == 0:
        print(f"No remapped categories found in episode results.")
        if unmapped_attributes:
            print(f"Unmapped attributes: {sorted(unmapped_attributes)}")
        return

    if VERBOSE:
        # Calculate overall stats
        num_success = sum(1 for ep in episode_results if ep.get("success"))
        num_failure = sum(1 for ep in episode_results if not ep.get("success"))
        num_total = len(episode_results)
        success_rate = num_success / num_total if num_total > 0 else 0
        avg_score = get_avg_score(episode_results)
        avg_score_str = format_score(avg_score)

        print(f"-------------EXPERIMENT SUMMARY BY REMAPPED CATEGORIES-------------------")
        print(f"Mapping: {remap}")
        print(f"Total episodes: {num_total}, Categories: {len(sorted_categories)}")
        print(f"{GREEN}Success {num_success}/{num_total} ({success_rate*100:.2f}%){RESET}, avg score: {avg_score_str}")
        if unmapped_attributes:
            print(f"Unmapped attributes (excluded): {sorted(unmapped_attributes)}")
        print(f"----------------------------------------------------------------")

        for category in sorted_categories:
            cat_data = category_results[category]
            num_success, num_failure, num_total, success_rate, failure_rate, success_runs, failure_runs = get_success_stats(cat_data)

            # Calculate average scores for episodes with this category
            episodes_with_cat = success_runs + failure_runs
            avg_score = get_avg_score(episodes_with_cat)
            avg_score_str = format_score(avg_score)

            # Find which original attributes map to this category
            original_attrs = [attr for attr, cat in remap.items() if cat == category]

            # Use get_grouped_result_table_str to display tasks under this category
            header, total_line, table, total_width = get_grouped_result_table_str(episodes_with_cat, group_by="task", show_scores=True, show_eps=True, show_duration=True, csv_compact=csv_compact)

            print(format_centered_header(f"{category} (from: {', '.join(original_attrs)})", total_width))
            print(header)
            print("-" * total_width)

            print(total_line)
            print(table)
        print("-" * total_width)
    else:
        # Create modified episode results with remapped attributes for table display
        remapped_episodes = []
        for episode in episode_results:
            attributes = episode.get("attributes", [])
            if attributes is None:
                continue

            # Get unique categories for this episode
            categories = set()
            for attr in attributes:
                if attr in remap:
                    categories.add(remap[attr])

            if categories:
                # Create a copy with remapped attributes
                ep_copy = episode.copy()
                ep_copy["attributes"] = list(categories)
                remapped_episodes.append(ep_copy)

        print_result_table(
            remapped_episodes,
            group_by="attributes",
            title="EXPERIMENT SUMMARY BY REMAPPED CATEGORIES",
            show_scores=True,
            show_eps=False,
            show_duration=True,
            csv=csv,
            csv_compact=csv_compact,
        )

        if unmapped_attributes:
            print(f"\nUnmapped attributes (excluded from above): {sorted(unmapped_attributes)}")

def load_task_to_scene_mapping() -> dict[str, str]:
    """
    Load task metadata and create a mapping from task names to scene names.

    Returns:
        Dictionary mapping task_name -> scene_name
    """
    metadata_path = os.path.join(TASK_DIR, "_metadata", "task_metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Task metadata file not found: {metadata_path}. "
            "Please run generate_task_metadata.py first to generate the metadata file."
        )

    with open(metadata_path, 'r') as f:
        tasks_data = json.load(f)

    task_to_scene = {}
    for task in tasks_data:
        task_name = task.get('task_name', '')
        scene = task.get('scene', '')
        if task_name:
            task_to_scene[task_name] = scene if scene else "(no scene)"

    return task_to_scene

def summarize_experiments_by_instruction_type(episode_results: list[dict] | str, VERBOSE=False, csv=False, csv_compact=False, show_metrics=True):
    """Summarize results comparing different instruction types. (Not yet implemented.)"""
    print("[RoboLab] summarize_experiments_by_instruction_type is not yet implemented.")


def summarize_experiments_by_scene(episode_results: list[dict] | str, VERBOSE=False, csv=False, csv_compact=False):
    """
    Summarize results for all tasks in an experiment by scene.

    Groups episodes by their scene (determined from task metadata) and calculates
    success/failure statistics for each scene.

    Args:
        episode_results: List of episode results or path to episode_results.json
        VERBOSE: If True, print detailed information for each scene
        csv: If True, output in CSV format
        csv_compact: If True, combine value and stddev into single column like '-9.14 (± 4.72)'
    """
    # Load data from files or dicts
    if not isinstance(episode_results, list):
        episode_results = load_file(episode_results)

    # Check if episode_results is None or empty
    if episode_results is None or len(episode_results) == 0:
        print(f"No episode results found or file is empty.")
        return

    # Load task-to-scene mapping from metadata
    try:
        task_to_scene = load_task_to_scene_mapping()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Group episodes by scene
    scene_results = {}
    all_scenes = set()
    unknown_scene_tasks = set()  # Track tasks with unknown scenes

    for episode in episode_results:
        env_name = episode.get("env_name", "")
        scene = task_to_scene.get(env_name, "(unknown scene)")
        if scene == "(unknown scene)":
            unknown_scene_tasks.add(env_name)
        all_scenes.add(scene)

        if scene not in scene_results:
            scene_results[scene] = {"success": [], "failure": []}

        if episode.get("success"):
            scene_results[scene]["success"].append(episode)
        else:
            scene_results[scene]["failure"].append(episode)

    # Sort scenes alphabetically, but put "(unknown scene)" last
    sorted_scenes = sorted([s for s in all_scenes if s != "(unknown scene)"])
    if "(unknown scene)" in all_scenes:
        sorted_scenes.append("(unknown scene)")

    if len(sorted_scenes) == 0:
        print(f"No scenes found in episode results.")
        return

    if VERBOSE:
        # Calculate overall stats
        num_success = sum(1 for ep in episode_results if ep.get("success"))
        num_failure = sum(1 for ep in episode_results if not ep.get("success"))
        num_total = len(episode_results)
        success_rate = num_success / num_total if num_total > 0 else 0
        failure_rate = num_failure / num_total if num_total > 0 else 0
        avg_score = get_avg_score(episode_results)
        avg_score_str = format_score(avg_score)

        print(f"-------------EXPERIMENT SUMMARY BY SCENE-------------------")
        print(f"Total episodes: {num_total}, Unique scenes: {len(sorted_scenes)}")
        print(f"{GREEN}Success {num_success}/{num_total} ({success_rate*100:.2f}%){RESET}, {RED}Failure {num_failure}/{num_total} ({failure_rate*100:.2f}%){RESET} avg score: {avg_score_str}")
        print(f"------------------------------------------------------------")

        for scene in sorted_scenes:
            scene_data = scene_results[scene]
            num_success, num_failure, num_total, success_rate, failure_rate, success_runs, failure_runs = get_success_stats(scene_data)

            # Calculate average scores for episodes in this scene
            episodes_in_scene = success_runs + failure_runs
            avg_score = get_avg_score(episodes_in_scene)
            avg_score_str = format_score(avg_score)

            # Use get_grouped_result_table_str to display tasks under this scene
            header, total_line, table, total_width = get_grouped_result_table_str(episodes_in_scene, group_by="task", show_scores=True, show_eps=True, show_duration=True, csv_compact=csv_compact)

            print(format_centered_header(scene, total_width))
            print(header)
            print("-" * total_width)

            print(total_line)
            print(table)
        print("-" * total_width)
    else:
        # Add scene info to episodes for grouping
        episode_results_with_scene = []
        for ep in episode_results:
            ep_copy = ep.copy()
            env_name = ep.get("env_name", "")
            ep_copy["scene"] = task_to_scene.get(env_name, "(unknown scene)")
            episode_results_with_scene.append(ep_copy)

        print_result_table(
            episode_results_with_scene,
            group_by="scene",
            title="EXPERIMENT SUMMARY BY SCENE",
            show_scores=True,
            show_eps=False,
            show_duration=True,
            csv=csv,
            csv_compact=csv_compact,
        )

    # Print unknown scene tasks at the end if any
    if unknown_scene_tasks:
        print(f"\n{BOLD}Warning:{RESET} The following {len(unknown_scene_tasks)} task(s) have unknown scenes (not found in task metadata):")
        for task in sorted(unknown_scene_tasks):
            print(f"  - {task}")

def summarize_experiment_results(episode_results: list[dict] | str, VERBOSE=False, csv=False, show_wrong_objects=False, exclude_containers=False, show_metrics=True, show_timing=False, csv_compact=False):
    """
    Summarize results for all tasks in an experiment.

    Args:
        results: Results dictionary or path to results.json
        episode_results: List of episode results or path to episode_results.json
        VERBOSE: If True, print detailed information for each task (shows stddev columns, wrong objects, episode IDs)
        csv: If True, output in CSV format for easy pasting into spreadsheets
        show_wrong_objects: If True, show wrong object grabbed column
        exclude_containers: If True, exclude container objects from wrong object counts
        show_metrics: If True, show trajectory metrics columns (EE SPARC, Path Length, Avg Speed)
        csv_compact: If True, combine value and stddev into single column like '-9.14 (± 4.72)'
    """
    # Load data from files or dicts
    # results = load_results_data(results)
    if not isinstance(episode_results, list):
        episode_results = load_file(episode_results)

    # Check if episode_results is None or empty
    if episode_results is None or len(episode_results) == 0:
        print(f"No episode results found or file is empty.")
        return

    print_result_table(
        episode_results,
        group_by="task",
        title="EXPERIMENT SUMMARY",
        show_scores=True,
        show_eps=VERBOSE,
        show_header=True,
        show_duration=True,
        show_wrong_objects=show_wrong_objects or VERBOSE,  # Show wrong objects in verbose mode
        exclude_containers=exclude_containers,
        show_metrics=show_metrics,
        show_metric_stddev=VERBOSE,  # Show stddev columns in verbose mode
        show_timing=show_timing,
        csv=csv,
        csv_compact=csv_compact,
    )

def summarize_task_results(episode_results: list[dict] | str, VERBOSE=False, csv=False, csv_compact=False):
    """
    Summarize results for tasks in the provided episode results.

    Args:
        episode_results: List of episode result dictionaries or path to episode_results.json
        VERBOSE: If True, print detailed information including individual episode results and subtask status
        csv: If True, output in CSV format for easy pasting into spreadsheets
        csv_compact: If True, combine value and stddev into single column like '-9.14 (± 4.72)'
    """
    # Load episode results data
    if not isinstance(episode_results, list):
        episode_results = load_file(episode_results)

    # Check if episode_results is None or empty
    if episode_results is None or len(episode_results) == 0:
        print(f"No episode results found or file is empty.")
        return

    # Get unique env names from the filtered episode results
    env_names = sorted(set(ep.get("env_name") for ep in episode_results if ep.get("env_name")))

    for env_name in env_names:
        # Filter for specific env
        task_episodes = [ep for ep in episode_results if ep.get("env_name") == env_name]

        print_result_table(
            task_episodes,
            group_by="task",
            title=f"ENV SUMMARY: {env_name}",
            show_scores=True,
            show_eps=True,
            show_total=False,  # Don't show total row for single task
            show_duration=True,
            csv=csv,
            csv_compact=csv_compact,
        )

        # Skip detailed episode output in CSV mode
        if csv:
            continue

        # Print individual episode details
        for ep in sorted(task_episodes, key=lambda x: x.get("episode", 0)):
            episode_id = ep.get("episode", "N/A")
            success_str = f"{GREEN}✓ Success{RESET}" if ep.get("success") else f"{RED}✗ Failure{RESET}"
            score_str = format_score(ep.get("score"))
            info = ep.get("reason", ep.get("info", ""))

            if info:
                info_str = f", {BOLD}Info: {info}{RESET}"
            else:
                info_str = ""
            print(f"{BOLD}Episode {episode_id}: {RESET} {success_str},{BOLD} Score: {score_str}{RESET}{info_str}")

            if VERBOSE:
                # Print subtask breakdown if task_output_dir is provided
                print_episode_subtask_status(episode_id, ep.get('data_dir'), ep.get('dt'), run_idx=ep.get('run'), env_id=ep.get('env_id'))
                # summarize_error_reasons([ep])
            summarize_timestep_errors([ep], indent="    ")

        # Print error summary (episode-level reasons)
        # summarize_error_reasons(task_episodes)

        # Print timestep-level error aggregation when verbose is enabled
        print(f"{BOLD}All Task Errors:{RESET}")
        summarize_timestep_errors(task_episodes)

        # Add separator between tasks if not the last one
        if env_name != env_names[-1]:
            print()

def _print_subtask_from_json(task_output_dir: str, episode_id: int, step_dt: float = None, indent: str = "    ", run_idx: int = None, env_id: int = None):
    """Print subtask status from JSON log file."""
    # Try per-env log file first (multi-env), then legacy format
    log_data = None
    if run_idx is not None and env_id is not None:
        per_env_path = os.path.join(task_output_dir, f"log_{run_idx}_env{env_id}.json")
        log_data = load_file(per_env_path) if os.path.exists(per_env_path) else None
    if log_data is None:
        rid = run_idx if run_idx is not None else episode_id
        log_data = load_file(os.path.join(task_output_dir, f"log_{rid}.json"))

    if log_data is None:
        print(f"{indent}No subtask log found for episode {episode_id}")
        return

    status_changes = extract_subtask_status_changes(log_data)

    if len(status_changes) == 0:
        # print(f"{indent}No subtask status changes in episode {episode_id}")
        return

    status_width = max(len(get_status_name(change["status"])) for change in status_changes)
    # Print header with optional timing columns
    if step_dt is not None:
        print(f"{indent}{'Step':<8} {'Time (s)':<10} {'Duration':<12} {'Status':<{status_width}} {'Score':<10} {'Info'}")
        # print(f"{indent}{'-' * 100}")
    else:
        print(f"{indent}{'Step':<8} {'Status':<{status_width}} {'Score':<10} {'Info'}")
        # print(f"{indent}{'-' * 80}")

    prev_step = 0
    for change in status_changes:
        step = change["step"]
        status_name = get_status_name(change["status"])
        score = change["score"]
        info = change["info"]  # Truncate long info strings

        if step_dt is not None:
            time_s = step * step_dt
            duration = (step - prev_step) * step_dt
            duration_str = f"+{duration:.2f}s"
            print(f"{indent}{step:<8} {time_s:<10.2f} {duration_str:<12} {status_name:<{status_width}} {score:<10.3f} {info}")
            prev_step = step
        else:
            print(f"{indent}{step:<8} {status_name:<{status_width}} {score:<10.3f} {info}")

def _print_subtask_from_hdf5(hdf5_path: str, episode: int, step_dt: float = None, indent: str = ""):
    """Print subtask status from HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        subtask_completion = f[f'data/demo_{episode}/subtask']  # type: ignore

        # Get the arrays
        completed = subtask_completion['completed'][:]  # type: ignore
        score = subtask_completion['score'][:]  # type: ignore
        status = subtask_completion['status'][:]  # type: ignore

        # Print dataset info
        step_dt_str = ""
        total_time_str = ""
        if step_dt is not None:
            total_time = len(status) * step_dt
            total_time_str = f", Total time: {total_time:.2f}s"
            step_dt_str = f", Step dt: {step_dt:.4f}s ({1/step_dt:.2f} Hz)"

        print(f"{indent}Episode: demo_{episode}, Total steps: {len(status)}{total_time_str}{step_dt_str}")

        # Print only steps where status is nonzero
        if step_dt is not None:
            print(f"{indent}{'Step':<6} {'Time (s)':<10} {'Duration':<12} {'Status':<45} {'Completed':<10} {'Score':<10}")
            # print(f"{indent}{'-' * 105}")
        else:
            print(f"{indent}{'Step':<6} {'Status':<45} {'Completed':<10} {'Score':<10}")
            # print(f"{indent}{'-' * 75}")

        nonzero_count = 0
        prev_nonzero_step = 0  # Start from step 0

        for step in range(len(status)):
            if status[step] != 0:  # type: ignore
                status_name = get_status_name(status[step])  # type: ignore

                if step_dt is not None:
                    time_s = step * step_dt
                    duration = (step - prev_nonzero_step) * step_dt
                    duration_str = f"+{duration:.2f}s"
                    print(f"{indent}{step:<6} {time_s:<10.2f} {duration_str:<12} {status_name:<45} {completed[step]:<10} {score[step]:<10.4f}")  # type: ignore
                else:
                    print(f"{indent}{step:<6} {status_name:<45} {completed[step]:<10} {score[step]:<10.4f}")  # type: ignore

                prev_nonzero_step = step
                nonzero_count += 1

        if nonzero_count == 0:
            print(f"{indent}(No nonzero status values found)")

        if not indent:  # Only add blank line for top-level calls
            print()

# ============================================================================
# Printing Functions for HDF5 files
# ============================================================================

def print_episode_subtask_status(episode_id: int, source: str, step_dt: float = None, indent: str = "    ", run_idx: int = None, env_id: int = None):
    """
    Print subtask status breakdown for a single episode from either JSON or HDF5 source.

    Args:
        episode_id: Episode number
        source: Either a directory containing JSON log files or an HDF5 file path
        step_dt: Optional timestep for timing information (only for HDF5)
        indent: Indentation prefix for output lines
        run_idx: Run index (for per-env log file resolution)
        env_id: Environment ID (for per-env log file resolution)
    """
    # Determine source type and load data
    if source.endswith('.hdf5') or source.endswith('.h5'):
        # HDF5 source
        _print_subtask_from_hdf5(source, episode_id, step_dt, indent)
    else:
        # JSON source (directory)
        _print_subtask_from_json(source, episode_id, step_dt, indent, run_idx=run_idx, env_id=env_id)

def print_all_episodes(hdf5_path: str, step_dt: float = None):
    """Print subtask status for all episodes in an HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f['data']  # type: ignore

        # Find all demo episodes
        episodes = sorted([key for key in data_group.keys() if key.startswith('demo_')])  # type: ignore

        if not episodes:
            print("No episodes found in the HDF5 file.")
            return

        print(f"Found {len(episodes)} episode(s) in {hdf5_path}\n")
        print("=" * 105 if step_dt is not None else "=" * 75)

        for episode_name in episodes:
            episode_num = int(episode_name.split('_')[1])
            print_episode_subtask_status(episode_num, hdf5_path, step_dt, indent="")
            print("=" * 105 if step_dt is not None else "=" * 75)

# ============================================================================
# Formatting Functions
# ============================================================================

def format_score(score: float | None, precision: int = 3) -> str:
    """Format a score as a string with N/A for None values."""
    return f"{score:.{precision}f}" if score is not None else "-"

def format_compact_value(value_str: str, stddev_str: str) -> str:
    """Format a value with stddev in compact format like '-9.14 (± 4.72)'."""
    if value_str == "-":
        return "-"
    if stddev_str == "-":
        return value_str
    return f"{value_str} (± {stddev_str})"

def format_centered_header(header_text: str, total_width: int) -> str:
    """Format a centered header with dashes and bold text."""
    dash_count = (total_width - len(header_text) - 2) // 2  # -2 for spaces around text
    remaining_dashes = total_width - len(header_text) - 2 - (dash_count * 2)
    return f"{'-' * dash_count}{BOLD} {header_text} {RESET}{'-' * (dash_count + remaining_dashes)}"

# ============================================================================
# Loading Data Functions
# ============================================================================

def load_timestep_from_config(config_path: str) -> float | None:
    """Load timestep information from env_cfg.json."""
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        print(f"Warning: env_cfg.json not found at {config_path}. Using default timestep.")
        return None

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        dt = config.get('sim', {}).get('dt', None)
        decimation = config.get('decimation', None)

        if dt is not None and decimation is not None:
            return dt * decimation
        else:
            print("Warning: Could not find dt or decimation in config. Using default.")
            return None
    except Exception as e:
        print(f"Warning: Error reading config file: {e}. Using default timestep.")
        return None


# Known metric fields that can be loaded from episode_metrics.json
# Trajectory metrics (smoothness, path length, speed)
KNOWN_METRIC_FIELDS = [
    'ee_sparc',           # End Effector SPARC (smoothness)
    'joint_sparc_mean',   # Joint SPARC mean
    'ee_path_length',     # End Effector path length
    'ee_speed_mean',      # End Effector mean speed
    'ee_speed_max',       # End Effector max speed
    'ee_isj',             # End Effector ISJ
    'joint_isj',          # Joint ISJ
    'joint_rmse_mean',    # Joint RMSE mean
]


def load_and_merge_episode_data(folder_path: str) -> list[dict]:
    """
    Load episode results and merge with episode metrics if available.

    This function loads episode results (.jsonl or legacy .json) and optionally
    merges in additional fields from episode_metrics.json if it exists.

    Args:
        folder_path: Path to the folder containing episode_results.jsonl
                     (or legacy episode_results.json) and optionally episode_metrics.json

    Returns:
        List of episode dictionaries with merged data
    """
    episode_metrics_file = os.path.join(folder_path, "episode_metrics.json")

    # Load episode results (supports both .jsonl and legacy .json)
    episode_results = load_episode_results(folder_path)
    if not episode_results:
        return []

    # Build a lookup for metrics if the file exists
    metrics_lookup = {}
    if os.path.exists(episode_metrics_file):
        metrics_data = load_file(episode_metrics_file)
        if metrics_data is not None:
            for metric in metrics_data:
                env_name = metric.get("env_name")
                episode = metric.get("episode")
                if env_name is not None and episode is not None:
                    key = (env_name, episode)
                    metrics_lookup[key] = metric

    # Merge metrics into episode results
    merged_results = []
    for episode in episode_results:
        # Add data directory and timestep
        episode['data_dir'] = os.path.join(folder_path, episode.get('env_name', ''))
        episode['dt'] = load_timestep_from_config(os.path.join(episode['data_dir'], 'env_cfg.json'))

        # First, flatten nested "metrics" dict to top-level (new format)
        nested_metrics = episode.get("metrics")
        if nested_metrics and isinstance(nested_metrics, dict):
            for field in KNOWN_METRIC_FIELDS:
                if field in nested_metrics and field not in episode:
                    episode[field] = nested_metrics[field]

        # Then, merge from episode_metrics.json if available (legacy format)
        key = (episode.get("env_name"), episode.get("episode"))
        if key in metrics_lookup:
            metric = metrics_lookup[key]
            # Add known metric fields that don't already exist in episode
            for field in KNOWN_METRIC_FIELDS:
                if field in metric and field not in episode:
                    episode[field] = metric[field]
            # Also copy duration if present in metrics but not in episode
            if 'duration' in metric and 'duration' not in episode:
                episode['duration'] = metric['duration']

        merged_results.append(episode)

    return merged_results


def get_available_metrics(episode_results: list[dict]) -> list[str]:
    """
    Detect which metric fields are available in the episode results.

    Args:
        episode_results: List of episode result dictionaries

    Returns:
        List of metric field names that are present in at least one episode
    """
    available = []
    for field in KNOWN_METRIC_FIELDS:
        for ep in episode_results:
            if field in ep and ep[field] is not None:
                available.append(field)
                break
    return available


def summarize_experiments_by_wrong_objects(episode_results: list[dict] | str,
                                           exclude_containers: bool = False,
                                           csv: bool = False):
    """
    Summarize results for all tasks by wrong object behavior.

    For each task, shows:
    - Total episodes
    - Episodes with wrong object grabs that succeeded (count and percentage)
    - Episodes with wrong object grabs that failed (count and percentage)
    - Which objects were grabbed during success
    - Which objects were grabbed during failure

    Args:
        episode_results: List of episode result dictionaries or path to episode_results.json
        exclude_containers: If True, exclude container objects from wrong object counts
        csv: If True, output in CSV format for easy pasting into spreadsheets
    """
    # Load episode results data
    if not isinstance(episode_results, list):
        episode_results = load_file(episode_results)

    # Check if episode_results is None or empty
    if episode_results is None or len(episode_results) == 0:
        print("No episode results found or file is empty.")
        return

    # Group episodes by env_name
    task_episodes = {}
    for ep in episode_results:
        env_name = ep.get("env_name", "Unknown")
        if env_name not in task_episodes:
            task_episodes[env_name] = []
        task_episodes[env_name].append(ep)

    # Define column widths
    sep = "," if csv else " | "
    task_width = 40

    # Headers
    if csv:
        headers = ["Task", "Total", "WO-Succ", "WO-Succ%", "WO-Fail", "WO-Fail%",
                   "Objects (Success)", "Objects (Failure)"]
        print(sep.join(headers))
    else:
        header = (f"{'Task':<{task_width}}{sep}{'Total':>6}{sep}{'WO-Succ':>8}{sep}{'WO-Succ%':>9}"
                  f"{sep}{'WO-Fail':>8}{sep}{'WO-Fail%':>9}{sep}{'Objects (Success)':<30}{sep}{'Objects (Failure)':<30}")
        total_width = len(header) + 20  # Account for potential ANSI codes
        print(format_centered_header("WRONG OBJECT SUMMARY", total_width))
        print(header)
        print("-" * total_width)

    # Collect data for each task
    all_data = []
    total_episodes = 0
    total_wo_success = 0
    total_wo_failure = 0
    all_objects_success = Counter()
    all_objects_failure = Counter()

    for task_name in sorted(task_episodes.keys()):
        episodes = task_episodes[task_name]
        num_episodes = len(episodes)
        total_episodes += num_episodes

        # Count episodes with wrong objects that succeeded vs failed
        wo_success_count = 0
        wo_failure_count = 0
        objects_success = Counter()
        objects_failure = Counter()

        for ep in episodes:
            events = ep.get("events", {})
            wrong_objects = events.get("wrong_objects_grabbed", [])

            # Filter out containers if requested
            if exclude_containers:
                wrong_objects = [obj for obj in wrong_objects if not is_container_object(obj)]

            if wrong_objects:
                is_success = ep.get("success", False)
                if is_success:
                    wo_success_count += 1
                    for obj in wrong_objects:
                        objects_success[obj] += 1
                        all_objects_success[obj] += 1
                else:
                    wo_failure_count += 1
                    for obj in wrong_objects:
                        objects_failure[obj] += 1
                        all_objects_failure[obj] += 1

        total_wo_success += wo_success_count
        total_wo_failure += wo_failure_count

        # Calculate percentages
        wo_success_pct = (wo_success_count / num_episodes * 100) if num_episodes > 0 else 0
        wo_failure_pct = (wo_failure_count / num_episodes * 100) if num_episodes > 0 else 0

        # Format object lists
        def format_objects(counter: Counter) -> str:
            if not counter:
                return "-"
            items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            return ", ".join(f"{obj}({cnt})" for obj, cnt in items)

        objects_success_str = format_objects(objects_success)
        objects_failure_str = format_objects(objects_failure)

        all_data.append({
            "task": task_name,
            "total": num_episodes,
            "wo_success": wo_success_count,
            "wo_success_pct": wo_success_pct,
            "wo_failure": wo_failure_count,
            "wo_failure_pct": wo_failure_pct,
            "objects_success": objects_success_str,
            "objects_failure": objects_failure_str,
        })

    # Print total row first
    total_wo_success_pct = (total_wo_success / total_episodes * 100) if total_episodes > 0 else 0
    total_wo_failure_pct = (total_wo_failure / total_episodes * 100) if total_episodes > 0 else 0

    def format_objects_total(counter: Counter) -> str:
        if not counter:
            return "-"
        items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:5]  # Top 5
        result = ", ".join(f"{obj}({cnt})" for obj, cnt in items)
        if len(counter) > 5:
            result += f" (+{len(counter) - 5} more)"
        return result

    if csv:
        total_row = [
            "TOTAL",
            str(total_episodes),
            str(total_wo_success),
            f"{total_wo_success_pct:.1f}%",
            str(total_wo_failure),
            f"{total_wo_failure_pct:.1f}%",
            format_objects_total(all_objects_success),
            format_objects_total(all_objects_failure),
        ]
        print(sep.join(total_row))
    else:
        total_row = (f"{BOLD}{'TOTAL':<{task_width}}{RESET}{sep}{total_episodes:>6}{sep}"
                     f"{total_wo_success:>8}{sep}{total_wo_success_pct:>8.1f}%{sep}"
                     f"{total_wo_failure:>8}{sep}{total_wo_failure_pct:>8.1f}%{sep}"
                     f"{format_objects_total(all_objects_success):<30}{sep}"
                     f"{format_objects_total(all_objects_failure):<30}")
        print(total_row)
        print("-" * len(header))

    # Print per-task rows
    for data in all_data:
        if csv:
            row = [
                data["task"],
                str(data["total"]),
                str(data["wo_success"]),
                f"{data['wo_success_pct']:.1f}%",
                str(data["wo_failure"]),
                f"{data['wo_failure_pct']:.1f}%",
                data["objects_success"],
                data["objects_failure"],
            ]
            print(sep.join(row))
        else:
            # Colorize counts
            wo_succ_str = f"{GREEN}{data['wo_success']:>8}{RESET}" if data['wo_success'] > 0 else f"{data['wo_success']:>8}"
            wo_fail_str = f"{RED}{data['wo_failure']:>8}{RESET}" if data['wo_failure'] > 0 else f"{data['wo_failure']:>8}"

            row = (f"{data['task']:<{task_width}}{sep}{data['total']:>6}{sep}"
                   f"{wo_succ_str}{sep}{data['wo_success_pct']:>8.1f}%{sep}"
                   f"{wo_fail_str}{sep}{data['wo_failure_pct']:>8.1f}%{sep}"
                   f"{data['objects_success']:<30}{sep}{data['objects_failure']:<30}")
            print(row)

    if not csv:
        print("-" * len(header))
        print(f"\nLegend: WO-Succ = Episodes with wrong object grabs that still succeeded")
        print(f"        WO-Fail = Episodes with wrong object grabs that failed")
        print(f"        Objects show: object_name(grab_count)")
