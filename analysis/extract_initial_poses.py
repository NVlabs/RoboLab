# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Extract initial camera and object poses from HDF5 files.

Reads episode_metrics.json (or episode_results.json), extracts initial poses from HDF5 files,
saves augmented data to episode_initial_poses.json, and prints results as a table.

Extracted data:
- <camera>_initial_pose: [x, y, z, qw, qx, qy, qz] (7-element array)
- <object>_initial_pose: [x, y, z, qw, qx, qy, qz] (7-element array)
"""

import argparse
import json
import os
from typing import Any

import h5py
import numpy as np

from robolab.constants import DEFAULT_OUTPUT_DIR

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'


def load_json(filepath: str) -> Any:
    """Load JSON file, returns None if file doesn't exist or is invalid."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def get_available_demos(hdf5_path: str) -> list:
    """Get list of available demo keys in the HDF5 file."""
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                return []
            return list(f["data"].keys())
    except Exception:
        return []


def extract_initial_poses(hdf5_path: str, demo_key: str) -> dict:
    """
    Extract initial camera extrinsics and object poses from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file
        demo_key: Key for the demo to load (e.g., "demo_0")

    Returns:
        Dictionary containing:
        - <camera_name>_initial_pose: [x, y, z, qw, qx, qy, qz]
        - <object_name>_initial_pose: [x, y, z, qw, qx, qy, qz]
    """
    poses = {}

    try:
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f or demo_key not in f["data"]:
                return poses

            demo = f["data"][demo_key]

            # Extract camera extrinsics
            # Try new location first: initial_state/cameras
            # Fall back to old location: initial_camera_extrinsics (for backwards compatibility)
            cam_group = None
            if "initial_state" in demo and "cameras" in demo["initial_state"]:
                cam_group = demo["initial_state"]["cameras"]
            elif "initial_camera_extrinsics" in demo:
                cam_group = demo["initial_camera_extrinsics"]

            if cam_group is not None:
                for camera_name in cam_group.keys():
                    camera = cam_group[camera_name]
                    if "position" in camera and "orientation" in camera:
                        # Position: shape (N, 3), take first row -> [x, y, z]
                        position = camera["position"][0, :]  # (3,)
                        # Orientation: shape (N, 4), take first row -> [qw, qx, qy, qz]
                        orientation = camera["orientation"][0, :]  # (4,)
                        # Combine: [x, y, z, qw, qx, qy, qz]
                        pose = np.concatenate([position, orientation]).tolist()
                        poses[f"{camera_name}_initial_pose"] = pose

            # Extract rigid object initial poses
            if "initial_state" in demo and "rigid_object" in demo["initial_state"]:
                obj_group = demo["initial_state"]["rigid_object"]
                for object_name in obj_group.keys():
                    obj = obj_group[object_name]
                    if "root_pose" in obj:
                        # root_pose: shape (3, 7), take first row -> [x, y, z, qw, qx, qy, qz]
                        root_pose = obj["root_pose"][0, :]  # (7,)
                        poses[f"{object_name}_initial_pose"] = root_pose.tolist()

    except Exception as e:
        print(f"Error extracting poses from {hdf5_path}/{demo_key}: {e}")

    return poses


def process_experiment_folder(
    folder_path: str,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Process a single experiment folder and extract initial poses.

    Reads episode_metrics.json (or episode_results.json), extracts poses from HDF5 data,
    and saves results to episode_initial_poses.json.

    Args:
        folder_path: Path to the experiment folder
        overwrite: If True, recompute poses even if they exist
        verbose: If True, print progress information

    Returns:
        List of episode dictionaries with poses added
    """
    episode_metrics_file = os.path.join(folder_path, "episode_metrics.json")
    output_file = os.path.join(folder_path, "episode_initial_poses.json")

    # Try to load episode_metrics.json first, fall back to episode results (.jsonl or .json)
    episode_data = load_json(episode_metrics_file)
    if episode_data is None:
        from robolab.core.logging.results import load_episode_results
        episode_data = load_episode_results(folder_path) or None
    if episode_data is None:
        if verbose:
            print(f"Warning: Could not load episode data from {folder_path}")
        return []

    # Load existing poses if not overwriting
    existing_poses = {}
    if not overwrite and os.path.exists(output_file):
        existing_data = load_json(output_file)
        if existing_data:
            for ep in existing_data:
                key = (ep.get("env_name"), ep.get("episode"))
                existing_poses[key] = ep

    # Process each episode
    processed_episodes = []
    skipped_count = 0
    extracted_count = 0
    failed_count = 0

    for ep in episode_data:
        # Try multiple folder name candidates
        run_name = ep.get("env_name")
        env_name = ep.get("env_name")
        episode_num = ep.get("episode")

        # For display/grouping, prefer env_name
        display_name = env_name or run_name
        key = (display_name, episode_num)

        # Check if already processed
        if key in existing_poses and not overwrite:
            existing = existing_poses[key]
            # Check if poses were already extracted
            has_poses = any(k.endswith("_initial_pose") for k in existing.keys())
            if has_poses:
                processed_episodes.append(existing)
                skipped_count += 1
                continue

        # Copy base episode data
        ep_with_poses = ep.copy()

        # Extract and add experiment_name from folder path
        experiment_name = extract_experiment_name(folder_path, ep.get("policy"))
        ep_with_poses["experiment_name"] = experiment_name

        # Try to find HDF5 file - supports both run_*.hdf5 (multi-env) and data.hdf5 (legacy)
        folder_candidates = []
        if env_name:
            folder_candidates.append(env_name)
        if run_name and run_name != env_name:
            folder_candidates.append(run_name)

        run_idx = ep.get("run")
        env_id = ep.get("env_id")
        hdf5_path = None
        demo_key = None

        for candidate in folder_candidates:
            candidate_dir = os.path.join(folder_path, candidate)
            if not os.path.isdir(candidate_dir):
                continue
            # Multi-env: run_{run_idx}.hdf5 with demo_{env_id}
            if run_idx is not None:
                run_path = os.path.join(candidate_dir, f"run_{run_idx}.hdf5")
                if os.path.exists(run_path):
                    hdf5_path = run_path
                    demo_key = f"demo_{env_id}" if env_id is not None else f"demo_{episode_num}"
                    break
            # Legacy: data.hdf5 with demo_{episode_num}
            legacy_path = os.path.join(candidate_dir, "data.hdf5")
            if os.path.exists(legacy_path):
                hdf5_path = legacy_path
                demo_key = f"demo_{episode_num}"
                break

        if hdf5_path is None:
            if verbose:
                print(f"Warning: HDF5 file not found for {env_name} episode {episode_num}")
            processed_episodes.append(ep_with_poses)
            failed_count += 1
            continue

        # Check if demo exists
        available_demos = get_available_demos(hdf5_path)
        if demo_key not in available_demos:
            if verbose:
                print(f"Warning: {demo_key} not found in {hdf5_path}")
            processed_episodes.append(ep_with_poses)
            failed_count += 1
            continue

        # Extract poses
        poses = extract_initial_poses(hdf5_path, demo_key)
        if poses:
            ep_with_poses.update(poses)
            extracted_count += 1
        else:
            failed_count += 1

        processed_episodes.append(ep_with_poses)

    # Save to output file
    if processed_episodes:
        with open(output_file, "w") as f:
            json.dump(processed_episodes, f, indent=2)
        if verbose:
            print(f"Saved {len(processed_episodes)} episodes to: {output_file}")
            print(f"  - Extracted poses: {extracted_count}")
            print(f"  - Skipped (existing): {skipped_count}")
            print(f"  - Failed: {failed_count}")
    elif verbose:
        print("No episodes to save.")

    return processed_episodes


def format_pose(pose: list | None, precision: int = 4) -> str:
    """Format a pose array for display."""
    if pose is None:
        return "-"
    return "[" + ", ".join(f"{v:.{precision}f}" for v in pose) + "]"


def format_value(value, precision: int = 4) -> str:
    """Format a value for display."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, list):
        return format_pose(value, precision)
    return str(value)


def get_all_pose_keys(episodes: list[dict]) -> list[str]:
    """Get all unique pose keys from episodes."""
    pose_keys = set()
    for ep in episodes:
        for key in ep.keys():
            if key.endswith("_initial_pose"):
                pose_keys.add(key)
    return sorted(pose_keys)


def extract_experiment_name(folder_path: str, policy: str | None = None) -> str:
    """Extract experiment name from folder path.

    Folder names are expected to be in format: <policy>_<experiment_name>
    e.g., 'pi0_table_variation' -> 'table_variation'

    Args:
        folder_path: Path to the experiment folder
        policy: Optional policy name to use for splitting

    Returns:
        The experiment name portion of the folder name
    """
    folder_name = os.path.basename(folder_path.rstrip('/'))

    # If policy is provided, use it to split
    if policy and folder_name.startswith(policy + "_"):
        return folder_name[len(policy) + 1:]

    # Try known policy prefixes
    known_policies = ["pi05_fast", "pi0_fast", "pi05", "pi0", "paligemma"]
    for prefix in known_policies:
        if folder_name.startswith(prefix + "_"):
            return folder_name[len(prefix) + 1:]

    # Fallback: split by first underscore
    parts = folder_name.split("_", 1)
    return parts[1] if len(parts) > 1 else folder_name


def get_all_field_keys(episodes: list[dict]) -> list[str]:
    """Get all unique non-pose field keys from episodes, in a sensible order."""
    # Define preferred order for common fields
    preferred_order = [
        "env_name", "task_name", "policy", "experiment_name", "run", "episode", "success", "score", "reason",
        "instruction", "attributes",
        "background", "table_material", "lighting_intensity", "lighting_color", "lighting_type",
        "episode_step", "duration", "dt",
        "ee_sparc", "joint_sparc_mean", "ee_isj", "joint_isj",
        "ee_path_length", "joint_rmse_mean", "ee_speed_max", "ee_speed_mean",
    ]

    # Collect all keys from all episodes
    all_keys = set()
    for ep in episodes:
        all_keys.update(ep.keys())

    # Remove pose keys
    all_keys = {k for k in all_keys if not k.endswith("_initial_pose")}

    # Sort: preferred order first, then alphabetically for the rest
    ordered = []
    for key in preferred_order:
        if key in all_keys:
            ordered.append(key)
            all_keys.remove(key)
    ordered.extend(sorted(all_keys))

    return ordered


def format_field_value(value, quote_text: bool = False) -> str:
    """Format a field value for CSV output.

    Args:
        value: The value to format
        quote_text: If True, wrap string values in quotes (for text fields like reason)
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        # For lists like attributes, join with semicolon
        return ";".join(str(v) for v in value)
    str_value = str(value)
    if quote_text:
        # Escape any existing quotes and wrap in quotes
        escaped = str_value.replace('"', '""')
        return f'"{escaped}"'
    return str_value


def print_episodes_table(
    episode_results: list[dict],
    csv: bool = False,
    show_all: bool = False,
    compact: bool = False,
    output_file: str | None = None,
):
    """
    Print a table showing each individual episode with its initial poses.

    Args:
        episode_results: List of episode dictionaries with poses
        csv: If True, output in CSV format
        show_all: If True, show all pose columns (cameras + objects)
        compact: If True, show only position (xyz) instead of full pose
        output_file: If provided, write to this file instead of stdout
    """
    if not episode_results:
        print("No episodes to display.")
        return

    # Get all pose keys
    all_pose_keys = get_all_pose_keys(episode_results)
    pose_keys = all_pose_keys

    # Get all other field keys
    field_keys = get_all_field_keys(episode_results)

    # Sort episodes by task name, then episode number
    sorted_episodes = sorted(
        episode_results,
        key=lambda x: (x.get("env_name") or "", x.get("episode", 0))
    )

    sep = ","

    # Build header - all fields first, then poses
    header_parts = list(field_keys)
    for key in pose_keys:
        header_parts.append(key)

    header = sep.join(header_parts)

    # Collect all lines
    lines = []

    if not csv and not output_file:
        lines.append(f"\n{BOLD}{'=' * 20} INITIAL POSES {'=' * 20}{RESET}")
    lines.append(header)
    if not csv and not output_file:
        lines.append("-" * min(len(header), 200))

    # Build each episode row
    for ep in sorted_episodes:
        row_parts = []

        # Add all regular fields
        for key in field_keys:
            value = ep.get(key)
            # Quote text fields that may contain commas or special characters
            quote_text = key in ("reason", "instruction")
            row_parts.append(format_field_value(value, quote_text=quote_text))

        # Add poses as arrays
        for key in pose_keys:
            pose = ep.get(key)
            if pose:
                if compact:
                    # Just xyz position as array
                    arr_str = "[" + ";".join(f"{v:.4f}" for v in pose[:3]) + "]"
                else:
                    # Full pose as array: [x, y, z, qw, qx, qy, qz]
                    arr_str = "[" + ";".join(f"{v:.4f}" for v in pose) + "]"
                row_parts.append(arr_str)
            else:
                row_parts.append("")

        lines.append(sep.join(row_parts))

    if not csv and not output_file:
        lines.append("-" * min(len(header), 200))
        if compact:
            lines.append(f"\nPose format: [x;y;z] (position only)")
        else:
            lines.append(f"\nPose format: [x;y;z;qw;qx;qy;qz]")

    # Output to file or stdout
    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Saved table to: {output_file}")
    else:
        for line in lines:
            print(line)


def print_summary_table(
    episode_results: list[dict],
    csv: bool = False,
):
    """
    Print a summary table grouped by task with pose statistics.

    Args:
        episode_results: List of episode dictionaries with poses
        csv: If True, output in CSV format
    """
    if not episode_results:
        print("No episodes to display.")
        return

    # Get all pose keys
    all_pose_keys = get_all_pose_keys(episode_results)

    # Group by env_name
    env_data = {}
    for ep in episode_results:
        env_name = ep.get("env_name") or "unknown"
        if env_name not in env_data:
            env_data[env_name] = []
        env_data[env_name].append(ep)

    sep = "," if csv else " "

    # Build header
    header_parts = ["Task", "Episodes", "Success Rate", "Poses Extracted"]
    header = sep.join(header_parts)

    # Print header
    if not csv:
        print(f"\n{BOLD}{'=' * 20} INITIAL POSES SUMMARY {'=' * 20}{RESET}")
    print(header)
    if not csv:
        print("-" * len(header))

    # Print total row first
    total_episodes = len(episode_results)
    total_success = sum(1 for ep in episode_results if ep.get("success"))
    total_with_poses = sum(
        1 for ep in episode_results
        if any(k.endswith("_initial_pose") for k in ep.keys())
    )

    if csv:
        total_parts = [
            f"TOTAL ({len(env_data)} envs)",
            str(total_episodes),
            f"{total_success/total_episodes*100:.1f}%",
            str(total_with_poses),
        ]
    else:
        total_parts = [
            f"{BOLD}TOTAL ({len(env_data)} envs){RESET}",
            str(total_episodes),
            f"{GREEN}{total_success/total_episodes*100:.1f}%{RESET}",
            str(total_with_poses),
        ]
    print(sep.join(total_parts))

    if not csv:
        print("-" * len(header))

    # Print per-env rows
    for env_name in sorted(env_data.keys()):
        episodes = env_data[env_name]
        n_episodes = len(episodes)
        n_success = sum(1 for ep in episodes if ep.get("success"))
        n_with_poses = sum(
            1 for ep in episodes
            if any(k.endswith("_initial_pose") for k in ep.keys())
        )

        if csv:
            row_parts = [
                env_name,
                str(n_episodes),
                f"{n_success/n_episodes*100:.1f}%",
                str(n_with_poses),
            ]
        else:
            rate = n_success / n_episodes if n_episodes > 0 else 0
            row_parts = [
                task,
                str(n_episodes),
                f"{GREEN if rate > 0.5 else RED}{rate*100:.1f}%{RESET}",
                str(n_with_poses),
            ]

        print(sep.join(row_parts))

    if not csv:
        print("-" * len(header))
        print(f"\nAvailable pose keys: {', '.join(all_pose_keys)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract initial camera and object poses from HDF5 data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_initial_poses.py output/var_results/pi0_table_variation
  python extract_initial_poses.py output/var_results/* --csv              # CSV to stdout
  python extract_initial_poses.py output/var_results/* --csv --output-file poses.csv  # Save to file
  python extract_initial_poses.py output/var_results/* --csv --compact    # CSV with just xyz positions
  python extract_initial_poses.py output/var_results/* --summary          # Summary view (counts only)
  python extract_initial_poses.py output/var_results/* --overwrite        # Force recompute
        """,
    )

    parser.add_argument(
        "folder",
        nargs="+",
        help="Folder name(s) or absolute path(s) containing results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute poses even if episode_initial_poses.json exists",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output in CSV format for copy-pasting",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary table instead of individual episodes",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all pose columns (all cameras and objects)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Show poses in compact format (just position xyz)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Save CSV output to this file instead of printing to stdout",
    )

    args = parser.parse_args()

    # Process all folders
    all_episodes = []

    for folder in args.folder:
        # Resolve folder path
        if os.path.isabs(folder):
            folder_path = folder
        elif os.path.exists(folder):
            folder_path = os.path.abspath(folder)
        else:
            folder_path = os.path.join(DEFAULT_OUTPUT_DIR, folder)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue

        print(f"\nProcessing: {folder_path}")

        episodes = process_experiment_folder(
            folder_path,
            overwrite=args.overwrite,
        )

        all_episodes.extend(episodes)

    # Print table
    if all_episodes:
        if args.summary:
            print_summary_table(
                all_episodes,
                csv=args.csv,
            )
        else:
            print_episodes_table(
                all_episodes,
                csv=args.csv,
                show_all=args.all,
                compact=args.compact,
                output_file=args.output_file,
            )
    else:
        print("\nNo episodes found to process.")


if __name__ == "__main__":
    main()
