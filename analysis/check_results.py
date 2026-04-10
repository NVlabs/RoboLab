# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Check integrity of episode_results.json against data.hdf5 files.

Verifies that each episode in episode_results.json has a corresponding
demo entry in the HDF5 file. Reports any missing/corrupt episodes.
"""

import argparse
import glob
import os

import h5py

from robolab.constants import DEFAULT_OUTPUT_DIR
from robolab.core.logging.results import load_episode_results
from robolab.core.utils.file_utils import load_file

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'


def check_demo_exists(hdf5_path: str, demo_key: str) -> bool:
    """Check if a demo key exists in the HDF5 file."""
    if not os.path.exists(hdf5_path):
        return False

    try:
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                return False
            return demo_key in f["data"]
    except Exception:
        return False


def get_available_demos(hdf5_path: str) -> list[str]:
    """Get list of available demo keys in the HDF5 file."""
    if not os.path.exists(hdf5_path):
        return []

    try:
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                return []
            return sorted(f["data"].keys())
    except Exception:
        return []


def get_demo_info(hdf5_path: str, demo_key: str) -> dict | None:
    """Get info about a specific demo in the HDF5 file."""
    if not os.path.exists(hdf5_path):
        return None

    try:
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f or demo_key not in f["data"]:
                return None
            demo = f["data"][demo_key]
            info = {
                "num_samples": demo.attrs.get("num_samples", "N/A"),
                "success": demo.attrs.get("success", "N/A"),
                "keys": list(demo.keys()),
            }
            # Get action shape if available
            if "actions" in demo:
                info["actions_shape"] = demo["actions"].shape
            return info
    except Exception as e:
        return {"error": str(e)}


def diagnose_hdf5(folder_path: str):
    """Print detailed diagnostic info about HDF5 files in the folder."""
    episode_results = load_episode_results(folder_path)

    if not episode_results:
        print(f"{RED}No episode results found{RESET}")
        return

    # Group by env_name
    envs = {}
    for ep in episode_results:
        env_name = ep.get("env_name", "unknown")
        if env_name not in envs:
            envs[env_name] = []
        envs[env_name].append(ep)

    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}HDF5 Diagnostic: {folder_path}{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")

    for env_name in sorted(envs.keys()):
        env_dir = os.path.join(folder_path, env_name)
        episodes_in_json = sorted([ep.get("episode") for ep in envs[env_name]])

        # Find HDF5 files: run_*.hdf5 (multi-env) or data.hdf5 (legacy)
        hdf5_files = sorted(glob.glob(os.path.join(env_dir, "run_*.hdf5")))
        if not hdf5_files:
            legacy = os.path.join(env_dir, "data.hdf5")
            if os.path.exists(legacy):
                hdf5_files = [legacy]

        print(f"\n{BOLD}Environment: {env_name}{RESET}")
        print(f"  HDF5 files: {[os.path.basename(f) for f in hdf5_files]}")
        print(f"  Episodes in JSON: {episodes_in_json}")

        if not hdf5_files:
            print(f"  {RED}No HDF5 files found!{RESET}")
            continue

        # Collect all demos across all HDF5 files
        available_demos = []
        for hf in hdf5_files:
            available_demos.extend(get_available_demos(hf))
        demo_numbers = sorted([int(d.split("_")[1]) for d in available_demos if d.startswith("demo_")])

        print(f"  Demos in HDF5: {available_demos}")
        print(f"  Demo numbers: {demo_numbers}")

        # Check for mismatches
        missing_in_hdf5 = [e for e in episodes_in_json if e not in demo_numbers]
        extra_in_hdf5 = [d for d in demo_numbers if d not in episodes_in_json]

        if missing_in_hdf5:
            print(f"  {RED}Missing in HDF5: {missing_in_hdf5}{RESET}")
        if extra_in_hdf5:
            print(f"  {YELLOW}Extra in HDF5 (not in JSON): {extra_in_hdf5}{RESET}")

        # Show info for each demo
        for demo_name in available_demos[:5]:  # Limit to first 5
            info = get_demo_info(hdf5_path, demo_name)
            if info:
                print(f"    {demo_name}: samples={info.get('num_samples', 'N/A')}, "
                      f"success={info.get('success', 'N/A')}, "
                      f"actions_shape={info.get('actions_shape', 'N/A')}")

        if len(available_demos) > 5:
            print(f"    ... and {len(available_demos) - 5} more demos")

    print(f"\n{BOLD}{'=' * 80}{RESET}")


def check_folder(folder_path: str, verbose: bool = False) -> dict:
    """
    Check all episodes in a folder for data integrity.

    Args:
        folder_path: Path to the results folder
        verbose: If True, print details for each episode

    Returns:
        Dictionary with check results
    """
    episode_results = load_episode_results(folder_path)
    if not episode_results:
        print(f"{RED}Error: No episode results found in {folder_path}{RESET}")
        return {"error": "episode results not found"}

    # Group episodes by env_name
    envs = {}
    for ep in episode_results:
        env_name = ep.get("env_name", "unknown")
        if env_name not in envs:
            envs[env_name] = []
        envs[env_name].append(ep)

    # Check each episode
    results = {
        "total": 0,
        "valid": 0,
        "missing_hdf5": 0,
        "missing_demo": 0,
        "corrupt_episodes": [],
        "missing_hdf5_envs": [],
    }

    for env_name, episodes in sorted(envs.items()):
        env_dir = os.path.join(folder_path, env_name)
        hdf5_files = sorted(glob.glob(os.path.join(env_dir, "run_*.hdf5")))
        if not hdf5_files:
            legacy = os.path.join(env_dir, "data.hdf5")
            if os.path.exists(legacy):
                hdf5_files = [legacy]
        hdf5_exists = len(hdf5_files) > 0

        if not hdf5_exists:
            results["missing_hdf5_envs"].append(env_name)
            for ep in episodes:
                results["total"] += 1
                results["missing_hdf5"] += 1
                results["corrupt_episodes"].append({
                    "env_name": env_name,
                    "episode": ep.get("episode"),
                    "reason": "HDF5 file not found",
                    "data": ep,
                })
            continue

        # Get available demos in HDF5
        available_demos = get_available_demos(hdf5_path)

        for ep in episodes:
            results["total"] += 1
            episode_num = ep.get("episode")
            demo_key = f"demo_{episode_num}"

            if demo_key in available_demos:
                results["valid"] += 1
                if verbose:
                    print(f"  {GREEN}✓{RESET} {env_name}/demo_{episode_num}")
            else:
                results["missing_demo"] += 1
                results["corrupt_episodes"].append({
                    "env_name": env_name,
                    "episode": episode_num,
                    "reason": f"demo_{episode_num} not in HDF5 (available: {available_demos})",
                    "data": ep,
                })

    return results


def format_value(value, width: int = 10) -> str:
    """Format a value for table display."""
    if value is None:
        return "-".center(width)
    if isinstance(value, float):
        return f"{value:.3f}".center(width)
    return str(value).center(width)


def print_results(results: dict, folder_path: str):
    """Print the check results."""
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}Integrity Check Results: {folder_path}{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")

    total = results.get("total", 0)
    valid = results.get("valid", 0)
    missing_hdf5 = results.get("missing_hdf5", 0)
    missing_demo = results.get("missing_demo", 0)

    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  Total episodes:     {total}")
    print(f"  {GREEN}Valid:              {valid}{RESET}")
    if missing_hdf5 > 0:
        print(f"  {RED}Missing HDF5 file:  {missing_hdf5}{RESET}")
    if missing_demo > 0:
        print(f"  {RED}Missing demo entry: {missing_demo}{RESET}")

    # Print missing HDF5 tasks
    missing_hdf5_tasks = results.get("missing_hdf5_tasks", [])
    if missing_hdf5_tasks:
        print(f"\n{BOLD}{RED}Tasks with missing data.hdf5:{RESET}")
        for task in missing_hdf5_tasks:
            print(f"  - {task}")

    # Print corrupt episodes in table format
    corrupt = results.get("corrupt_episodes", [])
    if corrupt:
        print(f"\n{BOLD}{RED}Corrupt/Missing Episodes ({len(corrupt)}):{RESET}")

        # Calculate column widths
        name_width = max(
            len(f"{item['env_name']}_{item['episode']}") for item in corrupt
        )
        name_width = max(name_width, len("Episode"))

        score_width = 10
        step_width = 12
        duration_width = 10

        # Print header
        header = (
            f"{'Episode':<{name_width}} | "
            f"{'Score':^{score_width}} | "
            f"{'Steps':^{step_width}} | "
            f"{'Duration':^{duration_width}}"
        )
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        # Print rows
        for item in corrupt:
            env_name = item["env_name"]
            episode = item["episode"]
            data = item["data"]

            episode_name = f"{env_name}_{episode}"
            score = data.get("score")
            episode_step = data.get("episode_step")
            duration = data.get("duration")

            row = (
                f"{RED}{episode_name:<{name_width}}{RESET} | "
                f"{format_value(score, score_width)} | "
                f"{format_value(episode_step, step_width)} | "
                f"{format_value(duration, duration_width)}"
            )
            print(row)

        print("-" * len(header))
    else:
        print(f"\n{GREEN}All episodes have valid HDF5 data!{RESET}")

    print(f"\n{BOLD}{'=' * 60}{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Check integrity of episode_results.json against data.hdf5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_results.py pi0_fast_v2
  python check_results.py /abs/path/to/results --verbose
  python check_results.py pi0_fast_v2 --diagnose  # detailed HDF5 info
  python check_results.py pi0_fast_v2 pi0_set1    # check multiple folders
        """,
    )

    parser.add_argument(
        "folder",
        nargs="+",
        help="Folder name(s) or absolute path(s) containing episode_results.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print status for each episode (not just errors)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Show detailed HDF5 diagnostic info (demos available, numbering, etc.)",
    )

    args = parser.parse_args()

    for folder in args.folder:
        # Resolve folder path
        if os.path.isabs(folder):
            folder_path = folder
        else:
            folder_path = os.path.join(DEFAULT_OUTPUT_DIR, folder)

        if not os.path.exists(folder_path):
            print(f"{RED}Warning: Folder not found: {folder_path}{RESET}")
            continue

        if args.diagnose:
            diagnose_hdf5(folder_path)
        else:
            print(f"\nChecking: {folder_path}")
            results = check_folder(folder_path, verbose=args.verbose)

            if "error" not in results:
                print_results(results, folder_path)


if __name__ == "__main__":
    main()
