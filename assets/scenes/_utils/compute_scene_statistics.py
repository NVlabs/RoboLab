# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Scene Statistics Calculator

Computes statistics from scene_metadata.json, including:
- Average number of objects per scene
- Distribution of scenes by object count
- Per-object appearance counts and scene listings

Only considers objects whose payload matches the pattern "../objects/<dataset>/<name>.usd".
Fixtures, ground planes, physics materials, and external URLs are excluded.

Usage:
    # Print statistics to terminal:
    python compute_scene_statistics.py

    # Print statistics and save to JSON:
    python compute_scene_statistics.py --save
"""

import json
import os
import re
from collections import defaultdict
from typing import Any

# Pattern: ../objects/<dataset>/.../<filename>.usd (any USD extension, arbitrary subdirectories)
OBJECT_PAYLOAD_PATTERN = re.compile(r"^\.\./objects/([^/]+)/(.+\.usda?)$")


def load_scene_metadata(metadata_path: str) -> dict[str, list[dict[str, Any]]]:
    """Load scene_metadata.json and return its contents."""
    with open(metadata_path, "r") as f:
        return json.load(f)


def extract_object_payloads(prims: list[dict[str, Any]]) -> list[str]:
    """
    Extract valid object payload paths from a scene's prim list.

    Only payloads matching "../objects/<dataset>/<name>.usd" are included.

    Args:
        prims: List of prim dictionaries from scene metadata.

    Returns:
        List of payload path strings that match the object pattern.
    """
    object_payloads = []
    for prim in prims:
        for payload in prim.get("payload", []):
            if OBJECT_PAYLOAD_PATTERN.match(payload):
                object_payloads.append(payload)
    return object_payloads


def compute_statistics(
    metadata: dict[str, list[dict[str, Any]]],
    ignore_scenes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute scene statistics from the metadata dictionary.

    Args:
        metadata: Dictionary mapping scene filenames to lists of prim dicts.
        ignore_scenes: Optional list of scene filenames to exclude from statistics.

    Returns:
        Dictionary with the following keys:
        - "total_scenes": int
        - "total_unique_objects": int
        - "average_objects_per_scene": float
        - "object_count_distribution": dict mapping object count (str) to number of scenes
        - "per_object_stats": dict mapping object payload path to
            {"count": int, "dataset": str, "object_name": str, "scenes": list[str]}
        - "per_scene_object_counts": dict mapping scene name to object count
    """
    ignore_set = set(ignore_scenes) if ignore_scenes else set()

    per_scene_objects: dict[str, list[str]] = {}
    object_to_scenes: dict[str, list[str]] = defaultdict(list)

    for scene_name, prims in sorted(metadata.items()):
        if scene_name in ignore_set:
            continue
        payloads = extract_object_payloads(prims)
        # De-duplicate within a scene (same object payload referenced multiple times)
        unique_payloads = list(dict.fromkeys(payloads))
        per_scene_objects[scene_name] = unique_payloads

        for payload in unique_payloads:
            object_to_scenes[payload].append(scene_name)

    # --- Aggregate stats ---
    total_scenes = len(per_scene_objects)
    object_counts = [len(objs) for objs in per_scene_objects.values()]
    avg_objects = sum(object_counts) / total_scenes if total_scenes > 0 else 0.0

    # Distribution: how many scenes have N objects
    count_distribution: dict[int, int] = defaultdict(int)
    for count in object_counts:
        count_distribution[count] += 1

    # Per-object stats
    per_object_stats = {}
    for payload, scenes in sorted(object_to_scenes.items()):
        match = OBJECT_PAYLOAD_PATTERN.match(payload)
        dataset = match.group(1) if match else "unknown"
        obj_filename = os.path.basename(match.group(2)) if match else payload
        obj_name = os.path.splitext(obj_filename)[0]
        per_object_stats[payload] = {
            "dataset": dataset,
            "object_name": obj_name,
            "count": len(scenes),
            "scenes": sorted(scenes),
        }

    # Per-scene object count (for CSV / quick lookup)
    per_scene_object_counts = {
        scene: len(objs) for scene, objs in sorted(per_scene_objects.items())
    }

    return {
        "total_scenes": total_scenes,
        "total_unique_objects": len(object_to_scenes),
        "average_objects_per_scene": round(avg_objects, 2),
        "object_count_distribution": dict(sorted(count_distribution.items())),
        "per_object_stats": per_object_stats,
        "per_scene_object_counts": per_scene_object_counts,
    }


def print_statistics(stats: dict[str, Any]) -> None:
    """Pretty-print the computed statistics to the terminal."""
    print("=" * 70)
    print("SCENE STATISTICS")
    print("=" * 70)

    print(f"\nTotal scenes:              {stats['total_scenes']}")
    print(f"Total unique objects:      {stats['total_unique_objects']}")
    print(f"Average objects per scene: {stats['average_objects_per_scene']}")

    # --- Object count distribution ---
    print("\n" + "-" * 70)
    print("OBJECT COUNT DISTRIBUTION (how many scenes have N objects)")
    print("-" * 70)
    print(f"  {'# Objects':<12} {'# Scenes':<12}")
    for obj_count, scene_count in sorted(stats["object_count_distribution"].items(), key=lambda x: int(x[0])):
        print(f"  {obj_count:<12} {scene_count:<12}")

    # --- Per-scene object counts ---
    print("\n" + "-" * 70)
    print("PER-SCENE OBJECT COUNTS")
    print("-" * 70)
    for scene, count in sorted(stats["per_scene_object_counts"].items()):
        print(f"  {scene:<55} {count}")

    # --- Per-object stats (table) ---
    print("\n" + "-" * 120)
    print("PER-OBJECT STATS (object appearances across scenes)")
    print("-" * 120)

    # Sort by count descending, then by name
    sorted_objects = sorted(
        stats["per_object_stats"].items(),
        key=lambda x: (-x[1]["count"], x[1]["object_name"]),
    )

    # Compute column widths
    name_w = max(len(info["object_name"]) for _, info in sorted_objects)
    name_w = max(name_w, len("Object"))
    ds_w = max(len(info["dataset"]) for _, info in sorted_objects)
    ds_w = max(ds_w, len("Dataset"))
    count_w = max(len(str(info["count"])) for _, info in sorted_objects)
    count_w = max(count_w, len("# Scenes"))

    header = f"  {'Object':<{name_w}}  {'Dataset':<{ds_w}}  {'# Scenes':>{count_w}}  Scenes"
    print(header)
    print(f"  {'-' * name_w}  {'-' * ds_w}  {'-' * count_w}  {'-' * 60}")

    max_scenes_width = 80
    for payload, info in sorted_objects:
        scenes_str = ", ".join(info["scenes"])
        if len(scenes_str) > max_scenes_width:
            scenes_str = scenes_str[:max_scenes_width] + "..."
        print(f"  {info['object_name']:<{name_w}}  {info['dataset']:<{ds_w}}  {info['count']:>{count_w}}  {scenes_str}")

    print("\n" + "=" * 120)


def save_statistics(stats: dict[str, Any], output_path: str) -> None:
    """
    Save statistics to a JSON file.

    The JSON keys are converted to strings where necessary for JSON compatibility.
    """
    # JSON doesn't support integer keys; convert distribution keys to strings
    json_stats = stats.copy()
    json_stats["object_count_distribution"] = {
        str(k): v for k, v in stats["object_count_distribution"].items()
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_stats, f, indent=2)
    print(f"\nStatistics saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    default_metadata_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "_metadata", "scene_metadata.json"
    )
    default_output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "_metadata", "scene_statistics.json"
    )

    parser = argparse.ArgumentParser(description="Compute statistics from scene_metadata.json")
    parser.add_argument(
        "--metadata-path",
        default=default_metadata_path,
        help="Path to scene_metadata.json",
    )
    parser.add_argument(
        "--output-path",
        default=default_output_path,
        help="Path to save the statistics JSON file",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save statistics to JSON file in addition to printing",
    )
    parser.add_argument(
        "--ignore-scenes",
        nargs="*",
        default=["base_empty.usda"],
        help="List of scene filenames to exclude from statistics (default: base_empty.usda)",
    )
    args = parser.parse_args()

    metadata = load_scene_metadata(args.metadata_path)
    stats = compute_statistics(metadata, ignore_scenes=args.ignore_scenes)
    print_statistics(stats)

    if args.save:
        save_statistics(stats, args.output_path)
