# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Read pre-generated task + scene metadata.

These JSON files are produced by the offline scripts that scan the RoboLab
benchmark and assets directories. We just consume them; importing the
task modules themselves would pull in IsaacLab, which the dashboard
deliberately keeps off its runtime path.

  * Tasks ship inside the package: ``robolab/tasks/_metadata/``
       - task_metadata.json: [{task_name, instruction, instruction_variants,
                               scene, contact_objects, attributes,
                               difficulty_score, difficulty_label, …}]
       - task_timing.json:   [{task_name, wall_total_s, policy_inference_avg_ms,
                               env_step_avg_ms, episode_s, it_per_sec}]
  * Scenes live alongside the assets, NOT in this worktree's sparse-checkout:
       - scene_metadata.json:   {scene_filename: [prim_dict, …]}
       - scene_statistics.json: {total_scenes, total_unique_objects, …}
    The path is configurable so users can point at any robolab checkout
    that has assets/.
"""

import functools
import json
from collections import defaultdict
from pathlib import Path

# Prims we don't want to surface as "objects" — they're scaffolding the user
# doesn't care about when browsing scene contents.
_SCENE_PRIM_SKIP = {
    "Looks", "PhysicsScene", "PhysicsMaterial", "GroundPlane",
    "DistantLight", "DomeLight", "SphereLight",
}


@functools.lru_cache(maxsize=8)
def _load_json(path: str, _mtime: int) -> object:
    return json.loads(Path(path).read_text())


def _safe_mtime(p: Path) -> int:
    try:
        return int(p.stat().st_mtime)
    except OSError:
        return 0


# ---- tasks ----------------------------------------------------------------

def task_metadata_path() -> Path:
    """Path to the bundled task_metadata.json.

    Resolves relative to robolab.constants.PACKAGE_DIR so this works inside
    any checkout (worktree or main).
    """
    import robolab.constants as rc
    return Path(rc.PACKAGE_DIR) / "robolab" / "tasks" / "_metadata" / "task_metadata.json"


def task_timing_path() -> Path:
    import robolab.constants as rc
    return Path(rc.PACKAGE_DIR) / "robolab" / "tasks" / "_metadata" / "task_timing.json"


def load_tasks() -> list[dict]:
    """Return the merged task list. Each entry includes timing fields when
    they exist in task_timing.json for the same task name."""
    meta_p = task_metadata_path()
    if not meta_p.exists():
        return []
    tasks = _load_json(str(meta_p), _safe_mtime(meta_p))
    if not isinstance(tasks, list):
        return []
    timing_p = task_timing_path()
    timing_by_name: dict[str, dict] = {}
    if timing_p.exists():
        raw = _load_json(str(timing_p), _safe_mtime(timing_p))
        if isinstance(raw, list):
            for t in raw:
                if isinstance(t, dict) and t.get("task_name"):
                    timing_by_name[t["task_name"]] = t
    out = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        merged = dict(t)
        if t.get("task_name") in timing_by_name:
            merged["timing"] = timing_by_name[t["task_name"]]
        out.append(merged)
    return out


def get_task(name: str) -> dict | None:
    for t in load_tasks():
        if t.get("task_name") == name:
            return t
    return None


def task_dir() -> Path:
    """Absolute path to robolab/tasks/ (parent of the folders we list)."""
    import robolab.constants as rc
    return Path(rc.TASK_DIR)


def default_task_folder() -> str:
    """First entry of constants.DEFAULT_TASK_SUBFOLDERS — the canonical folder
    the dashboard should show by default."""
    import robolab.constants as rc
    folders = getattr(rc, "DEFAULT_TASK_SUBFOLDERS", []) or ["benchmark"]
    return folders[0] if folders else "benchmark"


def list_task_folders() -> list[str]:
    """Discoverable folders under robolab/tasks/ that contain at least one
    .py file. Walks recursively (up to 4 levels deep) so nested subtask
    folders are surfaced. Returns ``/``-separated relative paths in DFS
    order with parents before children, e.g. ``["benchmark",
    "benchmark/easy", "benchmark/hard", "general"]``.

    Folders whose name starts with ``_`` or ``.`` (e.g. ``_metadata``,
    ``_utils``, ``_wip``, ``__pycache__``) are treated as private and
    skipped along with their entire subtree."""
    root = task_dir()
    if not root.is_dir():
        return []
    out: list[str] = []
    MAX_DEPTH = 4

    def walk(p: Path, prefix: str, depth: int) -> None:
        if depth > MAX_DEPTH:
            return
        for child in sorted(p.iterdir()):
            if not child.is_dir() or child.name.startswith("_") or child.name.startswith("."):
                continue
            rel = f"{prefix}{child.name}" if prefix else child.name
            if any(child.glob("*.py")):
                out.append(rel)
            walk(child, f"{rel}/", depth + 1)

    walk(root, "", 0)
    return out


def filter_tasks_by_folder(folder: str | list[str] | None) -> list[dict]:
    """Tasks whose ``filename`` lives under ``<folder>/...``.

    ``folder`` may be:
      * ``None``      — return everything in the metadata file.
      * ``str``       — single folder (legacy single-select callers).
      * ``list[str]`` — multiple folders. Returns the union of matches,
        deduplicated by task name (a given task lives in exactly one file
        under exactly one folder, so dedup is usually a no-op).
    """
    tasks = load_tasks()
    if not folder:
        return tasks
    folders = [folder] if isinstance(folder, str) else list(folder)
    prefixes = [f.strip("/").rstrip("/") + "/" for f in folders if f]
    if not prefixes:
        return tasks
    out: list[dict] = []
    seen: set[str] = set()
    for t in tasks:
        fn = t.get("filename") or ""
        if not any(fn.startswith(p) for p in prefixes):
            continue
        name = t.get("name") or fn
        if name in seen:
            continue
        seen.add(name)
        out.append(t)
    return out


def resolve_task_folder(path: str) -> tuple[Path | None, Path | None]:
    """Resolve a user-supplied task-folder path to an absolute Path.

    Returns ``(resolved, base)`` where ``base`` is the search root that
    matched — useful for the UI to group folders by where they came from
    ("everything that resolved against robolab/tasks/", etc).

    Resolution order:
      1. Preset/relative — ``<robolab/tasks>/<path>``.
      2. Absolute — ``path`` itself.
      3. Project-root-relative — ``<repo>/<path>``.
      4. CWD-relative — ``<cwd>/<path>``.

    Returns ``(None, None)`` if no candidate is an existing directory.
    """
    p_in = str(path or "").strip()
    if not p_in:
        return None, None
    tasks_root = task_dir()
    repo_root = tasks_root.parent.parent
    cwd = Path.cwd()
    candidates: list[tuple[Path, Path]] = []
    p_user = Path(p_in).expanduser()
    if p_user.is_absolute():
        # Only attempt the absolute branch — never combine an absolute path
        # with a search root (``tasks_root / "/abs"`` would silently equal
        # ``/abs`` and mis-report tasks_root as the base).
        candidates.append((p_user, p_user.parent))
    else:
        candidates.append((tasks_root / p_in, tasks_root))
        candidates.append((repo_root / p_in, repo_root))
        candidates.append((cwd / p_in, cwd))
    for c, root in candidates:
        try:
            if c.is_dir():
                return c.resolve(), root.resolve()
        except OSError:
            continue
    return None, None


def validate_task_folder(path: str) -> dict:
    """Inspect a user-supplied task-folder path and report whether it can be
    used. Surfaces a structured ``reason`` string for the UI to display.

    Reasons:
      * ``not_found``       — couldn't resolve to an existing directory.
      * ``no_py_files``     — directory exists but contains no ``*.py``.
      * ``no_metadata``     — folder has python files but no entries in
                              task_metadata.json (counts won't be accurate).
      * ``""`` (empty)      — ok.
    """
    resolved, base = resolve_task_folder(path)
    if resolved is None:
        return {"path": path, "resolved": None, "base": None, "ok": False,
                "py_count": 0, "metadata_count": 0, "reason": "not_found",
                "message": "path does not resolve to a directory"}
    py_files = list(resolved.glob("*.py"))
    py_count = len(py_files)
    if py_count == 0:
        return {"path": path, "resolved": str(resolved), "base": str(base),
                "ok": False, "py_count": 0, "metadata_count": 0,
                "reason": "no_py_files",
                "message": "directory has no .py task files"}
    # Try to count metadata entries — only meaningful for folders that live
    # under robolab/tasks/ (others won't appear in task_metadata.json).
    try:
        rel = resolved.relative_to(task_dir())
        metadata_count = len(filter_tasks_by_folder(str(rel)))
    except ValueError:
        metadata_count = 0
    reason = "" if metadata_count > 0 else "no_metadata"
    message = "" if metadata_count > 0 else (
        "directory has python files but no entries in task_metadata.json — "
        "task counts and stats won't be accurate until metadata is regenerated. "
        "Run: python robolab/tasks/_utils/generate_task_metadata.py"
    )
    return {"path": path, "resolved": str(resolved), "base": str(base),
            "ok": True, "py_count": py_count,
            "metadata_count": metadata_count,
            "reason": reason, "message": message}


def task_summary(folder: str | list[str] | None) -> dict:
    """Aggregate stats over one or more folders of tasks (union, deduped by
    task name) for the Tasks-index header."""
    tasks = filter_tasks_by_folder(folder)
    if not tasks:
        return {"folder": folder, "total": 0}
    unique_scenes = {t.get("scene") for t in tasks if t.get("scene")}
    difficulty: dict[str, int] = {}
    attributes: dict[str, int] = {}
    variant_counts: list[int] = []
    ep_lens: list[float] = []
    for t in tasks:
        d = (t.get("difficulty_label") or "unknown").strip() or "unknown"
        difficulty[d] = difficulty.get(d, 0) + 1
        for a in (t.get("attributes") or "").split(","):
            a = a.strip()
            if a:
                attributes[a] = attributes.get(a, 0) + 1
        variants = t.get("instruction_variants") or {}
        if isinstance(variants, dict):
            variant_counts.append(len(variants))
        try:
            ep_lens.append(float(t.get("episode_s") or 0))
        except (TypeError, ValueError):
            pass
    return {
        "folder": folder,
        "total": len(tasks),
        "unique_scenes": len(unique_scenes),
        "difficulty": dict(sorted(difficulty.items(), key=lambda kv: -kv[1])),
        "attributes": dict(sorted(attributes.items(), key=lambda kv: -kv[1])),
        "avg_instruction_variants": (sum(variant_counts) / len(variant_counts)) if variant_counts else None,
        "avg_episode_s": (sum(ep_lens) / len(ep_lens)) if ep_lens else None,
    }


# ---- scenes ---------------------------------------------------------------

def _candidate_scene_dirs(override: str | None) -> list[Path]:
    """Where to look for assets/scenes/_metadata/.

    Resolution order:
      1. CLI / function-call override
      2. ``ROBOLAB_SCENES_METADATA_DIR`` environment variable
      3. ``<robolab.constants.PACKAGE_DIR>/assets/scenes/_metadata``
         — works when this dashboard runs inside a checkout that has the
         assets directory present.
      4. ``<robolab.constants.PACKAGE_DIR>/../robolab/assets/scenes/_metadata``
         — sibling main checkout, typical when running from a worktree that
         excludes ``assets/`` via sparse-checkout.

    No user-specific path is ever assumed; if none of the candidates resolve,
    the dashboard will render the Scenes section with a message asking the
    user to pass ``--scenes-metadata-dir``.
    """
    import os
    out: list[Path] = []
    if override:
        out.append(Path(override))
    env = os.environ.get("ROBOLAB_SCENES_METADATA_DIR")
    if env:
        out.append(Path(env))
    try:
        import robolab.constants as rc
        out.append(Path(rc.PACKAGE_DIR) / "assets" / "scenes" / "_metadata")
        out.append(Path(rc.PACKAGE_DIR).parent / "robolab" / "assets" / "scenes" / "_metadata")
    except Exception:
        pass
    return out


def resolve_scenes_metadata_dir(override: str | None = None) -> Path | None:
    """First candidate that contains scene_metadata.json wins."""
    for c in _candidate_scene_dirs(override):
        if (c / "scene_metadata.json").exists():
            return c
    return None


def load_scenes(scenes_dir: Path | None) -> dict[str, list[dict]]:
    if scenes_dir is None:
        return {}
    p = scenes_dir / "scene_metadata.json"
    if not p.exists():
        return {}
    data = _load_json(str(p), _safe_mtime(p))
    return data if isinstance(data, dict) else {}


def load_scene_stats(scenes_dir: Path | None) -> dict:
    if scenes_dir is None:
        return {}
    p = scenes_dir / "scene_statistics.json"
    if not p.exists():
        return {}
    data = _load_json(str(p), _safe_mtime(p))
    return data if isinstance(data, dict) else {}


def filter_scene_prims(prims: list[dict]) -> list[dict]:
    """Drop scaffolding prims (Looks / PhysicsScene / lights / etc.)."""
    return [p for p in prims if isinstance(p, dict) and p.get("name") not in _SCENE_PRIM_SKIP]


def build_scene_index(scenes_dir: Path | None) -> list[dict]:
    """List of scenes with summary fields (object count, plus tasks that use it)."""
    scenes = load_scenes(scenes_dir)
    used_by = defaultdict(list)
    for t in load_tasks():
        sc = t.get("scene")
        if sc:
            used_by[sc].append(t.get("task_name") or t.get("filename") or "?")
    out = []
    for fname, prims in sorted(scenes.items()):
        if not isinstance(prims, list):
            continue
        objects = filter_scene_prims(prims)
        out.append({
            "scene": fname,
            "num_prims": len(prims),
            "num_objects": len(objects),
            "used_by": sorted(set(used_by.get(fname, []))),
        })
    return out
