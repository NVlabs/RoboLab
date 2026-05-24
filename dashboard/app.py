# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""FastAPI app for the RoboLab eval results dashboard."""

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dashboard.loaders.catalog import (
    build_scene_index,
    default_task_folder,
    filter_scene_prims,
    filter_tasks_by_folder,
    get_task,
    list_task_folders,
    validate_task_folder,
    load_scene_stats,
    load_scenes,
    load_tasks,
    resolve_scenes_metadata_dir,
    task_summary,
)
from dashboard.loaders.hdf5 import episode_timeseries, list_episode_keys
from dashboard.loaders.local import LocalLoader, _score_ci, _sr_beta_ci
from dashboard.loaders.sources import SourceRegistry

PKG_DIR = Path(__file__).parent


def _resolve_dt(task_dir: Path, env_id: int, run_index: int) -> float | None:
    """Read the per-step dt for this episode.

    Lookup order (the user wants env_cfg.json to be authoritative; the rest are
    only fallbacks for legacy runs that don't ship a recoverable env_cfg):

      1. env_cfg.json → ``sim.dt × decimation``  (preferred — the canonical
         engine-side timestep × the recorder decimation factor)
      2. per-env log JSON ``dt`` field
      3. None
    """
    import json
    cfg_path = task_dir / "env_cfg.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            sim = cfg.get("sim") or {}
            sim_dt = sim.get("dt")
            decimation = cfg.get("decimation")
            if isinstance(sim_dt, (int, float)) and isinstance(decimation, (int, float)):
                return float(sim_dt) * float(decimation)
        except (OSError, json.JSONDecodeError):
            pass
    for candidate in (
        task_dir / f"log_{run_index}_env{env_id}.json",
        task_dir / f"log_{env_id}.json",
    ):
        if not candidate.exists():
            continue
        try:
            d = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and d.get("dt"):
            return float(d["dt"])
    return None


def _overview_bucket():
    return {"n": 0, "s": 0, "runs": set(), "tasks": set(), "score_means": []}


def _event_severity(name: str) -> str:
    """Map an event name to a severity bucket the frontend can color-code on.
    Pure name-pattern routing — no hardcoded list of every event in the taxonomy."""
    u = (name or "").upper()
    if u.endswith("_SUCCESS") or u == "OK":
        return "success"
    if u.endswith("_FAILURE"):
        return "failure"
    if "DROPPED" in u or "WRONG" in u or "HIT" in u:
        return "failure"
    return "neutral"


def create_app(initial_dir: Path | None = None, scenes_dir: Path | None = None) -> FastAPI:
    """Build the FastAPI app.

    ``initial_dir`` is optional. When provided AND the persisted source list
    is empty, it's used as the first source; otherwise the user adds sources
    from the sidebar (and they persist to ~/.config/robolab-dashboard).
    """
    scenes_metadata_dir = resolve_scenes_metadata_dir(str(scenes_dir) if scenes_dir else None)
    app = FastAPI(title="RoboLab Results Dashboard")

    registry = SourceRegistry()
    if initial_dir is not None:
        sources = registry.seed_if_empty(Path(initial_dir).resolve()) or registry.load()
    else:
        sources = registry.load()
    loader = LocalLoader(sources)

    templates = Jinja2Templates(directory=str(PKG_DIR / "templates"))
    static_dir = PKG_DIR / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _asset_version() -> str:
        """File-mtime hash so /static/app.js?v=… changes whenever a file does."""
        try:
            mtimes = [int(p.stat().st_mtime) for p in static_dir.rglob("*") if p.is_file()]
            return str(max(mtimes)) if mtimes else "0"
        except OSError:
            return "0"

    def _refresh_loader() -> None:
        loader.set_sources(registry.load())

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "asset_version": _asset_version()},
        )

    # ---- API: sources -------------------------------------------------------

    @app.get("/api/sources")
    def list_sources():
        return {"sources": [str(p) for p in registry.load()]}

    @app.post("/api/sources")
    def add_source(payload: dict = Body(...)):
        path_str = payload.get("path")
        if not path_str or not isinstance(path_str, str):
            raise HTTPException(status_code=400, detail="missing 'path' (string)")
        try:
            dirs = registry.add(Path(path_str))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _refresh_loader()
        return {"sources": [str(p) for p in dirs]}

    @app.delete("/api/sources")
    def remove_source(path: str):
        dirs = registry.remove(Path(path))
        _refresh_loader()
        return {"sources": [str(p) for p in dirs]}

    @app.get("/api/sources/validate")
    def validate_source(path: str):
        """Inspect a results-directory path without mutating the registry.
        Surfaces structured ``reason`` / ``message`` so the sidebar can
        flag bad paths inline instead of failing silently."""
        try:
            p = Path(path).expanduser().resolve()
        except (TypeError, ValueError) as e:
            return {"path": path, "resolved": None, "base": None, "ok": False,
                    "reason": "invalid_path", "message": str(e)}
        if not p.exists():
            return {"path": path, "resolved": str(p), "base": str(p.parent),
                    "ok": False, "reason": "not_found",
                    "message": "path does not exist"}
        if not p.is_dir():
            return {"path": path, "resolved": str(p), "base": str(p.parent),
                    "ok": False, "reason": "not_a_directory",
                    "message": "path exists but is not a directory"}
        # Soft check — count subfolders that look like run dirs. Empty/zero
        # is still ok=True (user may be adding before any runs exist) but
        # comes back with a warning so the chip shows ⚠.
        try:
            subdirs = [c for c in p.iterdir() if c.is_dir()]
        except OSError as e:
            return {"path": path, "resolved": str(p), "base": str(p.parent),
                    "ok": False, "reason": "unreadable", "message": str(e)}
        reason = "" if subdirs else "no_subdirs"
        message = "" if subdirs else (
            "directory has no subfolders — once policies write results "
            "here, runs will show up automatically"
        )
        return {"path": path, "resolved": str(p), "base": str(p.parent),
                "ok": True, "subdir_count": len(subdirs),
                "reason": reason, "message": message}

    # ---- API: overview ------------------------------------------------------

    @app.get("/api/overview")
    def overview():
        """Benchmark-wide summary: per-task SR + score across all runs + per-policy totals.

        Score variance is pooled via the law of total variance so the overview
        ships a usable std even though we don't keep raw score values around:
            E[Var]   ≈ sum(n_i * std_i^2) / sum(n_i)        (within-group)
            Var[E]   ≈ sum(n_i * (mean_i - mean_total)^2) / sum(n_i)
            std_total = sqrt(E[Var] + Var[E])
        """
        import math
        runs = loader.list_runs()
        per_task: dict[str, dict] = defaultdict(_overview_bucket)
        per_policy: dict[str, dict] = defaultdict(_overview_bucket)
        for r in runs:
            for task in loader.list_tasks(r.run_id):
                pt = per_task[task.task]
                pt["n"] += task.num_episodes
                pt["s"] += task.num_success
                pt["runs"].add(r.run_id)
                if task.mean_score is not None:
                    pt["score_means"].append((task.mean_score, task.score_n, task.score_std))
                if r.policy:
                    pp = per_policy[r.policy]
                    pp["n"] += task.num_episodes
                    pp["s"] += task.num_success
                    pp["tasks"].add(task.task)
                    if task.mean_score is not None:
                        pp["score_means"].append((task.mean_score, task.score_n, task.score_std))

        def _pool(score_means: list[tuple[float, int, float | None]]):
            n_total = sum(n for _, n, _ in score_means)
            if not n_total:
                return None, None, 0
            mean_total = sum(m * n for m, n, _ in score_means) / n_total
            within = sum(n * (s ** 2 if s is not None else 0.0) for _, n, s in score_means) / n_total
            between = sum(n * (m - mean_total) ** 2 for m, n, _ in score_means) / n_total
            total_var = within + between
            return mean_total, (math.sqrt(total_var) if n_total > 1 else None), n_total

        tasks_out = []
        for t, v in per_task.items():
            mean, std, sn = _pool(v["score_means"])
            sr_lcb, sr_ucb = _sr_beta_ci(v["s"], v["n"])
            sc_lcb, sc_ucb = _score_ci(mean, std, sn)
            tasks_out.append({"task": t, "n": v["n"], "s": v["s"],
                              "rate": (v["s"] / v["n"]) if v["n"] else 0.0,
                              "sr_lcb": sr_lcb, "sr_ucb": sr_ucb,
                              "score": mean, "score_std": std, "score_n": sn,
                              "score_lcb": sc_lcb, "score_ucb": sc_ucb,
                              "runs": sorted(v["runs"])})
        tasks_out.sort(key=lambda x: x["task"])

        policies_out = []
        for p, v in sorted(per_policy.items()):
            mean, std, sn = _pool(v["score_means"])
            sr_lcb, sr_ucb = _sr_beta_ci(v["s"], v["n"])
            sc_lcb, sc_ucb = _score_ci(mean, std, sn)
            policies_out.append({"policy": p, "n": v["n"], "s": v["s"],
                                 "rate": (v["s"] / v["n"]) if v["n"] else 0.0,
                                 "sr_lcb": sr_lcb, "sr_ucb": sr_ucb,
                                 "score": mean, "score_std": std, "score_n": sn,
                                 "score_lcb": sc_lcb, "score_ucb": sc_ucb,
                                 "tasks": len(v["tasks"])})

        return {"sources": [str(s) for s in loader.sources], "num_runs": len(runs), "tasks": tasks_out, "policies": policies_out}

    @app.get("/api/runs")
    def list_runs():
        return [asdict(r) for r in loader.list_runs()]

    # ---- catalog: tasks + scenes (pre-generated metadata, IsaacLab-free) ----

    @app.get("/api/tasks/folders")
    def api_task_folders():
        return {"folders": list_task_folders(), "default": default_task_folder()}

    @app.get("/api/tasks/validate")
    def api_tasks_validate(path: str):
        """Inspect a task-folder path and report ok/warnings without mutating
        any state. Used by the Tasks page Add-folder popover."""
        return validate_task_folder(path)

    @app.get("/api/tasks/summary")
    def api_tasks_summary(folder: list[str] | None = Query(default=None)):
        # FastAPI binds repeated ?folder=… into a list. A single ?folder=…
        # also arrives as a 1-element list, so we can pass through as-is.
        return task_summary(folder)

    @app.get("/api/tasks")
    def api_tasks(folder: list[str] | None = Query(default=None)):
        return filter_tasks_by_folder(folder)

    @app.get("/api/tasks/{name}")
    def api_task(name: str):
        t = get_task(name)
        if t is None:
            raise HTTPException(status_code=404, detail=f"task not found: {name!r}")
        return t

    @app.get("/api/scenes")
    def api_scenes():
        if scenes_metadata_dir is None:
            return {"scenes": [], "metadata_dir": None,
                    "error": "scene_metadata.json not found — pass --scenes-metadata-dir or run from a checkout with assets/"}
        idx = build_scene_index(scenes_metadata_dir)
        for s in idx:
            s["has_image"] = _scene_image_path(s["scene"]) is not None
        return {"scenes": idx, "metadata_dir": str(scenes_metadata_dir)}

    @app.get("/api/scenes/_stats")
    def api_scene_stats():
        return load_scene_stats(scenes_metadata_dir)

    def _scene_image_path(filename: str) -> Path | None:
        """Resolve <scenes_metadata_dir>/../_images/<basename>.png for a scene."""
        if scenes_metadata_dir is None:
            return None
        base = Path(filename).stem  # "banana_bowl.usda" → "banana_bowl"
        # `_images/` lives next to the scene .usda files, not next to `_metadata/`.
        for candidate in (
            scenes_metadata_dir.parent / "_images" / f"{base}.png",
            scenes_metadata_dir.parent / "_images" / f"{base}.jpg",
        ):
            if candidate.is_file():
                return candidate
        return None

    @app.get("/api/scenes/{filename}")
    def api_scene(filename: str):
        if scenes_metadata_dir is None:
            raise HTTPException(status_code=503, detail="scene metadata dir not configured")
        scenes = load_scenes(scenes_metadata_dir)
        prims = scenes.get(filename)
        if prims is None:
            raise HTTPException(status_code=404, detail=f"scene not found: {filename!r}")
        used_by = sorted({t.get("task_name") for t in load_tasks()
                          if t.get("scene") == filename and t.get("task_name")})
        return {
            "scene": filename,
            "objects": filter_scene_prims(prims),
            "all_prims": prims,
            "used_by": used_by,
            "has_image": _scene_image_path(filename) is not None,
        }

    @app.get("/api/scenes/{filename}/image")
    def api_scene_image(filename: str):
        p = _scene_image_path(filename)
        if p is None:
            raise HTTPException(status_code=404, detail=f"no preview image for {filename!r}")
        return FileResponse(p, media_type="image/png" if p.suffix == ".png" else "image/jpeg")

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str):
        try:
            tasks = loader.list_tasks(run_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"run_id": run_id, "tasks": [asdict(t) for t in tasks]}

    @app.get("/api/runs/{run_id}/tasks/{task}/episodes")
    def list_episodes(run_id: str, task: str):
        try:
            eps = loader.list_episodes(run_id, task)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return [asdict(e) for e in eps]

    @app.get("/api/runs/{run_id}/tasks/{task}/episodes/{env_id}/run/{run_index}/video")
    def episode_video(run_id: str, task: str, env_id: int, run_index: int, name: str | None = None):
        """Return the requested camera mp4. If ``name`` is omitted, return the first one
        (prefers ``viewport``)."""
        ep = loader.get_episode(run_id, task, env_id, run_index)
        if ep is None or not ep.videos:
            raise HTTPException(status_code=404, detail="no video for episode")
        if name:
            match = next((v for v in ep.videos if v.name == name), None)
            if match is None:
                raise HTTPException(status_code=404, detail=f"no camera named {name!r}")
            return FileResponse(match.path, media_type="video/mp4")
        return FileResponse(ep.videos[0].path, media_type="video/mp4")

    @app.get("/api/runs/{run_id}/tasks/{task}/episodes/{env_id}/run/{run_index}/thumb")
    def episode_thumb(run_id: str, task: str, env_id: int, run_index: int):
        ep = loader.get_episode(run_id, task, env_id, run_index)
        if ep is None or not ep.last_frame_path:
            raise HTTPException(status_code=404, detail="no thumbnail")
        return FileResponse(ep.last_frame_path, media_type="image/png")

    @app.get("/api/runs/{run_id}/tasks/{task}/episodes/{env_id}/run/{run_index}/events")
    def episode_events_route(run_id: str, task: str, env_id: int, run_index: int):
        """Per-episode events parsed from log_<run>_env<env>.json (or log_<env>.json fallback).

        Returns ``{dt, events: [{step, time_s, code, name, info, score, severity}]}``.
        ``severity`` is derived from the event name (``*_SUCCESS`` → success,
        ``*_FAILURE``/``DROPPED``/``WRONG_OBJECT``/``HIT`` → failure, else neutral)
        so the frontend can color-code markers without hardcoding the taxonomy.
        """
        import json
        try:
            run_dir = loader._run_dir(run_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        task_dir = run_dir / task
        candidates = [
            task_dir / f"log_{run_index}_env{env_id}.json",
            task_dir / f"log_{env_id}.json",
        ]
        log_path = next((p for p in candidates if p.exists()), None)
        if log_path is None:
            raise HTTPException(status_code=404, detail="no log file for this episode")
        try:
            data = json.loads(log_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"log parse: {e}")

        # dt: env_cfg.json (sim.dt × decimation) is the canonical source;
        # fall back to the log's own dt only when env_cfg is missing.
        dt: float | None = _resolve_dt(task_dir, env_id, run_index)

        # Two log shapes: dict-with-events (new), or list-of-per-step-status (legacy).
        events: list[dict] = []
        if isinstance(data, dict):
            if dt is None:
                dt = data.get("dt")
            raw = data.get("events") or []
            for ev in raw:
                if not isinstance(ev, dict):
                    continue
                step = ev.get("step")
                if step is None:
                    continue
                events.append({
                    "step": int(step),
                    "time_s": float(step) * float(dt) if dt else None,
                    "code": ev.get("code"),
                    "name": ev.get("name") or "",
                    "info": ev.get("info") or "",
                    "score": ev.get("score"),
                    "severity": _event_severity(ev.get("name") or ""),
                })
        elif isinstance(data, list):
            # legacy: list of per-step status dicts. derive events on status change.
            prev = None
            for step, row in enumerate(data):
                if not isinstance(row, dict):
                    continue
                status = row.get("status")
                if status is None or status == prev:
                    prev = status
                    continue
                events.append({
                    "step": step,
                    "time_s": None,    # no dt in legacy format
                    "code": status,
                    "name": f"STATUS_{status}",
                    "info": (row.get("info") or "").strip(),
                    "score": row.get("score"),
                    "severity": _event_severity(""),
                })
                prev = status

        return {"dt": dt, "num_events": len(events), "events": events}

    @app.get("/api/runs/{run_id}/tasks/{task}/episodes/{env_id}/run/{run_index}/timeseries")
    def episode_timeseries_route(run_id: str, task: str, env_id: int, run_index: int):
        h5 = loader.hdf5_path(run_id, task, run_index=run_index)
        if h5 is None:
            raise HTTPException(status_code=404, detail="no HDF5 with episodes for this task")
        keys = list_episode_keys(h5)
        if not keys:
            raise HTTPException(status_code=404, detail="HDF5 has no episodes")

        # dt source: prefer env_cfg.json (sim.dt × decimation) — the authoritative
        # engine-side value — then fall back to log dt. We pass it as a fallback
        # that overrides any HDF5 attr that might disagree.
        try:
            task_dir_for_dt = loader._run_dir(run_id) / task
            dt_override = _resolve_dt(task_dir_for_dt, env_id, run_index)
        except ValueError:
            dt_override = None
        # Episode key conventions vary; try common patterns then fall back to index.
        candidates = [
            f"demo_{env_id}",
            f"demo_{run_index}_{env_id}",
            f"episode_{env_id}",
        ]
        key = next((k for k in candidates if k in keys), None)
        if key is None:
            # positional fallback by env_id within the run
            idx = env_id if env_id < len(keys) else 0
            key = keys[idx]
        return episode_timeseries(h5, key, dt_override=dt_override)

    return app
