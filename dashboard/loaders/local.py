# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Local filesystem loader for RoboLab eval outputs.

A "run" is a directory under ``output/`` (e.g. ``2026-05-19_12-00-50_pi05``).
A run contains one subdir per task; each task subdir has ``data.hdf5``,
``env_cfg.json``, per-env ``log_*.json``, and optionally viewport mp4s.

The run-level ``episode_results.jsonl`` (newer runs) is the canonical
source for per-episode metrics. Older runs may only have per-env JSONs;
we fall back to those.
"""

import functools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from robolab.core.logging.results import beta_ci_bounds, load_and_merge_episode_data


# Beta credible interval for SR — exact match to robolab.core.logging.results.
# lru_cache on (k, n) makes repeat lookups (very common: many buckets at the
# same n) free after the first call. 4096 entries comfortably covers a
# multi-thousand-task benchmark with mixed sample sizes.
@functools.lru_cache(maxsize=4096)
def _sr_beta_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    lo, hi = beta_ci_bounds(k, n)
    return float(lo), float(hi)


# Student-t CI for the score mean. For n>=30 it's effectively a normal CI;
# for small n it widens correctly. Same lru_cache trick.
@functools.lru_cache(maxsize=4096)
def _score_t_ci(mean_x1000: int, std_x1000: int, n: int) -> tuple[float, float]:
    # Cache key uses integer-quantized (mean, std) so two episodes with the
    # same rounded stats hit the same cache slot. Within 0.001 precision this
    # is harmless; for display we re-format anyway.
    from scipy.stats import t  # local import — scipy is already a robolab dep
    if n is None or n < 2:
        return float("nan"), float("nan")
    mean = mean_x1000 / 1000.0
    std = std_x1000 / 1000.0
    se = std / math.sqrt(n)
    half = float(t.ppf(0.975, n - 1)) * se
    return mean - half, mean + half


def _score_ci(mean: float | None, std: float | None, n: int | None) -> tuple[float | None, float | None]:
    if mean is None or std is None or n is None or n < 2:
        return None, None
    lo, hi = _score_t_ci(round(mean * 1000), round(std * 1000), int(n))
    return lo, hi


def _json_safe(v):
    """Coerce non-finite floats (NaN, ±Inf) to None so the API can JSON-encode them."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


# ---- mtime-keyed memoization -----------------------------------------------
#
# The (path, mtime) trick: lru_cache keys on both the path AND a mtime int.
# When the underlying file changes on disk, its mtime moves, so the next call
# creates a fresh cache entry instead of returning stale data. Old entries
# eventually evict by lru_cache's size cap.

@functools.lru_cache(maxsize=128)
def _cached_load_episodes(run_dir: str, _mtime: int) -> list[dict]:
    """Parse episode_results.jsonl + per-env logs for one run. mtime-invalidated."""
    return load_and_merge_episode_data(run_dir)


@functools.lru_cache(maxsize=256)
def _cached_hdf5_has_episodes(path: str, _mtime: int) -> bool:
    """True iff the HDF5 exists AND has at least one episode under /data."""
    try:
        import h5py  # local; importing at module top would slow `import dashboard`
        with h5py.File(path, "r") as f:
            return bool("data" in f and len(f["data"].keys()) > 0)
    except (OSError, Exception):
        return False


def _safe_mtime(p: Path) -> int:
    """Coarse mtime key — falls back to 0 on missing path so the cache still keys cleanly."""
    try:
        return int(p.stat().st_mtime)
    except OSError:
        return 0


def _run_mtime_key(run_dir: Path) -> int:
    """Pick the file most likely to change when eval results land.

    Newer runs write a single ``episode_results.jsonl`` at the end of the eval —
    that file's mtime is the right cache invalidator. For legacy runs without
    the JSONL, fall back to the run directory's mtime (which moves when files
    are added/removed).
    """
    jsonl = run_dir / "episode_results.jsonl"
    return _safe_mtime(jsonl) if jsonl.exists() else _safe_mtime(run_dir)


@dataclass
class CameraVideo:
    name: str            # short label, e.g. "viewport", "recording", "playback"
    path: str            # absolute fs path


@dataclass
class EpisodeRow:
    task: str
    episode: int            # episode index within (run, env) pair
    env_id: int
    run_index: int          # the "run" counter inside a multi-run eval (NOT the output dir)
    success: bool
    score: float | None     # subtask completion ratio (0..1); None if not recorded
    reason: str | None      # failure reason if success is False
    duration: float
    episode_step: int
    instruction: str
    instruction_type: str | None
    attributes: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)  # policy_inference_avg_ms, env_step_avg_ms, …
    policy: str | None = None
    videos: list[CameraVideo] = field(default_factory=list)
    last_frame_path: str | None = None   # absolute path, may be None
    has_hdf5: bool = False


@dataclass
class TaskSummary:
    task: str
    num_episodes: int
    num_success: int
    success_rate: float
    sr_lcb: float                # 95% Beta credible interval bounds for SR
    sr_ucb: float
    mean_score: float | None     # mean of per-episode `score` (None if none recorded)
    score_std: float | None      # sample std of per-episode score (None if <2 recorded)
    score_n: int                 # how many episodes contributed a finite score
    score_lcb: float | None      # 95% Student-t CI for the score mean
    score_ucb: float | None
    mean_duration: float
    instruction: str | None


@dataclass
class RunMeta:
    run_id: str             # qualified run id (source_basename/dir_name when needed)
    path: str               # absolute path to the run directory
    source: str             # absolute path to the source dir containing it
    policy: str | None
    num_tasks: int
    num_episodes: int
    num_success: int
    success_rate: float


class LocalLoader:
    """Scans one or more output roots for runs. Cheap to construct; lazy reads.

    Run IDs are typically the run directory's basename. If two source dirs
    contain a run with the same basename, the second (and onward) get prefixed
    with ``<source_basename>/`` to disambiguate.
    """

    def __init__(self, sources: list[Path] | Path):
        if isinstance(sources, (str, Path)):
            sources = [Path(sources)]
        self.sources: list[Path] = [Path(s).resolve() for s in sources]

    def set_sources(self, sources: list[Path]) -> None:
        self.sources = [Path(s).resolve() for s in sources]

    # ---- runs ----------------------------------------------------------------

    def list_runs(self) -> list[RunMeta]:
        runs: list[RunMeta] = []
        used_ids: set[str] = set()
        for src in self.sources:
            if not src.exists():
                continue
            for child in sorted(src.iterdir(), reverse=True):
                if not child.is_dir():
                    continue
                meta = self._run_meta(child, src, used_ids)
                if meta is not None:
                    runs.append(meta)
                    used_ids.add(meta.run_id)
        return runs

    def _run_meta(self, run_dir: Path, source: Path, used_ids: set[str]) -> RunMeta | None:
        eps = self._load_run_episodes(run_dir)
        if not eps:
            task_dirs = [d for d in run_dir.iterdir() if d.is_dir() and self._looks_like_task_dir(d)]
            if not task_dirs:
                return None
        policy = next((e.policy for e in eps if e.policy), None)
        success = sum(1 for e in eps if e.success)
        run_id = run_dir.name
        if run_id in used_ids:
            run_id = f"{source.name}/{run_dir.name}"
        return RunMeta(
            run_id=run_id,
            path=str(run_dir),
            source=str(source),
            policy=policy,
            num_tasks=len({e.task for e in eps}) if eps else len([d for d in run_dir.iterdir() if d.is_dir()]),
            num_episodes=len(eps),
            num_success=success,
            success_rate=(success / len(eps)) if eps else 0.0,
        )

    @staticmethod
    def _looks_like_task_dir(d: Path) -> bool:
        return (d / "data.hdf5").exists() or (d / "env_cfg.json").exists() or any(d.glob("log_*.json"))

    # ---- tasks ---------------------------------------------------------------

    def list_tasks(self, run_id: str) -> list[TaskSummary]:
        run_dir = self._run_dir(run_id)
        eps = self._load_run_episodes(run_dir)
        if not eps:
            # synthesize from subdirs without episode data
            summaries = []
            for d in sorted(run_dir.iterdir()):
                if d.is_dir() and self._looks_like_task_dir(d):
                    summaries.append(TaskSummary(task=d.name, num_episodes=0, num_success=0, success_rate=0.0,
                                                 sr_lcb=0.0, sr_ucb=1.0,
                                                 mean_score=None, score_std=None, score_n=0,
                                                 score_lcb=None, score_ucb=None,
                                                 mean_duration=0.0, instruction=None))
            return summaries

        groups: dict[str, list[EpisodeRow]] = defaultdict(list)
        for e in eps:
            groups[e.task].append(e)

        out = []
        for task in sorted(groups):
            es = groups[task]
            n = len(es)
            s = sum(1 for e in es if e.success)
            mean_dur = sum(e.duration for e in es) / n if n else 0.0
            scores = [e.score for e in es if e.score is not None]
            score_n = len(scores)
            if score_n:
                mean_score = sum(scores) / score_n
                if score_n > 1:
                    var = sum((x - mean_score) ** 2 for x in scores) / (score_n - 1)
                    score_std = var ** 0.5
                else:
                    score_std = None
            else:
                mean_score = None
                score_std = None
            sr_lcb, sr_ucb = _sr_beta_ci(s, n)
            score_lcb, score_ucb = _score_ci(mean_score, score_std, score_n)
            instr = next((e.instruction for e in es if e.instruction), None)
            out.append(TaskSummary(task=task, num_episodes=n, num_success=s,
                                    success_rate=s / n if n else 0.0,
                                    sr_lcb=sr_lcb, sr_ucb=sr_ucb,
                                    mean_score=mean_score,
                                    score_std=score_std,
                                    score_n=score_n,
                                    score_lcb=score_lcb, score_ucb=score_ucb,
                                    mean_duration=mean_dur, instruction=instr))
        return out

    # ---- episodes ------------------------------------------------------------

    def list_episodes(self, run_id: str, task: str) -> list[EpisodeRow]:
        run_dir = self._run_dir(run_id)
        eps = [e for e in self._load_run_episodes(run_dir) if e.task == task]
        task_dir = run_dir / task
        # an episode has hdf5 if EITHER data.hdf5 has episodes, OR a matching
        # run_<run_idx>.hdf5 exists with episodes.
        data_has = self._hdf5_has_episodes(task_dir / "data.hdf5")
        per_run_has = {
            int(p.stem.removeprefix("run_")): self._hdf5_has_episodes(p)
            for p in task_dir.glob("run_*.hdf5")
            if p.stem.removeprefix("run_").isdigit()
        }
        for e in eps:
            has_hdf5_eps = data_has or per_run_has.get(e.run_index, False)
            self._attach_media(task_dir, e, has_hdf5_eps=has_hdf5_eps)
        return eps

    @staticmethod
    def _hdf5_has_episodes(path: Path) -> bool:
        """True iff data.hdf5 exists AND has at least one episode under /data.
        Memoized on (path, mtime) so repeat probes are free."""
        if not path.exists():
            return False
        return _cached_hdf5_has_episodes(str(path), _safe_mtime(path))

    def get_episode(self, run_id: str, task: str, env_id: int, run_index: int) -> EpisodeRow | None:
        for e in self.list_episodes(run_id, task):
            if e.env_id == env_id and e.run_index == run_index:
                return e
        return None

    def hdf5_path(self, run_id: str, task: str, run_index: int | None = None) -> Path | None:
        """Locate the HDF5 file for this task. Two conventions exist:
          * single ``data.hdf5`` (default, IsaacLab recorder)
          * per-run ``run_<N>.hdf5`` (one file per run)
        Returns the first one that has at least one episode under ``/data``.
        """
        task_dir = self._run_dir(run_id) / task
        candidates: list[Path] = [task_dir / "data.hdf5"]
        if run_index is not None:
            candidates.append(task_dir / f"run_{run_index}.hdf5")
        # also accept any run_*.hdf5
        candidates.extend(sorted(task_dir.glob("run_*.hdf5")))
        for p in candidates:
            if p.exists() and self._hdf5_has_episodes(p):
                return p
        return None

    # ---- internals -----------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        """Resolve a run_id to an absolute path inside one of the configured sources.

        Supports two forms:
          * plain "<name>"            → first source containing a dir named <name>
          * "<source-name>/<name>"    → explicit source-qualified id used when names collide

        Guards against path traversal (rejects '..' or absolute fragments).
        """
        if run_id in ("", ".", ".."):
            raise ValueError(f"invalid run_id: {run_id!r}")
        parts = run_id.split("/")
        if any(p in ("", ".", "..") for p in parts) or len(parts) > 2:
            raise ValueError(f"invalid run_id: {run_id!r}")
        if len(parts) == 2:
            source_name, dir_name = parts
            for src in self.sources:
                if src.name == source_name and (src / dir_name).is_dir():
                    return src / dir_name
            raise ValueError(f"unknown run_id: {run_id!r}")
        # plain name — return the first source whose subdir matches
        for src in self.sources:
            candidate = src / run_id
            if candidate.is_dir():
                return candidate
        raise ValueError(f"unknown run_id: {run_id!r}")

    def _load_run_episodes(self, run_dir: Path) -> list[EpisodeRow]:
        """Delegate to the project's canonical loader (handles every output variant).
        Cached on the run dir's primary mtime so /summary and /episodes reuse the
        same parsed JSONL across requests."""
        try:
            raw = _cached_load_episodes(str(run_dir), _run_mtime_key(run_dir))
        except Exception:
            return []

        # Older runs leave run / env_id as None; synthesize a stable index per
        # (task, instruction) ordering so the UI can address episodes uniquely.
        per_task_counter: dict[str, int] = defaultdict(int)
        rows: list[EpisodeRow] = []
        for d in raw:
            task = d.get("env_name") or d.get("task_name") or "unknown"
            run_idx = d.get("run")
            env_id = d.get("env_id")
            if env_id is None:
                env_id = d.get("episode", per_task_counter[task])
            per_task_counter[task] += 1
            score_raw = d.get("score")
            try:
                score = float(score_raw) if score_raw is not None else None
            except (TypeError, ValueError):
                score = None
            if score is not None and not math.isfinite(score):
                score = None
            duration_raw = d.get("duration") or 0.0
            duration = float(duration_raw) if math.isfinite(float(duration_raw)) else 0.0
            try:
                run_index = int(run_idx) if run_idx is not None else 0
            except (TypeError, ValueError):
                # Some legacy runs set this to a string like "TaskName_0";
                # fall back to 0 rather than crashing the whole endpoint.
                run_index = 0
            try:
                episode_index = int(d.get("episode", env_id) or 0)
            except (TypeError, ValueError):
                episode_index = 0
            try:
                env_index = int(env_id or 0)
            except (TypeError, ValueError):
                env_index = 0
            rows.append(EpisodeRow(
                task=task,
                episode=episode_index,
                env_id=env_index,
                run_index=run_index,
                success=bool(d.get("success", False)),
                score=score,
                reason=d.get("reason"),
                duration=duration,
                episode_step=int(d.get("episode_step") or 0),
                instruction=d.get("instruction") or "",
                instruction_type=d.get("instruction_type"),
                attributes=list(d.get("attributes") or []),
                metrics=_json_safe(d.get("metrics") or {}),
                timing=_json_safe(d.get("timing") or {}),
                policy=d.get("policy"),
            ))
        return rows

    def _attach_media(self, task_dir: Path, e: EpisodeRow, *, has_hdf5_eps: bool | None = None) -> None:
        """Discover every mp4 belonging to (env_id, run_index) and label by suffix.

        Naming conventions seen in the wild (the ``env<N>`` token appears in
        more recent runs; the bare ``_<N>`` form is the legacy):
          * ``<instruction>_<run>_env<env>_viewport.mp4`` →  viewport (new)
          * ``<instruction>_env<env>_viewport.mp4``       →  viewport (new, run-implicit)
          * ``<instruction>_<env>_viewport.mp4``          →  viewport (legacy)
          * ``<instruction>_<run>_env<env>.mp4``          →  policy view / "recording" (new)
          * ``<instruction>_<env>.mp4``                   →  policy view / "recording" (legacy)
          * ``video_<run>_env<env>.mp4``                  →  playback-style
        """
        videos: list[CameraVideo] = []
        seen: set[Path] = set()

        def _add(name: str, p: Path) -> None:
            if p in seen:
                return
            videos.append(CameraVideo(name=name, path=str(p)))
            seen.add(p)
            if name == "viewport" and e.last_frame_path is None:
                png = p.with_name(p.stem + "_last.png")
                if png.exists():
                    e.last_frame_path = str(png)

        # Viewport: most-specific patterns first.
        for pat in (f"*_env{e.env_id}_viewport.mp4", f"*_{e.env_id}_viewport.mp4"):
            for p in task_dir.glob(pat):
                _add("viewport", p)

        # Recording (non-viewport policy/cam mp4s).
        for pat in (f"*_env{e.env_id}.mp4", f"*_{e.env_id}.mp4"):
            for p in task_dir.glob(pat):
                if p in seen:
                    continue
                # The bare-``_<N>`` glob would also catch ``..._<N>_viewport.mp4``
                # if env_id happens to be a suffix of "viewport" — guard it.
                if p.stem.endswith("_viewport"):
                    continue
                _add("recording", p)

        playback = task_dir / f"video_{e.run_index}_env{e.env_id}.mp4"
        if playback.exists():
            _add("playback", playback)

        e.videos = videos
        if has_hdf5_eps is None:
            has_hdf5_eps = self._hdf5_has_episodes(task_dir / "data.hdf5")
        e.has_hdf5 = has_hdf5_eps
