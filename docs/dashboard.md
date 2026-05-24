# Dashboard

A self-contained web dashboard for browsing the RoboLab benchmark and inspecting eval results. Boots in seconds, binds `0.0.0.0` so anyone on your LAN can reach it via your IP.

<video src="images/robolab_dashboard.mp4" autoplay controls muted loop playsinline width="800">
  Your viewer doesn't render inline video — see <a href="images/robolab_dashboard.mp4">robolab_dashboard.mp4</a>.
</video>

## Quickstart

```bash
# Already installed if you ran `uv sync` / `uv pip install -e .` — the
# dashboard ships in the default RoboLab dependency set.
source .venv/bin/activate

# Bare minimum: starts empty, add output directories from the sidebar.
robolab-dashboard

# Or seed it with one up-front (still editable in the UI):
robolab-dashboard --output-dir /path/to/output
# → Serving /path/to/output on http://0.0.0.0:8080
```

Then open `http://localhost:8080` locally or `http://<your-lan-ip>:8080` from another machine.

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--output-dir <path>` | *(none — optional)* | Optional initial results directory. If omitted, the dashboard starts with whatever sources you previously added from the sidebar (persisted to `~/.config/robolab-dashboard/sources.json`). Add or remove sources at runtime from the sidebar. |
| `--scenes-metadata-dir <path>` | auto-detected (see resolution order below) | Where to find `scene_metadata.json` + `_images/`. Only needed if running from a worktree that has `assets/` sparse-excluded. Also reads `$ROBOLAB_SCENES_METADATA_DIR`. Resolution order: CLI flag → env var → `<PACKAGE_DIR>/assets/scenes/_metadata` → sibling `robolab/` checkout. |
| `--port <int>` | `8080` | Bind port. |
| `--host <ip>` | `0.0.0.0` | Bind host. `127.0.0.1` to restrict to localhost. |
| `--reload` | off | Auto-reload on Python edits (dev). |

## What it shows

### Scenes

Browse every USD scene in the asset library — a card grid with a preview thumbnail per scene, the object count, and the number of tasks that reference each one. Click into any scene to see its full prim list (name, payload USD path, description, static-body flag) and a back-link to every task that uses it.

Data sources:
- `assets/scenes/_metadata/scene_metadata.json` — per-scene prim manifests
- `assets/scenes/_metadata/scene_statistics.json` — aggregate stats
- `assets/scenes/_images/<scene>.png` — preview images served via `/api/scenes/<file>/image`

### Tasks

Browse benchmark tasks defined under `robolab/tasks/<folder>/`. The folder dropdown defaults to `constants.DEFAULT_TASK_SUBFOLDERS[0]` (i.e. `benchmark`); any other subfolder with `.py` files (e.g. `test_tasks`, `randomize_initial_pose`) can be selected.

Above the task list, a summary card shows aggregate counts (Tasks · Unique scenes · Avg variants/task · Avg episode length) plus stackable **mini-card filters**: click a difficulty (simple / moderate / complex) or attribute (semantics, spatial, color, …) to filter the list. `Ctrl/Cmd-click` stacks multiple selections (OR within a kind, AND across kinds).

Per-task detail shows all instruction variants, a clickable scene preview, contact objects, terminations, subtasks, and subtask-stage counts. Click the scene to jump to its detail page.

Data sources:
- `robolab/tasks/_metadata/task_metadata.json` — pre-generated task manifest (run the standard `task_metadata` regen step to update)
- `robolab/tasks/_metadata/task_timing.json` — optional per-task wall-clock timing
- These JSONs are read directly; the dashboard never imports IsaacLab.

### Results

Add your experiment output directory to the left hand side bar. Each experiment folder must contain Task output folders, and an `episode_results.jsonl`. The dashboard parses each `hdf5` and compiles a report based on the data. If videos are available, they are shown.

All SR / Score cells carry **95% confidence intervals** with the half-width annotation: `29.7% [24.5–35.4] ±5.4`. SR uses an exact Beta credible interval (see `robolab.core.logging.results.beta_ci_bounds`); Score uses Student-t.

## Hosting on the LAN

The default `--host 0.0.0.0` binding makes the dashboard reachable on your machine's LAN IP. Share the URL with a colleague (e.g. `http://10.29.92.141:8080`) and they can browse without setting anything up locally. Tighten to `--host 127.0.0.1` if you'd rather keep it loopback-only.
