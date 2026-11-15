# Running Environments

Once you have [registered environments](environment_registration.md), this page covers how to create, step, and evaluate them — from basic single-task runs to full multi-task benchmark evaluation.

## Creating an Environment

Use `create_env` to create and initialize an environment from a registered name or a configuration object. This supports running multiple environments sequentially without restarting the simulation.

```python
from robolab.core.environments.runtime import create_env

env, env_cfg = create_env(
    "BananaInBowlTask",       # Registered environment name
    device="cuda:0",
    num_envs=1,
    use_fabric=True,
)

obs, _ = env.reset()
# ... step the environment ...
env.close()
```

**`create_env` parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scene` | `str` or `ManagerBasedEnvCfg` | *(required)* | Registered environment name, or a pre-built env config object |
| `device` | `str` | `"cuda:0"` | Device for simulation |
| `seed` | `int` | `0` | Random seed |
| `num_envs` | `int` | `1` | Number of parallel environments |
| `env_spacing` | `float` | `5.0` | Distance in meters between environment origins (see [Multi-Environment Configuration](#multi-environment-configuration)) |
| `use_fabric` | `bool` | `True` | Use Fabric for physics (see IsaacLab docs) |
| `events` | `dict` or configclass | `None` | Event configuration to merge into the environment (e.g., pose randomization, camera variation) |
| `instruction_type` | `str` or list of `str` | `None` | Optional instruction fields to expose; all fields are exposed by default |
| `policy` | `str` | `None` | Policy backend name, stored on `env_cfg` for downstream use |
| `renderer` | `str` | `"realtime"` | RTX renderer: `"realtime"` (RaytracedLighting) or `"pathtracing"` (PathTracing). Recorded in `env_cfg.json` (see [Renderer Selection](#renderer-selection)) |
| `rendering_mode` | `str` | `None` | Realtime quality preset (`"performance"`, `"balanced"`, `"quality"`); `None` leaves IsaacLab's default (`balanced`) |
| `eye` | `tuple` | `None` | Camera eye position override |
| `lookat` | `tuple` | `None` | Camera look-at position override |

## Querying Registered Environments

Use `get_envs` to retrieve environment names by task, tag, or all registered environments:

```python
from robolab.core.environments.factory import get_envs

all_envs = get_envs()                                # All registered environments
task_envs = get_envs(task="BananaInBowlTask")        # All variants of a task
task_envs = get_envs(task=["BananaInBowlTask", "RubiksCubeTask"])  # Multiple tasks
tag_envs = get_envs(tag="pick_place")                # All environments with a tag
tag_envs = get_envs(tag=["spatial", "simple"])        # Multiple tags
```

## Instruction Field Selection

When a task defines multiple instruction fields, you can expose all of them or
request a subset:

```python
env, env_cfg = create_env(
    "BananaInBowlTask",
    device="cuda:0",
    num_envs=1,
    instruction_type=["vague", "video"],
)

print(env_cfg.instruction)             # {"vague": "Put stuff...", "video": "demo.mp4"}
print(env_cfg._instruction_variants)   # The raw instruction field (dict or str)
```

The policy receives a dictionary-like object. `instruction["vague"]` returns a
normal string and records `"vague"` in `instruction_fields_used`. Inspecting
keys does not count as use. `instruction_fields` records what was exposed.

Exposing a field does not automatically transmit it. A context-aware policy
client can request selected fields and receive them exactly once before the
first inference call. See
[Evaluating a New Policy — Episode Context](policy.md#optional-send-episode-context-once)
for the client hooks and transport lifecycle.

For CLI evaluation, use `--instruction-type vague video`. Omitting the flag
exposes all fields. Existing single-string tasks and single-value CLI use remain
supported.

## Recorder Manager Patch

RoboLab uses a patched recorder manager for fine-grained control over HDF5 data recording. This must be set up **before** registering environments:

```python
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Patch recorder manager (before registration)
from robolab.core.logging.recorder_manager import patch_recorder_manager
patch_recorder_manager()

# Register environments (after patching)
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
auto_register_droid_envs()
```

At the end of each episode, explicitly call `end_episode` to flush recorded data to the HDF5 file:

```python
from robolab.core.environments.runtime import create_env, end_episode

env, env_cfg = create_env("BananaInBowlTask", device="cuda:0", num_envs=1, use_fabric=True)

# ... run episode ...

end_episode(env)   # Exports recorded data to run_*.hdf5
env.close()
```

When sensor images have different heights, `unpack_image_obs()` keeps each
individual view at its requested scale and pads shorter views at the bottom
when building `combined_image`. This allows sensor-video recording without
changing the images sent to a policy.

## Multi-Environment Configuration

When running with `num_envs > 1`, IsaacLab spawns multiple copies of the scene in a grid layout. The `env_spacing` parameter controls the distance (in meters) between environment origins.

**Setting environment spacing:**

```python
env, env_cfg = create_env(
    "BananaInBowlTask",
    device="cuda:0",
    num_envs=20,
    env_spacing=5.0,        # 5 meters between environment origins (default)
)
```

Or when using `parse_env_cfg` directly:

```python
env_cfg = parse_env_cfg("BananaInBowlTask",
    device="cuda:0", num_envs=20, env_spacing=5.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_envs` | `int` | `1` | Number of parallel environments to spawn |
| `env_spacing` | `float` | `5.0` | Distance in meters between environment origins in the grid layout |

**Choosing `env_spacing`:** The spacing must be large enough that objects in adjacent environments do not overlap or interfere with each other. The default of 5.0m is sufficient for most tabletop manipulation tasks. Reduce it (e.g., 2.0m) for compact scenes to fit more environments in the viewport, or increase it if your scene has large objects or long-reach robot motions.

**Choosing `num_envs`:** The number of parallel environments is limited by GPU VRAM. As a rough guide, a single DROID tabletop environment with one camera uses ~1–2 GB; scale accordingly. For empirically measured per-task ceilings on L40 (48 GB), see the [num_envs VRAM size guide](env_vram_size_guide.md). If you run out of memory, reduce `num_envs` and increase `--num-runs` to reach the same total episode count:

```bash
# 20 episodes total: 20 envs × 1 run (if VRAM allows)
python policies/pi0_family/run.py --headless --num_envs 20

# 20 episodes total: 10 envs × 2 runs (if 20 doesn't fit)
python policies/pi0_family/run.py --headless --num_envs 10 --num-runs 2
```

**Multi-env episode handling:** With `num_envs > 1`, each environment runs an independent episode. The built-in `run_eval.py` handles per-env termination, video recording, and result logging automatically. If you are writing a custom evaluation loop, see `robolab/eval/episode.py` for the multi-env episode runner pattern (`from robolab.eval import run_episode`), which manages per-env video writers, independent termination tracking, and batched policy inference.

## Initial Condition Randomization

To randomize where objects start between episodes, add `--init-pose-noise X` to any policy runner:

```bash
python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask --init-pose-noise 0.1
```

Objects that touch move together, objects that cannot move safely stay put, and every episode's layout is reproducible from `--seed`. See [Initial-Pose Randomization](init_pose_randomization.md) for the options and how layouts are sampled.

Other reset events, such as camera pose variation, can be passed to `create_env` through its `events` parameter. They run after the default scene reset, in the order given:

```python
from robolab.core.events.reset_camera import RandomizeCameraPoseUniform

events = RandomizeCameraPoseUniform.from_params(
    cameras=["over_shoulder_left_camera"],
    pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
)
env, env_cfg = create_env("BananaInBowlTask", events=events)
```

## Writing an Evaluation Script

A typical evaluation script follows this pattern: register environments, loop over tasks and episodes, run inference, log results. The example below shows the full pipeline with multi-task evaluation, video recording, trajectory metrics, and result logging.

The registration import at the top determines which environments are available. For DROID with joint-position actions, use the built-in registration directly. If you set up a custom registration (see [Environment Registration](environment_registration.md)), import your own function instead.

```python
# run_eval.py

import argparse
import cv2  # Must import before isaaclab
import os
import re
import sys
import traceback

from isaaclab.app import AppLauncher
from robolab.constants import get_timestamp, DEFAULT_TASK_SUBFOLDERS

# ── CLI args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate my policy on RoboLab benchmark")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs="+", default=None, help="Specific task(s) to evaluate")
parser.add_argument("--tag", nargs="+", default=None, help="Tag(s) of tasks to evaluate")
parser.add_argument("--task-dirs", nargs="+", default=DEFAULT_TASK_SUBFOLDERS)
parser.add_argument("--num-runs", type=int, default=1)
parser.add_argument("--remote-host", type=str, default="localhost")
parser.add_argument("--remote-port", type=int, default=8000)
parser.add_argument("--output-folder-name", type=str, default=None)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports that require IsaacSim runtime ─────────────────────────────────
import torch
from tqdm import tqdm

import robolab.constants
from robolab.constants import PACKAGE_DIR, set_output_dir
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env, end_episode
from robolab.core.logging.recorder_manager import patch_recorder_manager
from robolab.core.logging.results import (
    check_all_episodes_complete, check_run_complete,
    init_experiment, update_experiment_results, summarize_experiment_results,
)
from robolab.core.observations.observation_utils import unpack_image_obs
from robolab.core.utils.video_utils import VideoWriter

# ── Register environments ─────────────────────────────────────────────────
# Option A: Use the built-in DROID joint-position registration
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)

# Option B: Use your own custom registration (see environment_registration.md)
# from my_policy.register_envs import register_envs
# register_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)

# ── Import YOUR inference client ──────────────────────────────────────────
from my_policy.inference_client import MyPolicyClient

patch_recorder_manager()


def run_episode(env, env_cfg, client, episode, headless=False):
    """Run a single policy-controlled episode (single-env example).

    For multi-env, see robolab/eval/episode.py which handles per-env
    video writers, per-env policy clients, and independent termination.
    """
    obs, _ = env.reset()
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    instruction = env_cfg.instruction

    cleaned = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")
    output_dir = robolab.constants.get_output_dir()
    # For multi-env (num_envs > 1), use f"{cleaned}_{episode}_env{env_id}.mp4"
    video_writer = VideoWriter(os.path.join(output_dir, f"{cleaned}_{episode}.mp4"), video_fps)

    success = None
    episode_step = 0

    for step in tqdm(range(max_steps)):
        ret = client.infer(obs, instruction)

        if not headless:
            cv2.imshow(instruction, cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        action = torch.tensor(ret["action"]).unsqueeze(0)
        obs, reward, term, trunc, info = env.step(action)

        frame = unpack_image_obs(obs, scale=0.5).get("combined_image")
        video_writer.write(frame)

        episode_step = step
        if term:
            success = True
            break
        if trunc:
            success = False
            break

    video_writer.release()
    client.reset()
    return success, episode_step


def main():
    if args_cli.output_folder_name is None:
        args_cli.output_folder_name = get_timestamp() + "_my_policy"
    output_dir = os.path.join(PACKAGE_DIR, "output", args_cli.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    task_envs = get_envs(task=args_cli.task) if args_cli.task else get_envs(tag=args_cli.tag) if args_cli.tag else get_envs()
    episode_results_file, episode_results = init_experiment(output_dir)

    # Create the inference client once (shared across tasks)
    client = MyPolicyClient(remote_host=args_cli.remote_host, remote_port=args_cli.remote_port)

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        total_episodes = args_cli.num_runs * args_cli.num_envs
        if check_all_episodes_complete(episode_results=episode_results, env_name=task_env, num_episodes=total_episodes):
            print(f"[MyPolicy] Task `{task_env}` already done. Skipping.")
            continue

        env, env_cfg = create_env(task_env, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)

        for i in range(args_cli.num_runs):
            if check_run_complete(episode_results=episode_results, env_name=task_env, episode=i):
                continue

            print(f"[MyPolicy] Running {task_env} episode {i}: '{env_cfg.instruction}'")
            success, episode_step = run_episode(env, env_cfg, client, i, headless=args_cli.headless)

            end_episode(env)
            dt = env_cfg.sim.dt * env_cfg.decimation
            run_summary = {
                "env_name": task_env,
                "task_name": env_cfg._task_name,
                "episode": i,
                "instruction": env_cfg.instruction,
                "success": success,
                "episode_step": episode_step,
                "duration": episode_step * dt,
                "dt": dt,
            }
            episode_results = update_experiment_results(
                run_summary=run_summary,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
            )

        env.close()

    summarize_experiment_results(episode_results)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[MyPolicy] Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
```

## `run_eval.py` CLI Reference

The built-in `policies/pi0_family/run.py` supports the full set of evaluation features:

| Flag | Description | Default |
|------|-------------|---------|
| `--task TASK [TASK ...]` | Specific task name(s) to evaluate | all registered tasks |
| `--tag TAG [TAG ...]` | Evaluate all tasks with the given tag(s) | `None` |
| `--task-dirs DIR [DIR ...]` | Task directories to scan for registration | `DEFAULT_TASK_SUBFOLDERS` |
| `--policy BACKEND` | Policy backend (`pi0`, `pi0_fast`, `pi05`, `gr00t`, etc.) | `pi05` |
| `--num_envs N` | Number of parallel environments (= number of episodes per task) | `1` |
| `--num-runs N` | Number of sequential runs (default 1). Only increase if GPU memory limits `num_envs`. Total episodes = `num_runs * num_envs`. | `1` |
| `--instruction-type FIELD [FIELD ...]` | Optional instruction fields to expose | all fields |
| `--remote-host HOST` | Policy server host | `localhost` |
| `--remote-port PORT` | Policy server port | `8000` |
| `--output-folder-name NAME` | Output folder under `output/`. Reusing a previous folder skips completed episodes. | `<timestamp>_<policy>` |
| `--disable-subtask` | Disable subtask progress checking (drops score, reason, subtask log from results; tracking is on by default) | `False` |
| `--enable-gt-state` | Export per-env ground-truth simulator state to the inference client each step (`obs["gt_state"] = {env_id: state}`). See [Ground-Truth State Export](#ground-truth-state-export). | `False` |
| `--record-image-data` | Record image observations to HDF5 | `False` |
| `--video-mode MODE` | Which videos to save: `all` (sensor + viewport), `viewport` only, `sensor` only, or `none` | `all` |
| `--renderer MODE` | RTX renderer: `realtime` (RaytracedLighting) or `pathtracing` (PathTracing). See [Renderer Selection](#renderer-selection). | `realtime` |
| `--rendering-type MODE` | Realtime quality preset: `performance`, `balanced`, or `quality`. No effect under `--renderer pathtracing`. | IsaacLab default (`balanced`) |
| `--randomize-background` | Sample a random non-default background per task at registration time. The chosen texture is recorded in each task's `env_cfg.json`. See [Backgrounds — Per-Run Random Background per Task](background.md#per-run-random-background-per-task). | `False` |
| `--background-seed N` | Seed for reproducible per-task background sampling. Used with `--randomize-background`. | `None` |
| `--headless` | Run without live display window. **Recommended for multi-task runs** — see [GPU VRAM leak in non-headless mode](debug.md#gpu-vram-leak-in-non-headless-mode-across-environment-reloads) | `False` |
| `--enable-verbose` | Verbose output | `False` |
| `--enable-debug` | Debug output | `False` |

### Renderer Selection

Both `--renderer` modes are RTX renderers; the flag selects which one IsaacLab configures via [`RenderCfg`](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.RenderCfg) (it sets the `/rtx/rendermode` carb value).

- **`realtime`** (`RaytracedLighting`) — the default. Fast, deterministic per frame, and compatible with the tiled multi-env cameras the eval pipeline uses. This is what you want for evaluation.
- **`pathtracing`** (`PathTracing`) — a physically-based renderer with accurate global illumination, ambient occlusion, and soft shadows. It accumulates samples over successive frames, converging on a *static* scene.

> **Path tracing does affect the eval cameras.** It applies to the tiled `TiledCamera` sensors used for observations and viewport video, not just standalone captures — verified on a settled `num_envs=1` scene, where the tiled viewport output differs from realtime in a GI-consistent way (contact shadows, ambient occlusion, material response), not as noise. Two caveats remain: (1) PT accumulates samples on a *static* scene, so during a moving rollout — where the arm and objects change every frame — per-frame convergence drops and frames can be noisier than a settled capture; (2) it is substantially slower than realtime. So realtime stays the default and recommended renderer for large-N evaluation, while path tracing suits higher-fidelity or slow/static captures. The cleanest path-traced stills come from the standalone-viewport tools `misc/compare_renderers.py` and `misc/capture_scene_image.py`, which accumulate on a frozen scene.

`--rendering-type` is a quality preset for the realtime renderer only. The resolved renderer is recorded in every run's `env_cfg.json` (top-level `renderer`, plus `sim.render.carb_settings["/rtx/rendermode"]`) for provenance. For the underlying settings, see the [Omniverse RTX Renderer documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html).

**Examples:**

```bash
# Run all benchmark tasks
python policies/pi0_family/run.py --headless

# Run specific tasks
python policies/pi0_family/run.py --task BananaInBowlTask RubiksCubeTask

# Run tasks by tag
python policies/pi0_family/run.py --tag pick_place

# Run 20 parallel episodes
python policies/pi0_family/run.py --headless --num_envs 20

# If 20 envs don't fit in GPU memory, split into runs:
python policies/pi0_family/run.py --headless --num_envs 10 --num-runs 2

# Use a different policy backend (each lives under policies/<policy>/run.py)
python policies/gr00t/run.py --remote-host 10.0.0.1 --remote-port 5555

# Expose only one instruction field
python policies/pi0_family/run.py --task BananaInBowlTask --instruction-type vague

# Expose multiple fields to a multimodal policy
python policies/pi0_family/run.py --task BananaInBowlTask --instruction-type vague video

# Render with the path tracer (standalone viewport only; see Renderer Selection)
python policies/pi0_family/run.py --task BananaInBowlTask --renderer pathtracing

# Resume a previous run (skips completed episodes automatically)
python policies/pi0_family/run.py --output-folder-name 2026-01-24_15-35-59_pi05
```

**Runtime estimate:** Each task runs up to 900 steps. With policy inference overhead, expect roughly **~15 min per task per run** on a single GPU (at ~1–2 it/s depending on `num_envs` and policy backend). For the full 120-task benchmark this works out to approximately **~20–30 hours per 100 tasks** on a single machine. Tasks that succeed early terminate faster, so actual runtime is usually on the lower end.

## Evaluation Features

The built-in `run_eval.py` provides several features out of the box:

- **Resumability** — If you provide `--output-folder-name` pointing to an existing run, completed tasks and episodes are automatically skipped. This makes it safe to restart interrupted evaluations.
- **Trajectory metrics** — At the end of each episode, trajectory metrics (SPARC smoothness, path length, speed, joint tracking error) are computed from the HDF5 data and written directly into `episode_results.jsonl`. See [Data Storage — Episode Results](data.md#episode-results) for the full list.
- **Error event extraction** — Error events (wrong object grabbed, gripper hit table, object dropped, etc.) are extracted from the episode log and recorded in the results.
- **Video recording** — Two videos per episode: observation camera view and viewport camera view, saved to the task output directory.
- **Subtask tracking** — On by default: subtask completion scores and reasons are recorded per episode (`score`/`reason` in `episode_results.jsonl`, synced with the HDF5 `subtask/score` dataset and the per-env event log). Disable with `--disable-subtask`.
- **Result summarization** — After all tasks complete, a summary table is printed. For more detailed analysis, see [Analysis and Results Parsing](analysis.md).

## Ground-Truth State Export

With `--enable-gt-state`, the episode loop attaches privileged simulator state to the observation dict once per environment step (the policy rate — after decimated physics substeps), keyed per env:

```python
obs["gt_state"] = {env_id: state}   # one entry per active env
```

Clients read their own env's entry (the `InferenceClient._get_env_gt_state` helper), so multi-env evaluations stay independent. The per-env `state` is a raw snapshot produced by `GroundTruthStateExporter` (`robolab/eval/gt_state.py`); everything is plain numpy and msgpack-serialisable:

| Key | Contents |
|-----|----------|
| `objects.<name>.pos` | `(3,)` float32, **env-local frame** (world minus env origin), meters |
| `objects.<name>.quat` | `(4,)` float32, **world frame**, `(w, x, y, z)` |
| `objects.<name>.vel` | `(6,)` float32, world-frame linear (m/s) + angular (rad/s) |
| `robot.ee_pos` / `robot.ee_quat` | End-effector pose, same conventions as above |
| `robot.gripper_closedness` | float32 in `[0, 1]`, 0 = open, 1 = closed |
| `robot.objects_in_contact` | `list[str]` — objects touching the gripper per the contact sensor |
| `subtask.completed` / `.total` / `.score` / `.info` | Live subtask-recorder tracking for this env |
| `subtask.conditions` | Per-object condition satisfaction rows for the current subtask |
| `subtask.object_completed` | `{object: bool}` across the current and all past subtasks |
| `subtask.all_subtask_conditions` | `{"subtask_i": bool}` — every subtask's conditions re-evaluated against the live sim each step |
| `scene_objects` | Manipulable object names (the task's `contact_object_list` minus fixtures) |
| `step` | Environment steps since episode start |

One object entry exists per element of the task's `contact_object_list` (minus the `table`/`robot` fixtures). Objects are covered per env — each env's entry reflects that env's own poses, contacts, and subtask progress.

The snapshot is deliberately raw: derived quantities (lift thresholds, grasp heuristics) are the consumer's business, computed client-side from the per-step stream; see [Evaluating a New Policy — Ground-Truth State in Your Client](policy.md#optional-ground-truth-state-in-your-client).

**Ground truth through the observation pipeline.** Independently of `--enable-gt-state`, environments can be registered with `object_state_obs=True` (an argument to `auto_register_droid_envs`, not a CLI flag) to add an `object_state_obs` observation group with `<object>_pos` / `<object>_quat` / `<object>_vel` terms per task — same frame conventions, but batched, recorded to HDF5, and replayable like any other observation. Useful for consumers that want privileged object state inside the standard pipeline rather than on the inference side-channel.

## Robustness Evaluation Scripts

Additional evaluation scripts test policy robustness under visual or physical variations:

| Script | Description |
|--------|-------------|
| `policies/pi0_family/run_lighting.py` | Evaluate with lighting intensity, shadow, and color variations |
| `policies/pi0_family/run_background_variation.py` | Evaluate with different HDR background scenes |
| `policies/pi0_family/run_camera_pose_variation.py` | Evaluate with random camera pose perturbations |
| `policies/pi0_family/run_table_variation.py` | Evaluate with different table materials (oak, maple, bamboo, black) |

## Example Scripts

| Script | Description |
|--------|-------------|
| [`examples/run_empty.py`](../examples/run_empty.py) | Environment initialization test with random actions (no policy server needed) |
| [`policies/pi0_family/run.py`](../policies/pi0_family/run.py) | Full multi-task policy evaluation (supports multi-env) |

## See Also

- [Environment Registration](environment_registration.md) — How to register tasks as runnable environments
- [Evaluating a New Policy](policy.md) — Implementing an inference client for your model
- [Data Storage and Output](data.md) — Output directory structure, HDF5 layout, and result fields
- [Analysis and Results Parsing](analysis.md) — Scripts for summarizing experiment results
