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
| `instruction_type` | `str` | `"default"` | Instruction variant to use when the task defines multiple variants |
| `policy` | `str` | `None` | Policy backend name, stored on `env_cfg` for downstream use |
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

## Instruction Type Selection

When a task defines multiple instruction variants (see [task.md — Instruction Variants](task.md#3-instruction-variants)), you can select which variant to use at runtime:

```python
env, env_cfg = create_env(
    "BananaInBowlTask",
    device="cuda:0",
    num_envs=1,
    instruction_type="vague",
)

print(env_cfg.instruction)             # e.g., "Put stuff in the bowl"
print(env_cfg._instruction_variants)   # The raw instruction field (dict or str)
```

When `instruction_type` is not `"default"`, the type name is appended to the run name for distinguishable output files (e.g., `MyTask_vague_0.mp4`) and recorded in the episode results as `"instruction_type"`.

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
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs
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

**Choosing `num_envs`:** The number of parallel environments is limited by GPU VRAM. As a rough guide, a single DROID tabletop environment with one camera uses ~1–2 GB; scale accordingly. If you run out of memory, reduce `num_envs` and increase `--num-runs` to reach the same total episode count:

```bash
# 20 episodes total: 20 envs × 1 run (if VRAM allows)
python examples/policy/run_eval.py --headless --num_envs 20

# 20 episodes total: 10 envs × 2 runs (if 20 doesn't fit)
python examples/policy/run_eval.py --headless --num_envs 10 --num-runs 2
```

**Multi-env episode handling:** With `num_envs > 1`, each environment runs an independent episode. The built-in `run_eval.py` handles per-env termination, video recording, and result logging automatically. If you are writing a custom evaluation loop, see `robolab/eval/episode.py` for the multi-env episode runner pattern (`from robolab.eval import run_episode`), which manages per-env video writers, independent termination tracking, and batched policy inference.

## Initial Condition Randomization

To randomize object poses between episodes:

1. Load the env config explicitly using `parse_env_cfg`
2. Add your desired randomization as an `events` parameter
3. Pass the modified config to `create_env`

```python
from robolab.core.environments.runtime import create_env
from robolab.core.environments.config import parse_env_cfg
from robolab.core.events.reset_pose import RandomizeInitPoseUniform

env_cfg = parse_env_cfg("BananaInBowlTask",
    device="cuda:0", seed=42, num_envs=1, use_fabric=True)

env_cfg.events = RandomizeInitPoseUniform.from_params(
    objects=["banana", "bowl"],
    pose_range={"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
)

env, env_cfg = create_env(env_cfg,
    device="cuda:0", seed=42, num_envs=1, use_fabric=True)

# Between episodes, reset() will re-randomize poses according to the seed
```

You can also pass events directly to `create_env` using the `events` parameter. This is not encouraged, however.

```python
from robolab.core.events.reset_camera import RandomizeCameraPoseUniform

events = RandomizeCameraPoseUniform.from_params(
    cameras=["external_cam"],
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
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs
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

The built-in `examples/policy/run_eval.py` supports the full set of evaluation features:

| Flag | Description | Default |
|------|-------------|---------|
| `--task TASK [TASK ...]` | Specific task name(s) to evaluate | all registered tasks |
| `--tag TAG [TAG ...]` | Evaluate all tasks with the given tag(s) | `None` |
| `--task-dirs DIR [DIR ...]` | Task directories to scan for registration | `DEFAULT_TASK_SUBFOLDERS` |
| `--policy BACKEND` | Policy backend (`pi0`, `pi0_fast`, `pi05`, `gr00t`, etc.) | `pi05` |
| `--num_envs N` | Number of parallel environments (= number of episodes per task) | `1` |
| `--num-runs N` | Number of sequential runs (default 1). Only increase if GPU memory limits `num_envs`. Total episodes = `num_runs * num_envs`. | `1` |
| `--instruction-type TYPE` | Instruction variant (`default`, `vague`, `specific`, etc.) | `default` |
| `--remote-host HOST` | Policy server host | `localhost` |
| `--remote-port PORT` | Policy server port | `8000` |
| `--output-folder-name NAME` | Output folder under `output/`. Reusing a previous folder skips completed episodes. | `<timestamp>_<policy>` |
| `--enable-subtask` | Enable subtask progress checking (records score, reason, subtask log) | `False` |
| `--record-image-data` | Record image observations to HDF5 | `False` |
| `--video-mode MODE` | Which videos to save: `all` (sensor + viewport), `viewport` only, `sensor` only, or `none` | `all` |
| `--headless` | Run without live display window. **Recommended for multi-task runs** — see [GPU VRAM leak in non-headless mode](debug.md#gpu-vram-leak-in-non-headless-mode-across-environment-reloads) | `False` |
| `--enable-verbose` | Verbose output | `False` |
| `--enable-debug` | Debug output | `False` |

**Examples:**

```bash
# Run all benchmark tasks
python examples/policy/run_eval.py --headless

# Run specific tasks
python examples/policy/run_eval.py --task BananaInBowlTask RubiksCubeTask

# Run tasks by tag
python examples/policy/run_eval.py --tag pick_place

# Run 20 parallel episodes with subtask tracking
python examples/policy/run_eval.py --headless --num_envs 20 --enable-subtask

# If 20 envs don't fit in GPU memory, split into runs:
python examples/policy/run_eval.py --headless --num_envs 10 --num-runs 2 --enable-subtask

# Use a different policy backend
python examples/policy/run_eval.py --policy gr00t --remote-host 10.0.0.1 --remote-port 5555

# Use a specific instruction variant
python examples/policy/run_eval.py --task BananaInBowlTask --instruction-type vague

# Resume a previous run (skips completed episodes automatically)
python examples/policy/run_eval.py --output-folder-name 2026-01-24_15-35-59_pi05
```

**Runtime estimate:** Each task runs up to 900 steps. With policy inference overhead, expect roughly **~15 min per task per run** on a single GPU (at ~1–2 it/s depending on `num_envs` and policy backend). For the full 120-task benchmark this works out to approximately **~20–30 hours per 100 tasks** on a single machine. Tasks that succeed early terminate faster, so actual runtime is usually on the lower end.

## Evaluation Features

The built-in `run_eval.py` provides several features out of the box:

- **Resumability** — If you provide `--output-folder-name` pointing to an existing run, completed tasks and episodes are automatically skipped. This makes it safe to restart interrupted evaluations.
- **Trajectory metrics** — At the end of each episode, trajectory metrics (SPARC smoothness, path length, speed, joint tracking error) are computed from the HDF5 data and written directly into `episode_results.jsonl`. See [Data Storage — Episode Results](data.md#episode-results) for the full list.
- **Error event extraction** — Error events (wrong object grabbed, gripper hit table, object dropped, etc.) are extracted from the episode log and recorded in the results.
- **Video recording** — Two videos per episode: observation camera view and viewport camera view, saved to the task output directory.
- **Subtask tracking** — With `--enable-subtask`, subtask completion scores and reasons are recorded per episode.
- **Result summarization** — After all tasks complete, a summary table is printed. For more detailed analysis, see [Analysis and Results Parsing](analysis.md).

## Robustness Evaluation Scripts

Additional evaluation scripts test policy robustness under visual or physical variations:

| Script | Description |
|--------|-------------|
| `examples/policy/run_eval_lighting.py` | Evaluate with lighting intensity, shadow, and color variations |
| `examples/policy/run_eval_background_variation.py` | Evaluate with different HDR background scenes |
| `examples/policy/run_eval_camera_pose_variation.py` | Evaluate with random camera pose perturbations |
| `examples/policy/run_eval_table_variation.py` | Evaluate with different table materials (oak, maple, bamboo, black) |

## Example Scripts

| Script | Description |
|--------|-------------|
| [`examples/demo/run_empty.py`](../examples/demo/run_empty.py) | Environment initialization test with random actions (no policy server needed) |
| [`examples/policy/run_eval.py`](../examples/policy/run_eval.py) | Full multi-task policy evaluation (supports multi-env) |

## See Also

- [Environment Registration](environment_registration.md) — How to register tasks as runnable environments
- [Evaluating a New Policy](policy.md) — Implementing an inference client for your model
- [Data Storage and Output](data.md) — Output directory structure, HDF5 layout, and result fields
- [Analysis and Results Parsing](analysis.md) — Scripts for summarizing experiment results
