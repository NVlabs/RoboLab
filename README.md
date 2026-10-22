<h1><picture><source media="(prefers-color-scheme: dark)" srcset="docs/images/ribble.gif"><img src="docs/images/ribble-dark.gif" alt="" height="42" align="absmiddle" /></picture> RoboLab</h1>

[🌐 Website](https://research.nvidia.com/labs/srl/projects/robolab) · [📄 Paper](https://arxiv.org/abs/2604.09860) · [🏆 Leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html)

**RoboLab** is a task-based evaluation benchmark for robot manipulation policies built on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab). It provides 100+ manipulation tasks with automated success detection, a server-client policy architecture, and multi-environment parallel evaluation — designed for reproducible, large-scale benchmarking of generalist robot policies in simulation.

<h4 align="center"><a href="assets/objects/README.md">Objects</a> · <a href="assets/scenes/README.md">Scenes</a> · <a href="robolab/tasks/README.md">Tasks</a> · <a href="robolab/robots/README.md">Robots</a> · <a href="policies/README.md">Policy Clients</a></h4>

<div align="center">
  <img src="docs/images/robolab.png" alt="RoboLab Overview" width="800"/>
</div>

## News

- **[2026/10/18]** 🔥 Added [**Astranomy**](https://research.nvidia.com/labs/srl/projects/astranomy/) policy code, using general-purpose models (such as GPT-6 Astra) as a native robot policy.
- **[2026/10/13]** 🔥 [**LADDER**](https://ladder.insait.ai/) introduces RoboLab-Long, a suite of 44 tasks for evaluating long-horizon manipulation, as part of a 145-task benchmark spanning three simulation suites. See [Task Libraries](docs/ecosystem.md#task-libraries).
- **[2026/08/19]** 🤖 [**DROID+**](https://github.com/NVlabs/DroidPlus) is released: A simple DROID hardware stack. Your RoboLab policy setup will also work with DROID+.
- **[2026/08/07]** 🔥 [**RoboVoLo**](https://github.com/NVlabs/RoboVoLo) is released — a task library from the [VoLo](https://chicychen.github.io/VoLo/) project extending RoboLab with long-horizon and reasoning-heavy manipulation tasks, evaluated via the bundled [VoLo policy backend](policies/volo/README.md).

For all updates, including minor improvements and bug fixes, see the [Changelog](CHANGELOG.md).

## Key Features

- **RoboLab-120**: An initial set of 120 brand new benchmark [tasks](robolab/tasks/README.md) spanning pick-and-place, stacking, rearrangement, tool use, and more — each with language instructions and automated success/failure detection via composable predicates.
- **Bring your own robot**: Tasks are not tied to a specific robot embodiment, so you can plug in any robot compatible with IsaacLab!
- **Rich Asset Libraries**: See a list of [objects](assets/objects/README.md), [scenes](assets/scenes/README.md), and curated [backgrounds](assets/backgrounds/README.md) — everything you need to create new scenes and new tasks for your own evaluation needs.
- **AI-Enabled Workflows**: Generate new scenes and tasks **in minutes** using natural language with the [/robolab-scenegen](skills/robolab-scenegen/) and [/robolab-taskgen](skills/robolab-taskgen/) Claude Code skills.
- **Multi-Environment Parallel Evaluation**: Run multiple episodes in parallel across environments with vectorized conditionals and per-environment termination.
- **Results Dashboard with Episode Videos and Cross-Experiment Analysis**: A self-contained web [dashboard](docs/dashboard.md) for browsing scenes/tasks, replaying episode videos, and comparing results across experiments.

See the [Ecosystem](docs/ecosystem.md) page for projects built on RoboLab.

## Getting Started

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and a system `ffmpeg` (used for video recording). The Isaac Sim/Isaac Lab stack is selected with a mutually-exclusive extra: `isaac50` (the default), `isaac51`, or `isaac60`. See [Isaac Sim 6 support](docs/isaac_sim_6.md) and [Requirements](#requirements).

### Installation

```bash
sudo apt install ffmpeg
git clone <repo_url>
cd robolab
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra isaac50          # Isaac Sim 5.0 / Isaac Lab 2.2.0 (default)
# uv sync --extra isaac51        # Isaac Sim 5.1 / Isaac Lab 2.3.2.post1 (Python 3.11)
# uv sync --python 3.12 --extra isaac60  # Isaac Sim 6.0.1 / Isaac Lab 3.0 beta
```

The stacks cannot coexist in one environment. To keep multiple versions available, install each into its own venv:

```bash
UV_PROJECT_ENVIRONMENT=.venv-50 uv sync --python 3.11 --extra isaac50
UV_PROJECT_ENVIRONMENT=.venv-51 uv sync --python 3.11 --extra isaac51
UV_PROJECT_ENVIRONMENT=.venv-60 uv sync --python 3.12 --extra isaac60
```

Verify installation:
```bash
uv run pytest tests/
```

This runs the install-verification suite end-to-end: isaaclab importable, all task definitions valid, env factory populated, one full episode runs. The suite auto-accepts the NVIDIA Omniverse EULA so the run is fully headless with no prompts. More details at [Debugging → Diagnostic Scripts](docs/debug.md#diagnostic-scripts).

> **Running without activating the venv**: if you don't `source .venv/bin/activate`, prefix every `python` command with `uv run` (e.g. `uv run pytest tests/`).

> **EULA outside the test suite**: when running other entry points (e.g. `policies/pi0_family/run.py`) for the first time, set `export OMNI_KIT_ACCEPT_EULA=Y` once. Cached after first acceptance.

### Run without a policy

```bash
# Run an empty episode with random actions
python examples/run_empty.py --headless

# Playback recorded demonstration data
python examples/run_recorded.py --headless

# Toggle the gripper open/closed while holding the arm fixed (sanity-check
# the gripper action path; saves sensor + viewport video to
# output/run_gripper_toggle/<task>/)
python examples/run_gripper_toggle.py --task BananaInBowlTask --headless
```

> **Replay**: `run_recorded.py` restores the recorded initial state, replays the recorded actions open-loop, and by default replays with the exact env configuration saved next to the recording (`env_cfg.json`). Note that the recorded outcome is not invariant across simulator versions — contact dynamics evolve between IsaacSim/IsaacLab releases (see [Requirements](#requirements)) — and faithful reproduction requires recording and replaying with a single env. See **[Replaying Recorded Episodes](docs/replay.md)** for the full guide, including replaying your own recordings, `--env-config`, and `--validate-states`.

### Run with a policy

RoboLab uses a **server-client architecture**: your model runs as a standalone server, and RoboLab connects to it via a lightweight inference client. To quickly test RoboLab, try [Pi0.5 via OpenPI](policies/pi0_family/README.md).

Quick run after install in the RoboLab terminal, to see it working:

```bash
cd robolab
uv run python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask --num-envs 10
```
Use the [dashboard](#dashboard) to view the output written to your local folder.

### Common CLI Options

```bash
# Run headlessly
python policies/pi0_family/run.py --policy pi05 --headless

# Run on specific tasks (these two are good for sanity checking)
python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask RubiksCubeAndBananaTask

# Run on a tag of tasks
python policies/pi0_family/run.py --policy pi05 --tag semantics

# Run 12 parallel episodes per task
python policies/pi0_family/run.py --policy pi05 --headless --num-envs 12

# Disable subtask progress tracking (on by default; drops score/reason from results)
python policies/pi0_family/run.py --policy pi05 --disable-subtask

# Resume a previous run (skips completed episodes)
python policies/pi0_family/run.py --policy pi05 --output-folder-name my_previous_run
```

## Example Tasks

See the full [Benchmark Task Library](robolab/tasks/README.md) for all 120 tasks.

<div align="center">
  <img src="docs/images/Make_sure_all_the_white_mugs_are_upright_so_that_the_opening_is_facing_upwards_0_hstack_3X_fps24_width800.gif" alt="Make sure all the white mugs are upright so that the opening is facing upwards" width="800"/>
  <br><em>"Make sure all the white mugs are upright so that the opening is facing upwards."</em>
</div>

<div align="center">
  <img src="docs/images/Put_all_plastic_bottles_away_in_the_bin_3_hstack_3X_fps24.gif" alt="Put all plastic bottles away in the bin" width="800"/>
  <br><em>"Put all plastic bottles away in the bin."</em>
</div>

<div align="center">
  <img src="docs/images/Put_the_orange_measuring_cup_and_the_blue_measuring_cup_outside_of_the_plate_0_hstack_3X_fps24_width800.gif" alt="Put the orange measuring cup and the blue measuring cup outside of the plate" width="800"/>
  <br><em>"Put the orange measuring cup and the blue measuring cup outside of the plate."</em>
</div>

## Dashboard

A self-contained web dashboard for browsing the benchmark (scenes and tasks) and analyzing your experiment results.

```bash
uv run robolab-dashboard
# open http://localhost:8080
```

<video src="https://github.com/user-attachments/assets/5992e61b-9043-4602-8402-04459da38421" autoplay controls muted loop playsinline width="800">
  Your viewer doesn't render inline video — see <a href="https://github.com/user-attachments/assets/5992e61b-9043-4602-8402-04459da38421">robolab_dashboard.mp4</a>.
</video>

See **[docs/dashboard.md](docs/dashboard.md)** for the full feature tour, CLI
flags, and the API endpoints under the hood.

## Documentation

Full documentation is at **[docs/README.md](docs/README.md)**, covering:

- [Objects](docs/objects.md), [Scenes](docs/scene.md), [Tasks](docs/task.md) — Creating and managing assets and benchmark tasks
- [Robots](docs/robots.md), [Cameras](docs/camera.md), [Lighting](docs/lighting.md), [Backgrounds](docs/background.md) — Configuring simulation parameters
- [Environment Registration](docs/environment_registration.md) — Combining tasks with robot/observation/action configs
- [Inference Clients](policies/README.md) — A list of supported open-source models and clients
- [Replaying Recorded Episodes](docs/replay.md) — Playing back recorded HDF5 episodes faithfully
- [Analysis and Results](docs/analysis.md) — Summarizing, comparing, and auditing results
- [Dashboard](docs/dashboard.md) — Interactive web viewer for benchmark, tasks, scenes, and eval results
- [Subtask Checking](docs/subtask.md), [Conditionals](docs/task_conditionals.md), [Event Tracking](docs/event_tracking.md)
- [Ecosystem](docs/ecosystem.md) — Task libraries and projects built on RoboLab

## Requirements

| Dependency | Version |
|---|---|
| Isaac Sim | 5.0 (default), 5.1, or 6.0.1 |
| Isaac Lab | 2.2.0 (default), 2.3.2.post1, or 3.0 beta 2 patch 1 |
| Python | 3.12 for Isaac Sim 6; 3.11 for Isaac Sim 5 |
| Linux | Ubuntu 22.04+ |

> **Note on simulator versions**: the stacks ship different PhysX builds, so contact-rich dynamics (grasping, object settling) are not invariant across versions. Benchmark results are best compared on the same stack. RoboLab converts recorded quaternion layouts across versions, but numerical replay fidelity still requires the original simulator stack.

- **Disk space**: ~8 GB (assets account for ~7 GB)
- **GPU**: NVIDIA RTX GPU required. Recommend 48GB+ VRAM. See [Isaac Lab's hardware requirements](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#system-requirements) for recommended GPUs and VRAM.
- **Speed**: 30 GPU hours / 100 tasks, 1.4 it/s (assuming ~200ms inference step)

## License

The RoboLab framework is released under the [Apache License 2.0](./LICENSE).

Third-party dependency licenses are listed in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{yang2026robolab,
    author    = {Jenai Xuning Yang and Rishit Dagli and Alex Zook and Hugo Hadfield and Ankit Goyal and Stan Birchfield and Fabio Ramos and Jonathan Tremblay},
    title     = {{RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies}},
    booktitle = {Proceedings of Robotics: Science and Systems},
    year      = {2026},
    address   = {Sydney, Australia},
    month     = {July},
    url       = {https://arxiv.org/abs/2604.09860}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for acknowledgements, issues, and how to contribute.
