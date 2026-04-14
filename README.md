# RoboLab

[![Website](https://img.shields.io/badge/Website-RoboLab-blue?logo=googlechrome&logoColor=white)](https://research.nvidia.com/labs/srl/projects/robolab)
[![arXiv](https://img.shields.io/badge/arXiv-2604.09860-b31b1b?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2604.09860)

**RoboLab** is a task-based evaluation benchmark for robot manipulation policies built on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab). It provides 100+ manipulation tasks with automated success detection, a server-client policy architecture, and multi-environment parallel evaluation — designed for reproducible, large-scale benchmarking of generalist robot policies in simulation.

<div align="center">
  <img src="docs/images/robolab.png" alt="RoboLab Overview" width="800"/>
</div>

## Key Features

- **RoboLab-120**: An initial set of 120 brand new benchmark [tasks](robolab/tasks/README.md) spanning pick-and-place, stacking, rearrangement, tool use, and more — each with language instructions and automated success/failure detection via composable predicates.
- **Bring your own robot**: Tasks are not tied to a specific robot embodiment, so you can plug in any robot compatible with IsaacLab!
- **Rich Asset Libraries**: See a list of [objects](assets/objects/README.md), [scenes](assets/scenes/README.md), and curated [backgrounds](assets/backgrounds/README.md) — everything you need to create new scenes and new tasks for your own evaluation needs.
- **AI-Enabled Workflows**: Generate new scenes and tasks **in minutes** using natural language with the [/robolab-scenegen](skills/robolab-scenegen/) and [/robolab-taskgen](skills/robolab-taskgen/) Claude Code skills.
- **Multi-Environment Parallel Evaluation**: Run multiple episodes in parallel across environments with vectorized conditionals and per-environment termination.
- **Server-Client Policy Architecture**: Policy models run as standalone servers; RoboLab connects via lightweight inference clients (OpenPI, GR00T, and more).

## Getting Started

Requires [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [uv](https://docs.astral.sh/uv/getting-started/installation/). See [Requirements](#requirements) for versions and hardware.

### Installation

```bash
git clone https://github.com/NVlabs/RoboLab.git
cd robolab
uv venv --python 3.11
source .venv/bin/activate
uv pip install "setuptools<81"
uv sync
```

Verify installation:
```bash
python scripts/check_registered_envs.py
```

### Run without a policy

```bash
# Run an empty episode with random actions
python examples/demo/run_empty.py --headless

# Playback recorded demonstration data
python examples/demo/run_recorded.py --headless
```

### Run with a policy

RoboLab uses a **server-client architecture**: your model runs as a standalone server, and RoboLab connects to it via a lightweight inference client. To quickly test RoboLab, try [Pi0-5 via OpenPI](docs/inference.md#openpi-pi0--pi0-fast--pi05).

1. Start your policy server in a separate terminal.
2. Run evaluation:
   ```bash
   python examples/policy/run_eval.py --policy pi05 --task BananaInBowlTask --num-envs 12 --headless
   ```
3. Analyze results:
   ```bash
   python analysis/read_results.py output/<your_run_folder>
   ```

### Common CLI Options

```bash
# Run on specific tasks (these two are good for sanity checking)
python examples/policy/run_eval.py --policy pi05 --task BananaInBowlTask RubiksCubeAndBananaTask

# Run on a tag of tasks
python examples/policy/run_eval.py --policy pi05 --tag semantics

# Run 12 parallel episodes per task
python examples/policy/run_eval.py --policy pi05 --headless --num-envs 12

# Enable subtask progress tracking
python examples/policy/run_eval.py --policy pi05 --headless --enable-subtask

# Resume a previous run (skips completed episodes)
python examples/policy/run_eval.py --policy pi05 --output-folder-name my_previous_run
```

## Documentation

Full documentation is at **[docs/README.md](docs/README.md)**, covering:

- [Objects](docs/objects.md), [Scenes](docs/scene.md), [Tasks](docs/task.md) — Creating and managing assets and benchmark tasks
- [Robots](docs/robots.md), [Cameras](docs/camera.md), [Lighting](docs/lighting.md), [Backgrounds](docs/background.md) — Configuring simulation parameters
- [Environment Registration](docs/environment_registration.md) — Combining tasks with robot/observation/action configs
- [Inference Clients](docs/inference.md) — Built-in policy clients and server setup (OpenPI, GR00T)
- [Evaluating a New Policy](docs/policy.md) — Implementing your own inference client
- [Running Environments](docs/environment_run.md) — CLI reference and evaluation workflows
- [Analysis and Results](docs/analysis.md) — Summarizing, comparing, and auditing results
- [Subtask Checking](docs/subtask.md), [Conditionals](docs/task_conditionals.md), [Event Tracking](docs/event_tracking.md) — Advanced task features

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

## Requirements

| Dependency | Version |
|---|---|
| Isaac Sim | 5.0 |
| Isaac Lab | 2.2.0 |
| Python | 3.11 |
| Linux | Ubuntu 22.04+ |

- **Disk space**: ~8 GB (assets account for ~7 GB)
- **GPU**: NVIDIA RTX GPU required. Recommend 48GB+ VRAM. See [Isaac Lab's hardware requirements](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#system-requirements) for recommended GPUs and VRAM.
- **Speed**: 30 GPU hours / 100 tasks, 1.4 it/s (assuming ~200ms inference step)

## License

The RoboLab framework is released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation

```bibtex
@misc{yang2026robolab,
      title={RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies},
      author={Xuning Yang and Rishit Dagli and Alex Zook and Hugo Hadfield and Ankit Goyal and Stan Birchfield and Fabio Ramos and Jonathan Tremblay},
      year={2026},
      url={https://arxiv.org/abs/2604.09860},
}
```
