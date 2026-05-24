# RoboLab Documentation

## How RoboLab Works

RoboLab dynamically combines **tasks** with user-specified **robot**, **observations**, **actions**, and **simulation parameters** at environment registration time.

## Terminology

| Term | Meaning |
|------|---------|
| **scene** | A USD/USDA file describing the static contents of a workspace — objects, fixtures, table, and their spatial layout. Reusable across tasks. See [Scenes](scene.md). |
| **task** | A `Task` dataclass binding a scene to a language instruction, termination criteria, and (optional) subtasks. See [Tasks](task.md). |
| **environment** | A task combined with robot, camera, lighting, background, and simulation configs, registered as a Gymnasium env. `--num-envs N` spawns `N` parallel instances in a grid, each indexed by `env_id`. See [Environment Registration](environment_registration.md). |
| **episode** | One trajectory from one instance of an environment from reset to termination. |
| **run** | One sequential pass over all environments (one reset → step loop → termination → `end_episode` cycle). If running with `--num-envs N`, then each run produces `N` episodes.|


The core concepts are:

#### Objects, Scenes, Tasks
- **[Objects](objects.md)** — USD object assets with physics properties for manipulation
- **[Scenes](scene.md)** — USD-based environments containing objects, fixtures, and spatial layout
- **[Tasks](task.md)** — Language instructions, termination criteria, and scene bindings
- **[Task Libraries](task_libraries.md)** — Managing task collections, generating metadata, and viewing statistics
#### Task Conditionals
- **[Subtask Checking](subtask.md)** — Granular progress tracking within tasks
- **[Conditionals](task_conditionals.md)** — Predicate logic for defining success/failure conditions
- **[Event Tracking](event_tracking.md)** — Monitoring task-relevant events during execution
#### Variations
- **[Robots](robots.md)** — Robot articulation configs, actuators, and action spaces
- **[Cameras](camera.md)** — Scene cameras and robot-attached cameras
- **[Lighting](lighting.md)** — Scene lighting (sphere, directional, and custom lights)
- **[Backgrounds](background.md)** — HDR/EXR dome light backgrounds
#### Environments
- **[Environment Registration](environment_registration.md)** — How tasks are combined with robot/observation/action configs into runnable Gymnasium environments
- **[Environment Generation](environment_generation.md)** — Contact sensor creation, subtask trackers, and runtime environment internals
- **[Running Environments](environment_run.md)** — Creating environments, evaluation scripts, CLI reference, and robustness testing
- **[`num_envs` VRAM size guide](env_vram_size_guide.md)** — Per-task `num_envs` ceiling on L40, measured against pi05
#### Policy
- **[Inference Clients](inference.md)** — Built-in policy clients and server setup instructions (OpenPI, GR00T)
#### Output
- **[Data Storage and Output](data.md)** — Output directory structure, HDF5 layout, and episode result fields
- **[Analysis and Results Parsing](analysis.md)** — Scripts for summarizing, comparing, and auditing experiment results
#### Debug
- **[Debugging](debug.md)** — Verbose/debug flags, world state inspection, and diagnostic scripts
- **[Known Issues](known_issues.md)** — Documented bugs and workarounds


## Developing and Working with RoboLab

If you're building a new benchmark and a new experiment workflow, follow the steps below in order.
Otherwise, pick whichever applies to your use case.

### Creating new assets, tasks, and benchmarks

1. **[Creating New Objects](objects.md)** — Author USD object assets with rigid body, collision, and friction properties. Includes pipeline for catalog generation, screenshots, and physics tuning.
2. **[Creating New Scenes](scene.md)** — Compose objects into USD scenes using IsaacSim. Includes settling, metadata generation, and screenshot utilities.
3. **[Creating New Tasks](task.md)** — Define task dataclasses with language instructions, termination criteria, and scene bindings. Tasks can live in your own repository.
4. **[Managing Task Libraries](task_libraries.md)** — Organize tasks into collections, generate metadata (JSON, CSV, README), and compute statistics.

### Configuring robots, cameras, lighting, and backgrounds

- **[Robots](robots.md)** — Define or customize robot articulation, actuators, and action spaces. Use built-in configs (DROID, Franka) or bring your own from IsaacLab.
- **[Cameras](camera.md)** — Set up scene cameras and robot-attached cameras (e.g., wrist cameras).
- **[Lighting](lighting.md)** — Configure scene lighting for evaluation or robustness testing.
- **[Backgrounds](background.md)** — Set HDR/EXR dome light backgrounds for realistic scene rendering.

### Evaluating a new policy against the benchmark

1. **[Setting Up Environment Registration](environment_registration.md)** — Register tasks with your robot/observation/action/simulation settings. For DROID with joint-position actions, the built-in registration can be used directly.
2. **[Evaluating a New Policy](policy.md)** — Implement an inference client for your model and run multi-task evaluation. Everything can live in your own separate repository.

### Analysis

1. **[Statistical Significance of Results](statistical_significance.md)** — Discussion on how to run evaluations such that your results are statistically significant.

### Browsing the benchmark and eval results

- **[Dashboard](dashboard.md)** — Self-contained web viewer for scenes, tasks, and eval outputs. Runs locally with `robolab-dashboard --output-dir output/`; binds `0.0.0.0` so anyone on your LAN can hit your IP.

### AI Workflows

- **[Scene Generation](scene.md#ai-workflows-scene-generation)** — Generate USD scenes from natural language using the `/robolab-scenegen` Claude Code skill. See [`skills/robolab-scenegen/`](../skills/robolab-scenegen/).
- **[Task Generation](task.md#ai-workflows-task-generation)** — Generate task files from natural language using the `/robolab-taskgen` Claude Code skill. See [`skills/robolab-taskgen/`](../skills/robolab-taskgen/).
